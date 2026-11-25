//! Checks per-layer calibration Gram taps against the CPU interpreter.

use tessera_graph::{CalibTap, Composite, Op, Role};
use tessera_interp::{Tensor, run};
use tessera_lower::{cut, cut_region};
use tessera_models::{Stack, mla, qwen4_exp, qwen35};
use tessera_testkit::{Noise, weights};

fn cut_weights(
    full: &tessera_graph::Graph,
    sub: &tessera_graph::Graph,
    all: &[Tensor],
) -> Vec<Tensor> {
    let mut out = Vec::with_capacity(sub.weights.len());
    for w in &sub.weights {
        let at = full
            .weights
            .iter()
            .position(|fw| fw.path == w.path)
            .expect("cut weight exists in full graph");
        out.push(all[at].clone());
    }
    out
}

fn toy_tokens(seqs: usize, ctx: usize, vocab: u64) -> (Tensor, Tensor) {
    let ids: Vec<u32> = (0..seqs * ctx)
        .map(|i| ((i * 7 + 3) as u64 % vocab) as u32)
        .collect();
    let pos: Vec<u32> = (0..seqs).flat_map(|_| 0..ctx as u32).collect();
    (
        Tensor::u32(vec![seqs, ctx], ids),
        Tensor::u32(vec![seqs, ctx], pos),
    )
}

#[test]
fn mla_calib_taps() {
    let mut cfg = mla::Config::deepseek_small();
    cfg.stack.layers = 2;
    cfg.dense_layers = 1;
    let g = mla::calib_program::<mla::Moe>(&cfg);
    let w = weights(&g, &mut Noise(7));
    let (tok, pos) = toy_tokens(2, 4, cfg.stack.vocab);
    let full_outs = run(&g, &w, 0, &[tok.clone(), pos.clone()]).unwrap();
    let entry = &g.entries[0];

    let embed_id = g
        .nodes
        .iter()
        .enumerate()
        .find_map(|(i, n)| matches!(n.op, Op::Select(_)).then_some(tessera_graph::NodeId(i as u32)))
        .expect("embed node");
    let emb_g = cut(&g, &[embed_id]);
    let mut hidden = run(&emb_g, &[], 0, &[w[0].clone(), tok]).unwrap()[0].clone();

    let layers: Vec<_> = g
        .regions
        .iter()
        .filter(|r| r.kind == Composite::Layer)
        .collect();
    assert_eq!(layers.len(), 2);

    let want_roles = [
        vec![
            CalibTap::AttnIn,
            CalibTap::QResid,
            CalibTap::KvLatent,
            CalibTap::OIn,
            CalibTap::MlpIn,
            CalibTap::DenseDownIn,
        ],
        vec![
            CalibTap::AttnIn,
            CalibTap::QResid,
            CalibTap::KvLatent,
            CalibTap::OIn,
            CalibTap::MlpIn,
            CalibTap::RouterIds,
            CalibTap::RouterGates,
            CalibTap::RoutedInner,
            CalibTap::SharedDownIn,
        ],
    ];

    let mut full_cursor = 1usize;
    for (l, region) in layers.iter().enumerate() {
        let lg = cut_region(&g, region);
        let lw = cut_weights(&g, &lg, &w);
        let mut feeds = Vec::new();
        for port in &lg.entries[0].inputs {
            match port.role {
                Role::Positions => feeds.push(pos.clone()),
                _ => feeds.push(hidden.clone()),
            }
        }
        let l_outs = run(&lg, &lw, 0, &feeds).unwrap();
        hidden = l_outs[0].clone();

        let got_taps: Vec<CalibTap> = lg.entries[0].outputs[1..]
            .iter()
            .map(|o| match o.role {
                Role::Calib(t) => t,
                other => panic!("unexpected role {other:?}"),
            })
            .collect();
        assert_eq!(got_taps, want_roles[l], "layer {l} tap roles");

        for (k, out) in l_outs[1..].iter().enumerate() {
            assert_eq!(
                entry.outputs[full_cursor + k].role,
                lg.entries[0].outputs[1 + k].role
            );
            assert_eq!(
                out.bytes(),
                full_outs[full_cursor + k].bytes(),
                "layer {l} tap {k} exact match"
            );
        }
        full_cursor += want_roles[l].len();
    }
    assert_eq!(full_cursor, full_outs.len());
}

#[test]
fn qwen4_calib_taps() {
    let cfg = qwen4_exp::Config {
        stack: Stack::sized([64, 32, 2, 2, 16, 16], 2, 8),
        kv_heads: 2,
        rotary: 8,
        rope_theta: 1e4,
        eps: 1e-6,
        hc_count: 2,
        hc_rank: 4,
        experts: 4,
        top_k: 2,
        shared_ffn: 16,
        linear: qwen35::Recurrent {
            key_heads: 2,
            value_heads: 2,
            key_dim: 8,
            value_dim: 8,
            window: 4,
        },
        layers: vec![qwen35::Mixing::Linear, qwen35::Mixing::Full],
        ple: None,
        eos_token: 0,
        tied: false,
    };
    let g = qwen4_exp::calib_program::<qwen4_exp::Moe>(&cfg);
    let w = weights(&g, &mut Noise(11));
    let (tok, pos) = toy_tokens(1, 4, cfg.stack.vocab);
    let full_outs = run(&g, &w, 0, &[tok, pos]).unwrap();
    let layers: Vec<_> = g
        .regions
        .iter()
        .filter(|r| r.kind == Composite::Layer)
        .collect();
    assert_eq!(layers.len(), cfg.layers.len());
    let want = [
        CalibTap::MlpIn,
        CalibTap::RouterIds,
        CalibTap::RouterGates,
        CalibTap::RoutedInner,
        CalibTap::SharedDownIn,
    ];
    let mut cursor = 1usize;
    for (l, region) in layers.iter().enumerate() {
        let lg = cut_region(&g, region);
        let got: Vec<CalibTap> = lg.entries[0].outputs[1..]
            .iter()
            .map(|o| match o.role {
                Role::Calib(t) => t,
                other => panic!("unexpected role {other:?}"),
            })
            .collect();
        assert_eq!(got, want, "qwen4 layer {l} taps");
        for k in 0..want.len() {
            assert_eq!(g.entries[0].outputs[cursor + k].role, Role::Calib(want[k]));
            assert!(!full_outs[cursor + k].bytes().is_empty());
        }
        cursor += want.len();
    }
    assert_eq!(cursor, full_outs.len());
}
