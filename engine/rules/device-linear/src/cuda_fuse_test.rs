use super::device_test::push_node;
use tessera_gpukit::backend::{Backend, picked};
use tessera_graph::DimName::Rows;
use tessera_graph::{Act, DType, Dim, F16, Graph, NodeId, Op, PathSeg, Row, Stage, TensorType};
use tessera_interp::{Tensor, run};
use tessera_kernels as _;
use tessera_linear as _;
use tessera_lower::form::{BF16_FORM, named};
use tessera_lower::{Device, Pins, Weight, lower};
use tessera_mem::trellis;
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::graphs::linear;
use tessera_testkit::{Noise, cosine_f32};

fn push_weight(g: &mut Graph, name: &str, ty: TensorType) -> NodeId {
    let idx = g.weights.len() as u32;
    g.weights.push(tessera_graph::Weight {
        path: vec![PathSeg::Field(name.into())],
        ty: ty.clone(),
    });
    push_node(g, Op::Weight(idx), Vec::new(), ty)
}

fn push_mm(g: &mut Graph, x: NodeId, w: NodeId, out_ty: TensorType) -> NodeId {
    push_node(g, Op::Matmul, vec![x, w], out_ty)
}

fn push_row(g: &mut Graph, stage: Stage, args: Vec<NodeId>, ty: TensorType) -> NodeId {
    push_node(g, Op::Row(Row(vec![stage])), args, ty)
}

fn make_tr_weight(k: usize, n: usize, side: u32, rng: &mut Noise) -> (Vec<u8>, Tensor) {
    let tr_form = named("trellis_k4_rot").unwrap();
    let entries: Vec<u8> = (0..n * k / 2).map(|_| (rng.word() & 0xFF) as u8).collect();
    let mut bytes = trellis::packed(k, n, side, &entries);
    let scales = (0..k + n).flat_map(|i| F16::from(0.75 + (i % 7) as f32 / 8.0).0.to_le_bytes());
    bytes.extend(scales);
    let w = Tensor::f32(vec![n, k], tr_form.unpack_all(&bytes));
    (bytes, w)
}

fn fuse_test_graph(kind: u8, k: u64, h: u64) -> Graph {
    let rows = [Dim::bounded(Rows, 2048)];
    let mut g = linear(&rows, k, h);
    let x = g.nodes[g.entries[0].outputs[0].node.0 as usize].args[0];
    let x_ty = g.nodes[x.0 as usize].ty.clone();
    let k_dim = x_ty.shape.0[1].clone();
    let h_ty = g.nodes[g.entries[0].outputs[0].node.0 as usize].ty.clone();
    let h_dim = h_ty.shape.0[1].clone();
    let w_kh = TensorType::new(vec![h_dim.clone(), k_dim.clone()], DType::F32);
    let w_hk = TensorType::new(vec![k_dim.clone(), h_dim.clone()], DType::F32);
    let vec_k = TensorType::new(vec![k_dim.clone()], DType::F32);
    let vec_h = TensorType::new(vec![h_dim.clone()], DType::F32);
    g.weights.clear();
    g.nodes.clear();
    let x = push_node(
        &mut g,
        Op::Input { entry: 0, port: 0 },
        Vec::new(),
        x_ty.clone(),
    );
    let nw = push_weight(&mut g, "nw", vec_k.clone());
    match kind {
        // SwiGLU: x + down(silu(gate(rms(x, nw))) * up(rms(x, nw)))
        0 => {
            let norm = push_row(
                &mut g,
                Stage::RmsNorm { eps: 1e-6 },
                vec![x, nw],
                x_ty.clone(),
            );
            let w_up = push_weight(&mut g, "up", w_kh.clone());
            let w_gt = push_weight(&mut g, "gate", w_kh);
            let w_dn = push_weight(&mut g, "down", w_hk);
            let up = push_mm(&mut g, norm, w_up, h_ty.clone());
            let gt = push_mm(&mut g, norm, w_gt, h_ty.clone());
            let act = push_row(&mut g, Stage::Act(Act::Silu), vec![gt], h_ty.clone());
            let inner = push_row(&mut g, Stage::Mul, vec![act, up], h_ty);
            let dn = push_mm(&mut g, inner, w_dn, x_ty.clone());
            let out = push_row(&mut g, Stage::Add, vec![x, dn], x_ty);
            g.entries[0].outputs[0].node = out;
        }
        // ViT MLP: x + fc2(gelu_tanh(fc1(ln(x, nw, nb)) + b1)) + b2
        1 => {
            let nb = push_weight(&mut g, "nb", vec_k.clone());
            let c = push_row(&mut g, Stage::Center, vec![x], x_ty.clone());
            let n = push_row(
                &mut g,
                Stage::RmsNorm { eps: 1e-6 },
                vec![c, nw],
                x_ty.clone(),
            );
            let ln = push_row(&mut g, Stage::Add, vec![n, nb], x_ty.clone());
            let w1 = push_weight(&mut g, "fc1", w_kh);
            let b1 = push_weight(&mut g, "b1", vec_h);
            let w2 = push_weight(&mut g, "fc2", w_hk);
            let b2 = push_weight(&mut g, "b2", vec_k);
            let y1 = push_mm(&mut g, ln, w1, h_ty.clone());
            let y1b = push_row(&mut g, Stage::Add, vec![y1, b1], h_ty.clone());
            let act = push_row(&mut g, Stage::Act(Act::GeluTanh), vec![y1b], h_ty);
            let y2 = push_mm(&mut g, act, w2, x_ty.clone());
            let y2b = push_row(&mut g, Stage::Add, vec![y2, b2], x_ty.clone());
            let out = push_row(&mut g, Stage::Add, vec![x, y2b], x_ty);
            g.entries[0].outputs[0].node = out;
        }
        // QKV: rms(x, nw) -> q, k, v summed for parity check.
        2 => {
            let norm = push_row(&mut g, Stage::RmsNorm { eps: 1e-6 }, vec![x, nw], x_ty);
            let wq = push_weight(&mut g, "q", w_kh.clone());
            let wk = push_weight(&mut g, "k", w_kh.clone());
            let wv = push_weight(&mut g, "v", w_kh);
            let q = push_mm(&mut g, norm, wq, h_ty.clone());
            let k_out = push_mm(&mut g, norm, wk, h_ty.clone());
            let v = push_mm(&mut g, norm, wv, h_ty.clone());
            let qk = push_row(&mut g, Stage::Add, vec![q, k_out], h_ty.clone());
            let out = push_row(&mut g, Stage::Add, vec![qk, v], h_ty);
            g.entries[0].outputs[0].node = out;
        }
        // ViT QKV: ln(x, nw, nb) -> qkv + b1
        _ => {
            let nb = push_weight(&mut g, "nb", vec_k);
            let c = push_row(&mut g, Stage::Center, vec![x], x_ty.clone());
            let n = push_row(
                &mut g,
                Stage::RmsNorm { eps: 1e-6 },
                vec![c, nw],
                x_ty.clone(),
            );
            let ln = push_row(&mut g, Stage::Add, vec![n, nb], x_ty);
            let w1 = push_weight(&mut g, "qkv", w_kh);
            let b1 = push_weight(&mut g, "b1", vec_h);
            let y = push_mm(&mut g, ln, w1, h_ty.clone());
            let out = push_row(&mut g, Stage::Add, vec![y, b1], h_ty);
            g.entries[0].outputs[0].node = out;
        }
    }
    g
}

pub(super) fn check_prep_fuse_cases(
    backend: Backend,
    cases: &[(u8, usize, usize, u64, &str, bool)],
    min_cos: f32,
) {
    if picked() != Some(backend) {
        return;
    }
    let Some(settings) = runs().into_iter().find(|s| s.gpu() == Some(backend)) else {
        return;
    };
    let rows = [Dim::bounded(Rows, 2048)];
    let probe = lower(
        &linear(&rows, 64, 64),
        0,
        &[(Rows, 1)],
        &Device::guess(&settings),
    )
    .unwrap();
    let platform = open(probe, &settings).unwrap().platform();
    let tr_form = named("trellis_k4_rot").unwrap();
    let mut rng = Noise(0xF05E_2B8B);

    for &(kind, k, h, m, expect_prep, absorbs_norm) in cases {
        let g = fuse_test_graph(kind, k as u64, h as u64);
        let pins = Pins::all(tr_form)
            .pin("nw", BF16_FORM)
            .pin("nb", BF16_FORM)
            .pin("b1", BF16_FORM)
            .pin("b2", BF16_FORM);
        let dev = Device::default().on(platform.clone()).pinned(pins);
        let plan = lower(&g, 0, &[(Rows, m)], &dev).unwrap();
        let names: Vec<_> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        let prep_count = names.iter().filter(|&&n| n == expect_prep).count();
        assert_eq!(
            prep_count, 1,
            "kind={kind} m={m}: expected 1 {expect_prep} in {names:?}"
        );
        if absorbs_norm {
            assert!(
                names
                    .iter()
                    .all(|&n| !n.contains("block_norm") && !n.contains("block_layer")),
                "kind={kind} m={m}: unfused norm in {names:?}"
            );
            assert!(
                names
                    .iter()
                    .all(|&n| { n != "trellis_prep_f16" && n != "trellis_prep" }),
                "kind={kind} m={m}: unfused prep in {names:?}"
            );
        }
        let mut ws = Vec::new();
        let mut packed: Vec<(usize, Weight)> = Vec::new();
        for (i, w) in g.weights.iter().enumerate() {
            let name = match &w.path[0] {
                PathSeg::Field(s) => s.as_str(),
                _ => "",
            };
            if matches!(name, "nw" | "nb" | "b1" | "b2") {
                let len = if name == "b1" { h } else { k };
                let raw: Vec<f32> = (0..len).map(|_| rng.draw() * 0.4 + 0.5).collect();
                let rt = BF16_FORM.roundtrip(&raw);
                ws.push(Tensor::f32(vec![len], rt.clone()));
                packed.push((i, Weight::Packed(BF16_FORM, BF16_FORM.pack(&rt))));
            } else {
                let (in_d, out_d) = if matches!(name, "down" | "fc2") {
                    (h, k)
                } else {
                    (k, h)
                };
                let (bytes, tw) = make_tr_weight(in_d, out_d, (i as u32 & 1) * 2, &mut rng);
                ws.push(tw);
                packed.push((i, Weight::Packed(tr_form, bytes)));
            }
        }
        let x = Tensor::f32(
            vec![m as usize, k],
            (0..m as usize * k).map(|_| rng.draw() * 0.5).collect(),
        );
        let want = run(&g, &ws, 0, std::slice::from_ref(&x)).unwrap();
        let mut runner = open(plan, &settings).unwrap();
        for (i, pw) in &packed {
            runner.load_weight(*i as u32, pw).unwrap();
        }
        let got = runner.run(&[x.view()]).unwrap().remove(0);
        let cos = cosine_f32(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(cos >= min_cos, "kind={kind} m={m}: cos={cos} < {min_cos}");
    }
}

#[test]
fn cuda_prep_fuse_parity() {
    let cases: &[(u8, usize, usize, u64, &str, bool)] = &[
        (0, 512, 1024, 47, "trellis_prep_pair", false),
        (0, 512, 1024, 65, "trellis_prep_pair", false),
        (0, 2048, 1024, 768, "trellis_prep_rms2", true),
        (1, 512, 1024, 65, "trellis_prep_f16", false),
        (1, 2048, 1024, 768, "trellis_prep_ln", true),
        (2, 512, 512, 65, "trellis_prep_multi", false),
        (2, 512, 512, 192, "trellis_prep_multi", false),
        (3, 2048, 1024, 768, "trellis_prep_ln", true),
    ];
    check_prep_fuse_cases(Backend::Cuda, cases, 0.99999);
}
