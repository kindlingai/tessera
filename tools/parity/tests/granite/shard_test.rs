//! Attention splits by tensor, the experts by matrix.

use super::{SPEC, STEPPER, small};
use tessera_graph::{Parts, Placement, Stage, shard};
use tessera_interp::{Tensor, run, run_parts};
use tessera_lower::{Device, lower};
use tessera_models::decoder::{Config, Moe, Scales, program, stepper};
use tessera_parity::{Spec, Stepper, pair, step_ranks};
use tessera_run::Loopback;
use tessera_run::testing::host;
use tessera_testkit::{Noise, fill, max_abs_diff};
use tessera_trace::DimName::{Context, Seqs};
use tessera_trace::{DimName, TensorType};

#[test]
fn routed_experts() {
    let g = shard(&small(), &Parts::even(2)).unwrap().graph;
    let placed = |p: Placement| g.weights.iter().filter(|w| w.ty.placement == p).count();
    // Per layer: q, k, v and both sets, then o.
    assert_eq!(placed(Placement::Split { axis: 0 }), 10);
    assert_eq!(placed(Placement::Split { axis: 1 }), 2);
    let sums = |n: &tessera_graph::Node| matches!(&n.op, tessera_graph::Op::Row(r) if r.0 == [Stage::AllReduce]);
    assert_eq!(g.nodes.iter().filter(|n| sums(n)).count(), 4);
    let routed = g
        .nodes
        .iter()
        .filter(|n| n.ty.placement == Placement::Routed);
    assert!(routed.count() > 0);
}

const SIZES: [(DimName, u64); 2] = [(Seqs, 1), (Context, 8)];

/// Heads, KV heads and experts all divide by four.
fn quarters() -> Config {
    Config {
        stack: tessera_models::Stack::sized([32, 64, 2, 8, 8, 32], 2, 8),
        kv_heads: 4,
        rope_theta: 1e4,
        eps: 1e-6,
        experts: 4,
        top_k: 2,
        scales: Scales {
            embed: 2.0,
            attention: Some(0.1),
            residual: 0.5,
            logits: 0.25,
        },
        tied: true,
    }
}

/// Three quarters on one rank, one on the other.
#[test]
fn uneven_ranks() {
    let g = program::<(), Moe>(&quarters());
    let parts = Parts(vec![3, 1]);
    let sharded = shard(&g, &parts).unwrap().graph;
    let mut noise = Noise(9);
    let mut draw = |ty: &TensorType| fill(&g, ty, &SIZES, &mut noise).unwrap();
    let weights: Vec<Tensor> = g.weights.iter().map(|w| draw(&w.ty)).collect();
    let inputs: Vec<Tensor> = g.entries[0].inputs.iter().map(|p| draw(&p.ty)).collect();
    let want = run(&g, &weights, 0, &inputs).unwrap();
    let want = want[0].floats().unwrap();
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let got = run_parts(&sharded, &weights, 0, &inputs, &parts).unwrap();
    let off = max_abs_diff(got[0].floats().unwrap(), want) / scale;
    assert!(off < 1e-4, "interpreter off {off}");
    let host = host();
    let device = Device::guess(&host)
        .zoned(pair())
        .ranked(vec![0, 1])
        .uneven(parts.0);
    let plans = [0, 1].map(|r| lower(&sharded, 0, &SIZES, &device.clone().as_rank(r)).unwrap());
    // Rank 0 holds three experts, rank 1 holds one.
    let experts = |plan: &tessera_lower::Plan| {
        let set = |l: &&tessera_lower::Loaded| sharded.weights[l.weight as usize].ty.rank() == 3;
        let l = plan.weights.iter().filter(set).find(|l| l.share.is_some());
        l.and_then(|l| l.share).map(|s| (s.first, s.count))
    };
    assert_eq!(
        (experts(&plans[0]), experts(&plans[1])),
        (Some((0, 3)), Some((3, 1)))
    );
    let mut ranks = Loopback::open(&plans, &host).unwrap();
    for (at, w) in weights.iter().enumerate() {
        ranks.load(at as u32, w.view()).unwrap();
    }
    let feeds: Vec<tessera_lower::View> = inputs.iter().map(Tensor::view).collect();
    let got = ranks.run(&feeds).unwrap();
    let off = max_abs_diff(got[0].floats().unwrap(), want) / scale;
    assert!(off < 1e-3, "plans off {off}");
}

/// Six KV heads on one rank, two on the other.
#[test]
fn uneven_step_heads() {
    let sharded = step_ranks(&stepper::<(), Moe>(&quarters()), &Parts(vec![3, 1]));
    assert_eq!(sharded.whole, []);
    let g = &sharded.graph;
    let placed = |p: Placement| g.weights.iter().filter(|w| w.ty.placement == p).count();
    // Per layer: q, k, v and both sets, then o.
    assert_eq!(placed(Placement::Split { axis: 0 }), 10);
    assert_eq!(placed(Placement::Split { axis: 1 }), 2);
    let heads = Placement::Split { axis: 2 };
    assert!(g.kept.iter().all(|k| k.placement == heads));
}

fn one_expert() -> Config {
    Config {
        top_k: 1,
        ..tessera_models::decoder::Config::granite_small()
    }
}

/// One expert per row keeps the top-k axis.
#[test]
fn steps_with_one_expert() {
    let spec = Spec {
        small: Some(|| program::<(), Moe>(&one_expert())),
        stepper: Some(Stepper {
            small: || stepper::<(), Moe>(&one_expert()),
            ..STEPPER
        }),
        ..SPEC
    };
    tessera_parity::steps(&spec);
}
