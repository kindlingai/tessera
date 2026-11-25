//! The toy decoder over two ranks matches one.

use tessera_graph::{Graph, Parts, Placement, shard};
use tessera_interp::{Tensor, run, run_ranks};
use tessera_lower::{Device, Plan, View, lower};
use tessera_models::decoder::{Config, Swiglu, program, stepper};
use tessera_parity::{pair, step_ranks};
use tessera_run::Loopback;
use tessera_run::testing::{host, runs};
use tessera_testkit::{Noise, fill, max_abs_diff};
use tessera_trace::DimName::{Context, Seqs};
use tessera_trace::{DimName, Norm, TensorType};

const SIZES: [(DimName, u64); 2] = [(Seqs, 1), (Context, 8)];

/// The toy graph, sharded too, with faux weights and inputs.
fn toy() -> (Graph, Graph, Vec<Tensor>, Vec<Tensor>) {
    let g = program::<Norm, Swiglu>(&Config::toy(2, 8));
    let sharded = shard(&g, &Parts::even(2))
        .unwrap()
        .graph
        .wired(tessera_graph::DType::F32);
    let mut noise = Noise(5);
    let mut draw = |ty: &TensorType| fill(&g, ty, &SIZES, &mut noise).unwrap();
    let weights = g.weights.iter().map(|w| draw(&w.ty)).collect();
    let inputs = g.entries[0].inputs.iter().map(|p| draw(&p.ty)).collect();
    (g, sharded, weights, inputs)
}

/// How far `got` is from the whole graph's logits, relative.
fn off(got: &Tensor, g: &Graph, weights: &[Tensor], inputs: &[Tensor]) -> f32 {
    let want = run(g, weights, 0, inputs).unwrap();
    let want = want[0].floats().unwrap();
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    max_abs_diff(got.floats().unwrap(), want) / scale
}

#[test]
fn two_ranks() {
    let (g, sharded, weights, inputs) = toy();
    let split = |axis| {
        let is = |w: &&tessera_graph::Weight| w.ty.placement == Placement::Split { axis };
        sharded.weights.iter().filter(is).count()
    };
    // Per layer: five column splits, two row splits.
    assert_eq!((split(0), split(1)), (10, 4));
    let two = run_ranks(&sharded, &weights, 0, &inputs, 2).unwrap();
    let diff = off(&two[0], &g, &weights, &inputs);
    assert!(diff < 1e-4, "{diff}");
}

#[test]
fn two_rank_plan() {
    let (g, sharded, weights, inputs) = toy();
    let host = host();
    let device = Device::guess(&host).zoned(pair()).ranked(vec![0, 1]);
    let plans = [0, 1].map(|r| lower(&sharded, 0, &SIZES, &device.clone().as_rank(r)).unwrap());
    let plan = &plans[0];
    let sums = plan
        .steps
        .iter()
        .filter(|s| s.kernel.name() == "all_reduce");
    assert_eq!((plan.ranks, sums.count()), (2, 4));
    // Each rank holds half of every split weight.
    let whole = lower(&g, 0, &SIZES, &Device::guess(&host)).unwrap();
    let held = |p: &Plan| p.weights.iter().map(|l| l.slot.len).sum::<usize>();
    let split: usize = (sharded.weights.iter().zip(&weights))
        .filter(|(w, _)| w.ty.placement != Placement::Replicated)
        .map(|(_, t)| t.len())
        .sum();
    assert_eq!(held(&whole) - held(plan), split / 2);
    assert!(device.cost(plan).seconds.is_finite());
    // On the host, then on any ready GPU.
    for settings in runs() {
        let mut ranks = Loopback::open(&plans, &settings).unwrap();
        for (at, w) in weights.iter().enumerate() {
            ranks.load(at as u32, w.view()).unwrap();
        }
        let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
        let got = ranks.run(&feeds).unwrap();
        let diff = off(&got[0], &g, &weights, &inputs);
        assert!(diff < 1e-3, "{diff} on {}", ranks.rank(0).device());
    }
}

/// Weights placed so, with a path step `name` if given.
fn placed(g: &Graph, placement: Placement, name: Option<&str>) -> usize {
    let named = |w: &&tessera_graph::Weight| {
        name.is_none_or(|n| w.path.contains(&tessera_graph::PathSeg::Field(n.into())))
    };
    let on = g.weights.iter().filter(|w| w.ty.placement == placement);
    on.filter(named).count()
}

#[test]
fn step_splits_heads() {
    let g = stepper::<Norm, Swiglu>(&Config::toy(2, 8));
    let sharded = step_ranks(&g, &Parts::even(2));
    assert_eq!(sharded.whole, []);
    let g = &sharded.graph;
    let (columns, rows) = (Placement::Split { axis: 0 }, Placement::Split { axis: 1 });
    for name in ["q", "k", "v"] {
        assert_eq!(placed(g, columns, Some(name)), 2, "{name}");
    }
    assert_eq!(placed(g, rows, Some("o")), 2);
    let heads = Placement::Split { axis: 2 };
    assert!(g.kept.iter().all(|k| k.placement == heads));
}

#[test]
fn uneven_heads_stay_whole() {
    // Two KV heads cannot take three quarters and one.
    let g = stepper::<Norm, Swiglu>(&Config::toy(2, 8));
    let sharded = step_ranks(&g, &Parts(vec![3, 1]));
    // Per layer q, k, v and o stay whole.
    assert_eq!(sharded.whole.len(), 8);
    assert_eq!(
        placed(&sharded.graph, Placement::Split { axis: 1 }, None),
        2
    );
    let whole = |k: &TensorType| k.placement == Placement::Replicated;
    assert!(sharded.graph.kept.iter().all(whole));
}
