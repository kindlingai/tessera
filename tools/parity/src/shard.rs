//! The small graph over two ranks matches one rank.

use super::{Spec, host, runs};
use sensai_graph::{Graph, Parts, Placement, Sharded, shard};
use sensai_interp::{Tensor, run, run_parts, run_ranks};
use sensai_lower::form::BF16_FORM;
use sensai_lower::{Cluster, Device, Link, View, Weight, lower};
use sensai_run::Loopback;
use sensai_testkit::{Noise, fill, max_abs_diff};
use sensai_trace::DimName::{Context, Rows, Seqs};
use sensai_trace::{DimName, TensorType};

const SIZES: [(DimName, u64); 2] = [(Seqs, 1), (Context, 8)];
const RANKS: usize = 2;

/// Two GPU zones, linked both ways.
pub fn pair() -> Cluster {
    Cluster::two("a", "b")
}

/// `n` GPU zones, each linked to every other.
fn mesh(n: usize) -> Cluster {
    let mut cluster = pair();
    let link = cluster.links[0];
    for to in 2..n {
        let zone = cluster.zones[0].clone();
        cluster.add(zone);
        for from in 0..to {
            cluster.links.push(Link { from, to, ..link });
            let (from, to) = (to, from);
            cluster.links.push(Link { from, to, ..link });
        }
    }
    cluster
}

/// Within a thousandth of the largest value.
fn close(label: &str, got: &Tensor, want: &Tensor) {
    let (got, want) = (got.floats().unwrap(), want.floats().unwrap());
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let off = max_abs_diff(got, want) / scale;
    assert!(off < 1e-3, "{label}: off {off:e}");
}

/// Sharded, the interpreter and the plans over two ranks agree.
pub fn shards(spec: &Spec) {
    let Some(small) = spec.small else { return };
    let g = small();
    let sharded = shard(&g, &Parts::even(RANKS)).unwrap().graph;
    let mut noise = Noise(5);
    let mut draw = |ty: &TensorType| fill(&g, ty, &SIZES, &mut noise).unwrap();
    // As bf16, so each rank can load its part packed.
    let half = |t: Tensor| Tensor::f32(t.shape.clone(), BF16_FORM.roundtrip(t.floats().unwrap()));
    let weights: Vec<Tensor> = g.weights.iter().map(|w| half(draw(&w.ty))).collect();
    let inputs: Vec<Tensor> = g.entries[0].inputs.iter().map(|p| draw(&p.ty)).collect();
    // Each rank holds its own matrices of every set.
    for w in sharded.weights.iter().filter(|w| w.ty.rank() == 3) {
        assert_eq!(w.ty.placement, Placement::Split { axis: 0 }, "{:?}", w.path);
    }
    let want = run(&g, &weights, 0, &inputs).unwrap();
    let two = run_ranks(&sharded, &weights, 0, &inputs, RANKS).unwrap();
    close(&format!("{} interpreted", spec.name), &two[0], &want[0]);
    let host = host();
    let device = Device::guess(&host).zoned(pair()).ranked(vec![0, 1]);
    let plans = [0, 1].map(|r| lower(&sharded, 0, &SIZES, &device.clone().as_rank(r)).unwrap());
    let plan = &plans[0];
    let split = (sharded.weights.iter()).any(|w| w.ty.placement != Placement::Replicated);
    let sums = (plan.steps.iter()).filter(|s| s.kernel.name() == "all_reduce");
    assert_eq!(
        sums.count() > 0,
        split,
        "{}: sums without splits",
        spec.name
    );
    for settings in runs() {
        let mut ranks = Loopback::open(&plans, &settings).unwrap();
        for (at, w) in weights.iter().enumerate() {
            let packed = Weight::Packed(BF16_FORM, BF16_FORM.pack(w.floats().unwrap()));
            ranks.load_weight(at as u32, &packed).unwrap();
        }
        let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
        let got = ranks.run(&feeds).unwrap();
        let label = format!("{} on {}", spec.name, ranks.rank(0).device());
        close(&label, &got[0], &want[0]);
    }
}

/// One launch of `n` rows in slot zero.
fn launch(n: usize) -> [Tensor; 3] {
    let column = |of: fn(u32) -> u32| Tensor::u32(vec![n], (0..n as u32).map(of).collect());
    [column(|p| p * 7 % 32), column(|p| p), column(|_| 0)]
}

/// A step graph over `parts` matches one rank's logits.
pub fn step_ranks(g: &Graph, parts: &Parts) -> Sharded {
    let sharded = shard(g, parts).unwrap();
    let mut noise = Noise(7);
    let sizes = [(Seqs, 1), (Context, 8)];
    let mut draw = |ty: &TensorType| fill(g, ty, &sizes, &mut noise).unwrap();
    let weights: Vec<Tensor> = g.weights.iter().map(|w| draw(&w.ty)).collect();
    let n = 6;
    let inputs = launch(n);
    let want = run(g, &weights, 0, &inputs).unwrap();
    let got = run_parts(&sharded.graph, &weights, 0, &inputs, parts).unwrap();
    close("step interpreted", &got[1], &want[1]);
    let ranks: Vec<u32> = (0..parts.ranks() as u32).collect();
    let host = host();
    let device = Device::guess(&host).zoned(mesh(ranks.len()));
    let device = device.ranked(ranks.clone());
    let device = device.uneven(parts.0.clone());
    let sizes = [(Seqs, 1), (Rows, n as u64), (Context, 8)];
    let lowered = |r: &u32| lower(&sharded.graph, 0, &sizes, &device.clone().as_rank(*r));
    let plans: Vec<_> = ranks.iter().map(|r| lowered(r).unwrap()).collect();
    for settings in runs() {
        let mut ranks = Loopback::open(&plans, &settings).unwrap();
        for (at, w) in weights.iter().enumerate() {
            ranks.load(at as u32, w.view()).unwrap();
        }
        let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
        let got = ranks.run(&feeds).unwrap();
        close(
            &format!("step on {}", ranks.rank(0).device()),
            &got[1],
            &want[1],
        );
    }
    sharded
}
