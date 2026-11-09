//! Each rule's launches against the CPU interpreter, on Metal.
#![cfg(target_os = "macos")]

use graph::{DType, DimName, Graph, NodeId, TensorType, verify};
use interp::{Cpu, Data, Tensor, run};
use lower::metal::{Feed, Runner};
use lower::{cut, lower};
use models::qwen3::{Config, program};
use ops::testing::{gpu_available, max_abs_diff};

const SIZES: [(DimName, u64); 2] = [(DimName::Seqs, 2), (DimName::Context, 5)];

/// A fixed stream of values in `[-1, 1)`.
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f32 {
        let step = self.0.wrapping_mul(6_364_136_223_846_793_005);
        self.0 = step.wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// Random floats, or ids below 8, shaped like `ty`.
fn fill(g: &Graph, ty: &TensorType, noise: &mut Noise) -> Tensor {
    let size = |name| SIZES.iter().find(|(n, _)| *n == name).map(|(_, s)| *s);
    let dim = |e: &graph::DimExpr| e.poly(&g.dims).unwrap().eval(&size).unwrap() as usize;
    let shape: Vec<usize> = ty.shape.0.iter().map(dim).collect();
    let mut values = (0..shape.iter().product()).map(|_| noise.next());
    let data = match ty.dtype {
        DType::U32 => Data::U32(values.map(|v| ((v + 1.0) * 4.0) as u32).collect()),
        _ => Data::F32(values.by_ref().collect()),
    };
    Tensor { shape, data }
}

fn feed(t: &Tensor) -> Feed<'_> {
    match &t.data {
        Data::F32(v) => Feed::Floats(v),
        Data::U32(v) => Feed::Ids(v),
        Data::I32(_) => unreachable!("no signed inputs"),
    }
}

/// Runs entry 0 both ways; returns the largest difference.
fn gap(g: &Graph, weights: &[Tensor], inputs: &[Tensor]) -> f32 {
    let want = run(g, &mut Cpu, weights, 0, inputs).unwrap();
    let runner = Runner::new(lower(g, 0, &SIZES).unwrap()).unwrap();
    for (at, w) in weights.iter().enumerate() {
        runner.load(at as u32, w.floats().unwrap()).unwrap();
    }
    let feeds: Vec<Feed> = inputs.iter().map(feed).collect();
    let got = runner.run(&feeds).unwrap();
    max_abs_diff(&got[0], want[0].floats().unwrap())
}

#[test]
fn each_rule_matches_cpu() {
    if !gpu_available() {
        return eprintln!("no GPU here; skipped");
    }
    let g = program(&Config::toy(4, 16));
    let plan = lower(&g, 0, &SIZES).unwrap();
    let mut noise = Noise(7);
    let mut seen: Vec<&[NodeId]> = Vec::new();
    for step in &plan.steps {
        if seen.contains(&&step.covers[..]) {
            continue;
        }
        seen.push(&step.covers);
        // The nodes one rule covered, as their own graph.
        let sub = cut(&g, &step.covers);
        verify(&sub).unwrap();
        let ports = &sub.entries[0].inputs;
        let inputs: Vec<Tensor> = ports
            .iter()
            .map(|p| fill(&sub, &p.ty, &mut noise))
            .collect();
        let off = gap(&sub, &[], &inputs);
        assert!(off < 1e-4, "{:?} is off by {off:e}", step.kernel);
    }
    assert!(seen.len() > 20, "only {} rule uses checked", seen.len());
}

#[test]
fn whole_model_matches_cpu() {
    if !gpu_available() {
        return eprintln!("no GPU here; skipped");
    }
    let g = program(&Config::toy(4, 16));
    let mut noise = Noise(11);
    let weights: Vec<Tensor> = g
        .weights
        .iter()
        .map(|w| fill(&g, &w.ty, &mut noise))
        .collect();
    let ports = &g.entries[0].inputs;
    let inputs: Vec<Tensor> = ports.iter().map(|p| fill(&g, &p.ty, &mut noise)).collect();
    let off = gap(&g, &weights, &inputs);
    assert!(off < 1e-3, "logits are off by {off:e}");
}
