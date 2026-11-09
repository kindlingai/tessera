//! Each rule's launches against the CPU interpreter, on the GPU.

use graph::DimName::{Context, HeadDim, Heads, Rows, Seqs};
use graph::{DType, Dim, DimName, Graph, NodeId, Size, TensorType, verify};
use interp::{Cpu, Data, Tensor, run};
use lower::{Device, Feed, cut, lower, open};
use models::{Stack, nanogpt, qwen3};
use ops::testing::{Noise, gpu_available, max_abs_diff};
use trace::{EntryKind, Mask, Model, Port, Role};

const SIZES: [(DimName, u64); 3] = [(Seqs, 2), (Rows, 3), (Context, 5)];

/// Fewer queries than keys, then a softmax.
fn decode() -> Graph {
    let dim = |name, size| Dim { name, size };
    let bounded = [(Seqs, 4), (Rows, 8), (Context, 16)];
    let mut dims: Vec<Dim> = bounded.map(|(n, max)| dim(n, Size::Bounded(max))).into();
    dims.extend([dim(Heads, Size::Fixed(2)), dim(HeadDim, Size::Fixed(4))]);
    let m = Model::new(dims);
    let port = |rows| Port {
        role: Role::Tokens,
        ty: TensorType::new([Seqs, rows, Heads, HeadDim], DType::F32),
    };
    let [q, k, v] = m.entry(
        EntryKind::Forward,
        [port(Rows), port(Context), port(Context)],
    );
    let out = m.attend(Mask::Causal, 0.5, [&q, &k, &v], None).softmax();
    m.finish([(Role::Logits, &out)])
}

/// A toy of each model, and ops neither one uses.
fn graphs() -> Vec<Graph> {
    let stack = Stack {
        vocab: 32,
        hidden: 16,
        layers: 2,
        heads: 4,
        head_dim: 4,
        ffn: 64,
        max_seqs: 4,
        max_context: 16,
    };
    let tiny = nanogpt::Config { stack, eps: 1e-5 };
    let qwen = qwen3::program(&qwen3::Config::toy(4, 16));
    vec![qwen, nanogpt::program(&tiny), decode()]
}

/// Random floats, or ids below 8, shaped like `ty`.
fn fill(g: &Graph, ty: &TensorType, noise: &mut Noise) -> Tensor {
    let size = |name| SIZES.iter().find(|(n, _)| *n == name).map(|(_, s)| *s);
    let dim = |e: &graph::DimExpr| e.poly(&g.dims).unwrap().eval(&size).unwrap() as usize;
    let shape: Vec<usize> = ty.shape.0.iter().map(dim).collect();
    let mut values = (0..shape.iter().product()).map(|_| noise.draw());
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
fn gap(g: &Graph, weights: &[Tensor], inputs: &[Tensor], device: &Device) -> f32 {
    let want = run(g, &mut Cpu, weights, 0, inputs).unwrap();
    let runner = open(lower(g, 0, &SIZES, device).unwrap()).unwrap();
    for (at, w) in weights.iter().enumerate() {
        runner.load(at as u32, w.floats().unwrap()).unwrap();
    }
    let feeds: Vec<Feed> = inputs.iter().map(feed).collect();
    let got = runner.run(&feeds).unwrap();
    max_abs_diff(&got[0], want[0].floats().unwrap())
}

/// One device per kernel choice, so each gets checked.
fn devices() -> [Device; 3] {
    let tiled = || Device::guess().without("thin");
    let thin = Device::guess().without("matmul");
    [tiled(), thin, tiled().without("attend")]
}

#[test]
fn each_rule_matches_cpu() {
    if !gpu_available() {
        return eprintln!("no GPU here; skipped");
    }
    let mut noise = Noise(7);
    let mut checked = 0;
    for (g, device) in graphs().iter().flat_map(|g| devices().map(|d| (g, d))) {
        let plan = lower(g, 0, &SIZES, &device).unwrap();
        let mut seen: Vec<&[NodeId]> = Vec::new();
        for step in &plan.steps {
            if seen.contains(&&step.covers[..]) {
                continue;
            }
            seen.push(&step.covers);
            // The nodes one rule covered, as their own graph.
            let sub = cut(g, &step.covers);
            verify(&sub).unwrap();
            let ports = &sub.entries[0].inputs;
            let inputs: Vec<Tensor> = ports
                .iter()
                .map(|p| fill(&sub, &p.ty, &mut noise))
                .collect();
            let off = gap(&sub, &[], &inputs, &device);
            assert!(off < 1e-4, "{:?} is off by {off:e}", step.kernel);
        }
        checked += seen.len();
    }
    assert!(checked > 120, "only {checked} rule uses checked");
}

#[test]
fn whole_models_match_cpu() {
    if !gpu_available() {
        return eprintln!("no GPU here; skipped");
    }
    let mut noise = Noise(11);
    for g in graphs() {
        let weights: Vec<Tensor> = g
            .weights
            .iter()
            .map(|w| fill(&g, &w.ty, &mut noise))
            .collect();
        let ports = &g.entries[0].inputs;
        let inputs: Vec<Tensor> = ports.iter().map(|p| fill(&g, &p.ty, &mut noise)).collect();
        for device in devices() {
            let off = gap(&g, &weights, &inputs, &device);
            assert!(off < 1e-3, "outputs are off by {off:e}");
        }
    }
}
