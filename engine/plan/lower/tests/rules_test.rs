//! Each rule's launches against the CPU interpreter, on the GPU.

// Linking the crates registers their rules.
use kernels as _;
use linear as _;

use gpukit::testing::{Noise, max_abs_diff};
use graph::DimName::{Context, HeadDim, Heads, Hidden, Rows, Seqs, Vocab};
use graph::{DType, Dim, DimName, Graph, NodeId, Size, TensorType, verify};
use interp::{Tensor, run};
use lower::form::BF16_FORM;
use lower::{Device, Pins, View, cut, lower, open, rules};
use models::{Stack, nanogpt, qwen3};
use settings::read as settings;
use trace::{EntryKind, Mask, Model, Port, Role};

const SIZES: [(DimName, u64); 3] = [(Seqs, 2), (Rows, 3), (Context, 5)];

/// Fewer queries than keys, then a softmax and a mean.
fn decode() -> Graph {
    let dim = |name, size| Dim { name, size };
    let bounded = [(Seqs, 4), (Rows, 8), (Context, 16)];
    let mut dims: Vec<Dim> = bounded.map(|(n, max)| dim(n, Size::Bounded(max))).into();
    dims.extend([dim(Heads, Size::Fixed(2)), dim(HeadDim, Size::Fixed(16))]);
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
    m.finish([(Role::Logits, &out), (Role::Logits, &out.mean(1))])
}

/// Each row's likeliest entry, over more entries than threads.
fn likeliest() -> Graph {
    let dim = |name, size| Dim { name, size };
    let bounded = |name| dim(name, Size::Bounded(8));
    let m = Model::new(vec![
        bounded(Seqs),
        bounded(Rows),
        dim(Vocab, Size::Fixed(600)),
    ]);
    let ty = TensorType::new([Seqs, Rows, Vocab], DType::F32);
    let role = Role::Logits;
    let [logits] = m.entry(EntryKind::Forward, [Port { role, ty }]);
    m.finish([(Role::Tokens, &logits.top_k(1u64))])
}

/// Rows of 256, rotated in two blocks.
fn rotated() -> Graph {
    let dim = |name, size| Dim { name, size };
    let bounded = |name| dim(name, Size::Bounded(8));
    let m = Model::new(vec![
        bounded(Seqs),
        bounded(Rows),
        dim(Hidden, Size::Fixed(256)),
    ]);
    let ty = TensorType::new([Seqs, Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let [x] = m.entry(EntryKind::Forward, [Port { role, ty }]);
    m.finish([(Role::Logits, &x.hadamard(128))])
}

/// A toy of each model, and ops neither one uses.
fn graphs() -> Vec<Graph> {
    let stack = Stack {
        vocab: 32,
        hidden: 32,
        layers: 2,
        heads: 4,
        head_dim: 8,
        ffn: 64,
        max_seqs: 4,
        max_context: 16,
    };
    let tiny = nanogpt::Config { stack, eps: 1e-5 };
    let qwen = qwen3::program(&qwen3::Config::toy(4, 16));
    vec![
        qwen,
        nanogpt::program(&tiny),
        decode(),
        likeliest(),
        rotated(),
    ]
}

/// Random floats, or ids below 8, shaped like `ty`.
fn fill(g: &Graph, ty: &TensorType, noise: &mut Noise) -> Tensor {
    let size = |name| SIZES.iter().find(|(n, _)| *n == name).map(|(_, s)| *s);
    let dim = |e: &graph::DimExpr| e.poly(&g.dims).unwrap().eval(&size).unwrap() as usize;
    let shape: Vec<usize> = ty.shape.0.iter().map(dim).collect();
    let mut values = (0..shape.iter().product()).map(|_| noise.draw());
    match ty.dtype {
        DType::U32 => Tensor::u32(shape, values.map(|v| ((v + 1.0) * 4.0) as u32).collect()),
        _ => Tensor::f32(shape, values.by_ref().collect()),
    }
}

/// Runs entry 0 both ways. Returns the largest difference.
fn gap(g: &Graph, weights: &[Tensor], inputs: &[Tensor], device: &Device) -> f32 {
    let want = run(g, weights, 0, inputs).unwrap();
    let mut runner = open(lower(g, 0, &SIZES, device).unwrap(), &settings()).unwrap();
    for (at, w) in weights.iter().enumerate() {
        runner.load(at as u32, w.view()).unwrap();
    }
    let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
    let got = runner.run(&feeds).unwrap().remove(0);
    if want[0].dtype() != DType::U32 {
        return max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
    }
    // Indices must match exactly.
    let same = got.indices().unwrap() == want[0].indices().unwrap();
    if same { 0.0 } else { f32::INFINITY }
}

/// One device per registered rule, favoring it everywhere.
fn devices() -> Vec<(&'static str, Device)> {
    let favor = |r: &&lower::Rule| (r.name, Device::guess(&settings()).favoring(r.name));
    rules().iter().map(favor).collect()
}

#[test]
fn each_rule_matches_cpu() {
    let mut noise = Noise(7);
    let mut used: Vec<&str> = Vec::new();
    for (g, (rule, device)) in graphs()
        .iter()
        .flat_map(|g| devices().into_iter().map(move |d| (g, d)))
    {
        let form = lower::form::FORMS
            .iter()
            .copied()
            .find(|f| f.block() > 1 && rule.ends_with(f.name()));
        let device = match form {
            Some(form) => device.pinned(Pins::all(form)),
            None => device,
        };
        let Ok(plan) = lower(g, 0, &SIZES, &device) else {
            continue;
        };
        let mut seen: Vec<&[NodeId]> = Vec::new();
        for step in &plan.steps {
            used.extend((step.rule == rule).then_some(rule));
            if step.covers.is_empty() || seen.contains(&&step.covers[..]) {
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
            let weights: Vec<Tensor> = sub
                .weights
                .iter()
                .map(|w| {
                    let value = fill(&sub, &w.ty, &mut noise);
                    match value.floats() {
                        Ok(data) => Tensor::f32(
                            value.shape.clone(),
                            device.pins.form(&w.path).roundtrip(data),
                        ),
                        Err(_) => value,
                    }
                })
                .collect();
            let off = gap(&sub, &weights, &inputs, &device);
            assert!(off < 1e-4, "{:?} is off by {off:e}", step.kernel);
        }
    }
    // A new rule must be reachable from some graph here.
    for rule in rules() {
        // Hand-written kernels wait for a compiler.
        if settings().gpu().is_none() && rule.name == "dense" {
            continue;
        }
        assert!(
            used.contains(&rule.name),
            "no graph exercises {}",
            rule.name
        );
    }
}

#[test]
fn whole_models_match_cpu() {
    let mut noise = Noise(11);
    for g in graphs() {
        let weights: Vec<Tensor> = g
            .weights
            .iter()
            .map(|w| fill(&g, &w.ty, &mut noise))
            .collect();
        let ports = &g.entries[0].inputs;
        let inputs: Vec<Tensor> = ports.iter().map(|p| fill(&g, &p.ty, &mut noise)).collect();
        for (rule, device) in devices() {
            let off = gap(&g, &weights, &inputs, &device);
            assert!(off < 1e-3, "favoring {rule}, outputs are off by {off:e}");
        }
        // Weights the GPU reads as bf16, the CPU reads widened.
        let narrow = |t: &Tensor| match t.floats() {
            Ok(v) => Tensor::f32(t.shape.clone(), BF16_FORM.roundtrip(v)),
            Err(_) => t.clone(),
        };
        let weights: Vec<Tensor> = weights.iter().map(narrow).collect();
        for (rule, device) in devices() {
            let device = device.pinned(Pins::all(BF16_FORM));
            let off = gap(&g, &weights, &inputs, &device);
            assert!(
                off < 1e-3,
                "favoring {rule} as bf16, outputs are off by {off:e}"
            );
        }
    }
}
