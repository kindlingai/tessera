//! The per-model suite: what every model test runs alike.

// Linking these registers their lowering rules.
use kernels as _;
use linear as _;

mod golden;

pub use golden::{Case, Check, Compare, Golden, Reference, Source, Tolerance, golden};
pub use golden::{pulled, unit};

use bundle::names::{repo, resolve};
use checkpoint::{Slice, Stored};
use interp::{Tensor, run};
use lower::form::named;
use lower::{Device, Pins, Plan, Settings, View, lower};
use run::{Run, open};
use settings::read as settings;
use std::collections::BTreeMap;
use testkit::{Noise, cosine_f32, fill, max_abs_diff};
use trace::DimName::{Context, Seqs};
use trace::{Composite, Graph, TensorType};

/// One model as the suite sees it.
pub struct Spec {
    pub name: &'static str,
    /// The graph at its released sizes.
    pub release: fn() -> Graph,
    pub parameters: u64,
    /// Regions of each kind the release graph marks.
    pub regions: &'static [(Composite, usize)],
    /// A graph small enough for the host executor.
    pub small: Option<fn() -> Graph>,
    /// Real files' slices, each with the graph it feeds.
    pub files: &'static [Manifest],
    /// Outputs a reference recorded.
    pub golden: Option<Golden>,
}

/// Tokens and positions, `[seqs, len]` each.
pub fn inputs(seqs: usize, tokens: &[u32]) -> [Tensor; 2] {
    let len = tokens.len() / seqs;
    let shape = vec![seqs, len];
    let positions = (0..seqs).flat_map(|_| 0..len as u32).collect();
    let ids = |data| Tensor::u32(shape.clone(), data);
    [ids(tokens.to_vec()), ids(positions)]
}

/// A runner for `plan` with every weight loaded.
pub fn loaded(plan: Plan, g: &Graph, weights: &[Tensor], settings: &Settings) -> Box<dyn Run> {
    let mut runner = open(plan, settings).unwrap();
    for (at, w) in weights.iter().enumerate() {
        let path = &g.weights[at].path;
        (runner.load(at as u32, w.view())).unwrap_or_else(|e| panic!("weight {at} {path:?}: {e}"));
    }
    runner
}

/// The interpreter's first output.
pub fn interpreted(g: &Graph, weights: &[Tensor], seqs: usize, tokens: &[u32]) -> Vec<f32> {
    let out = run(g, weights, 0, &inputs(seqs, tokens)).unwrap();
    out[0].floats().unwrap().to_vec()
}

/// How often a plan launches each kernel.
pub fn kernels(plan: &Plan) -> BTreeMap<&'static str, usize> {
    let mut by = BTreeMap::new();
    for step in &plan.steps {
        *by.entry(step.kernel.name()).or_default() += step.launches().len();
    }
    by
}

/// Seconds each step of the plan in place takes alone.
pub fn seconds(runner: &mut dyn Run) -> Vec<f64> {
    let steps = 0..runner.plan().steps.len();
    steps.map(|i| runner.repeat(i, 1).unwrap()).collect()
}

/// Launches and seconds, by kernel.
pub fn table(plan: &Plan, steps: &[f64]) {
    let mut by: BTreeMap<&str, (usize, f64)> = BTreeMap::new();
    for (step, seconds) in plan.steps.iter().zip(steps) {
        let (count, total) = by.entry(step.kernel.name()).or_default();
        (*count, *total) = (*count + step.launches().len(), *total + seconds);
    }
    for (name, (count, seconds)) in by {
        eprintln!("  {name:8} {count:4} launches {seconds:.4} s");
    }
}

/// A manifest's text and the graph its file feeds.
pub type Manifest = (&'static str, fn() -> Graph);

/// Two layers of two heads, for a faux run.
pub fn small_stack(hidden: u64, head_dim: u64, ffn: u64) -> models::Stack {
    models::Stack {
        vocab: 64,
        hidden,
        layers: 2,
        heads: 2,
        head_dim,
        ffn,
        max_seqs: 1,
        max_context: 16,
    }
}

/// The tests every model runs: `suite!(SPEC)`.
macro_rules! suite {
    ($spec:expr) => {
        #[test]
        fn traces() {
            crate::common::traces(&$spec);
        }

        #[test]
        fn files() {
            crate::common::files(&$spec);
        }

        #[test]
        fn faux_weights() {
            crate::common::faux(&$spec);
        }

        #[test]
        fn golden() {
            crate::common::golden(&$spec);
        }
    };
}
pub(crate) use suite;

/// The release graph survives JSON, verifies and has its sizes.
pub fn traces(spec: &Spec) {
    let g = (spec.release)();
    let back = Graph::from_json(&g.to_json()).unwrap();
    assert_eq!(back, g);
    graph::verify(&back).unwrap();
    let marked = |kind| g.regions.iter().filter(|r| r.kind == kind).count();
    for &(kind, n) in spec.regions {
        assert_eq!(marked(kind), n, "{} {kind:?} regions", spec.name);
    }
    let all: usize = spec.regions.iter().map(|(_, n)| n).sum();
    assert_eq!(g.regions.len(), all, "{} regions", spec.name);
    let count = |w: &graph::Weight| w.ty.shape.elements(&g.dims).unwrap().as_const().unwrap();
    let parameters: u64 = g.weights.iter().map(count).sum();
    assert_eq!(parameters, spec.parameters, "{} parameters", spec.name);
}

/// A real file's slices, as `tessera-eval --slices` wrote them.
pub fn slices(json: &str) -> Vec<Slice> {
    let entries: Vec<serde_json::Value> = serde_json::from_str(json).unwrap();
    let slice = |e: &serde_json::Value| Slice {
        name: e["name"].as_str().unwrap().to_string(),
        shape: (e["shape"].as_array().unwrap().iter())
            .map(|n| n.as_u64().unwrap() as usize)
            .collect(),
        stored: Stored::parse(e["stored"].as_str().unwrap()).unwrap(),
    };
    entries.iter().map(slice).collect()
}

/// Every weight claims a slice of each real file.
pub fn files(spec: &Spec) {
    let rules = repo(spec.name).unwrap_or_default();
    for (manifest, graph) in spec.files {
        let g = graph();
        match resolve(&g, &slices(manifest), &[&rules]) {
            Ok(claims) => assert_eq!(claims.len(), g.weights.len()),
            Err(e) => panic!("{}: {e}", spec.name),
        }
    }
}

/// Rows the faux run feeds.
const ROWS: u64 = 8;

/// Random weights in every form: device against interpreter.
pub fn faux(spec: &Spec) {
    let Some(small) = spec.small else { return };
    let g = small();
    let mut noise = Noise(17);
    let sizes = [(Seqs, 1), (Context, ROWS)];
    let mut draw = |ty: &TensorType| fill(&g, ty, &sizes, &mut noise).unwrap();
    let weights: Vec<Tensor> = g.weights.iter().map(|w| draw(&w.ty)).collect();
    let ports = &g.entries[0].inputs;
    let inputs: Vec<Tensor> = ports.iter().map(|p| draw(&p.ty)).collect();
    let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
    let settings = settings();
    for form in ["f32", "bf16", "q8_0", "q4_0", "nvfp4"] {
        let f = named(form).unwrap();
        // The interpreter sees the weights as the form keeps them.
        let kept: Vec<Tensor> = (weights.iter())
            .map(|t| Tensor::f32(t.shape.clone(), f.roundtrip(t.floats().unwrap())))
            .collect();
        let want = run(&g, &kept, 0, &inputs).unwrap();
        let device = Device::guess(&settings).pinned(Pins::all(f));
        let plan = lower(&g, 0, &sizes, &device).unwrap();
        let mut runner = loaded(plan, &g, &kept, &settings);
        let got = runner.run(&feeds).unwrap().remove(0);
        let (got, want) = (got.floats().unwrap(), want[0].floats().unwrap());
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let (off, cos) = (max_abs_diff(got, want) / scale, cosine_f32(got, want));
        eprintln!(
            "{} as {form} on {}: off {off:.2e}, cos {cos:.7}",
            spec.name,
            runner.device()
        );
        assert!(
            off < 1e-3 && cos > 0.9999,
            "{} as {form}: off {off:e} cos {cos}",
            spec.name
        );
    }
}
