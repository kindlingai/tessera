//! The per-model suite: what every model test runs alike.

// Linking these registers their lowering rules.
use kernels as _;
use linear as _;

use bundle::names::{repo, resolve};
use checkpoint::{Slice, Stored};
use gpukit::testing::{Noise, cosine_f32, max_abs_diff};
use interp::{Tensor, run};
use lower::form::named;
use lower::{Device, Pins, View, lower, open};
use settings::read as settings;
use trace::DimName::{Context, Seqs};
use trace::{Composite, DType, DimName, Graph, TensorType};

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
fn slices(json: &str) -> Vec<Slice> {
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

/// Random floats, or ids below 8, shaped like `ty`.
fn fill(g: &Graph, ty: &TensorType, noise: &mut Noise) -> Tensor {
    let sizes = [(Seqs, 1), (Context, ROWS)];
    let size = |name: DimName| sizes.iter().find(|(n, _)| *n == name).map(|(_, s)| *s);
    let dim = |e: &graph::DimExpr| e.poly(&g.dims).unwrap().eval(&size).unwrap() as usize;
    let shape: Vec<usize> = ty.shape.0.iter().map(dim).collect();
    let mut values = (0..shape.iter().product()).map(|_| noise.draw());
    match ty.dtype {
        DType::U32 => Tensor::u32(shape, values.map(|v| ((v + 1.0) * 4.0) as u32).collect()),
        _ => Tensor::f32(shape, values.by_ref().collect()),
    }
}

/// Random weights in every form: device against interpreter.
pub fn faux(spec: &Spec) {
    let Some(small) = spec.small else { return };
    let g = small();
    let mut noise = Noise(17);
    let weights: Vec<Tensor> = (g.weights.iter())
        .map(|w| fill(&g, &w.ty, &mut noise))
        .collect();
    let ports = &g.entries[0].inputs;
    let inputs: Vec<Tensor> = ports.iter().map(|p| fill(&g, &p.ty, &mut noise)).collect();
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
        let plan = lower(&g, 0, &[(Seqs, 1), (Context, ROWS)], &device).unwrap();
        let mut runner = open(plan, &settings).unwrap();
        for (at, w) in kept.iter().enumerate() {
            runner.load(at as u32, w.view()).unwrap();
        }
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
