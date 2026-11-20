//! The per-model suite: what every model test runs alike.

// Linking these registers their lowering rules.
use sensai_rules as _;

mod golden;
mod gpu;
mod shard;
mod step;

pub use golden::{Case, Check, Compare, Golden, LOGITS, Reference, Tolerance, golden};
pub use golden::{pulled, turn, unit};
pub use gpu::gpu;
pub use shard::{pair, shards, step_ranks};
pub use step::steps;

use sensai_bundle::names::{repo, resolve};
use sensai_checkpoint::{Slice, Stored};
use sensai_interp::{Tensor, run};
use sensai_lower::form::named;
use sensai_lower::{Device, Pick, Pins, Plan, Settings, View, lower, rules};
use sensai_run::testing::{host, runs};
use sensai_run::{Run, open};
use sensai_testkit::{Noise, cosine_f32, fill, max_abs_diff};
use sensai_trace::DimName::{Context, Seqs};
use sensai_trace::{Composite, DimName, Graph, TensorType};

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
    /// The step entries of a model that decodes.
    pub stepper: Option<Stepper>,
    /// Real files' slices, each with the graph it feeds.
    pub files: &'static [Manifest],
    /// Outputs a reference recorded.
    pub golden: Option<Golden>,
}

/// Step graphs: one beside `small`, one at release sizes.
pub struct Stepper {
    pub small: fn() -> Graph,
    pub release: fn() -> Graph,
    /// How far steps may drift from the forward pass.
    pub drift: f32,
}

/// Steps and the forward pass sum in the same order.
pub const DRIFT: f32 = 1e-5;

/// Absorbed MLA attention sums in another order.
pub const MLA_DRIFT: f32 = 1e-4;

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

/// A manifest's text and the graph its file feeds.
pub type Manifest = (&'static str, fn() -> Graph);

/// A decoder's graphs: `decoder!(HeadNorm, Mlp, small config, release config fn)`.
#[macro_export]
macro_rules! decoder {
    ($norm:ty, $mlp:ty, $small:expr, $release:expr) => {
        fn small() -> Graph {
            program::<$norm, $mlp>(&$small)
        }

        fn release() -> Graph {
            program::<$norm, $mlp>(&$release(8, 40_960))
        }

        const STEPPER: Stepper = Stepper {
            small: || stepper::<$norm, $mlp>(&$small),
            release: || stepper::<$norm, $mlp>(&$release(1, 1024)),
            drift: $crate::DRIFT,
        };
    };
}

/// The tests every model runs: `suite!(SPEC)`.
#[macro_export]
macro_rules! suite {
    ($spec:expr) => {
        #[test]
        fn traces() {
            $crate::traces(&$spec);
        }

        #[test]
        fn files() {
            $crate::files(&$spec);
        }

        #[test]
        fn faux_weights() {
            $crate::faux(&$spec);
        }

        #[test]
        fn faux_rules() {
            $crate::faux_rules(&$spec);
        }

        #[test]
        fn golden() {
            $crate::golden(&$spec);
        }

        #[test]
        fn steps() {
            $crate::steps(&$spec);
        }

        #[test]
        fn gpu() {
            $crate::gpu(&$spec);
        }

        #[test]
        fn shards() {
            $crate::shards(&$spec);
        }

        #[test]
        fn native() {
            $crate::native(&$spec);
        }
    };
}

/// The release graph survives JSON, verifies and has its sizes.
pub fn traces(spec: &Spec) {
    let g = (spec.release)();
    let back = Graph::from_json(&g.to_json()).unwrap();
    assert_eq!(back, g);
    sensai_graph::verify(&back).unwrap();
    let marked = |kind| g.regions.iter().filter(|r| r.kind == kind).count();
    for &(kind, n) in spec.regions {
        assert_eq!(marked(kind), n, "{} {kind:?} regions", spec.name);
    }
    let all: usize = spec.regions.iter().map(|(_, n)| n).sum();
    assert_eq!(g.regions.len(), all, "{} regions", spec.name);
    let count =
        |w: &sensai_graph::Weight| w.ty.shape.elements(&g.dims).unwrap().as_const().unwrap();
    let parameters: u64 = g.weights.iter().map(count).sum();
    assert_eq!(parameters, spec.parameters, "{} parameters", spec.name);
}

/// A real file's slices, from `tessera-eval --slices` output.
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
const FAUX: [(DimName, u64); 2] = [(Seqs, 1), (Context, ROWS)];

/// Faux weights and inputs for `g`, at the faux sizes.
fn faux_data(g: &Graph) -> (Vec<Tensor>, Vec<Tensor>) {
    let mut noise = Noise(17);
    let mut draw = |ty: &TensorType| fill(g, ty, &FAUX, &mut noise).unwrap();
    let weights: Vec<Tensor> = g.weights.iter().map(|w| draw(&w.ty)).collect();
    let ports = &g.entries[0].inputs;
    let inputs = ports.iter().map(|p| draw(&p.ty));
    (weights, inputs.collect())
}

/// One plan on the faux data against the interpreter.
fn faux_check(
    label: &str,
    g: &Graph,
    data: &(Vec<Tensor>, Vec<Tensor>),
    settings: &Settings,
    device: Device,
) -> Box<dyn Run> {
    let (weights, inputs) = data;
    let form = device.pins.form(&[]);
    // The interpreter sees the weights after the form's round trip.
    let kept: Vec<Tensor> = (weights.iter())
        .map(|t| Tensor::f32(t.shape.clone(), form.roundtrip(t.floats().unwrap())))
        .collect();
    let want = run(g, &kept, 0, inputs).unwrap();
    let plan = lower(g, 0, &FAUX, &device).unwrap();
    let mut runner = loaded(plan, g, &kept, settings);
    let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
    let got = runner.run(&feeds).unwrap().remove(0);
    let (got, want) = (got.floats().unwrap(), want[0].floats().unwrap());
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let (off, cos) = (max_abs_diff(got, want) / scale, cosine_f32(got, want));
    eprintln!(
        "{label} on {}: off {off:.2e}, cos {cos:.7}",
        runner.device()
    );
    assert!(off < 1e-3 && cos > 0.9999, "{label}: off {off:e} cos {cos}");
    runner
}

/// Random weights on native CPU kernels, against the interpreter.
pub fn native(spec: &Spec) {
    let Some(small) = spec.small else { return };
    let g = small();
    let data = faux_data(&g);
    let cpu = Settings {
        backend: Pick::Cpu,
        ..host()
    };
    for form in ["f32", "bf16", "q8_0", "q4_0", "nvfp4"] {
        let device = Device::guess(&cpu).pinned(Pins::all(named(form).unwrap()));
        // A form the rows cannot take is not this test's.
        let Ok(plan) = lower(&g, 0, &FAUX, &device) else {
            continue;
        };
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        let native = names.iter().filter(|n| n.starts_with("cpu_")).count();
        assert!(native > 0, "{}: no native kernels in {names:?}", spec.name);
        let label = format!("{} native as {form}", spec.name);
        let runner = faux_check(&label, &g, &data, &cpu, device);
        let (compiled, made) = runner.compiled();
        assert_eq!(compiled, made, "{label}: every tree compiles");
    }
}

/// Random weights in every form: device against interpreter.
pub fn faux(spec: &Spec) {
    let Some(small) = spec.small else { return };
    let g = small();
    let data = faux_data(&g);
    let forms = ["f32", "bf16", "q8_0", "q4_0", "nvfp4"];
    for (settings, form) in runs().iter().flat_map(|s| forms.map(|f| (s, f))) {
        let device = Device::guess(settings).pinned(Pins::all(named(form).unwrap()));
        faux_check(
            &format!("{} as {form}", spec.name),
            &g,
            &data,
            settings,
            device,
        );
    }
}

/// Faux weights as f32 and bf16, favoring each rule.
pub fn faux_rules(spec: &Spec) {
    let Some(small) = spec.small else { return };
    let g = small();
    let data = faux_data(&g);
    let runs = runs();
    let pairs = runs
        .iter()
        .flat_map(|s| rules().into_iter().map(move |r| (s, r)));
    for (settings, rule) in pairs {
        for form in ["f32", "bf16"] {
            let pins = Pins::all(named(form).unwrap());
            let device = Device::guess(settings).favoring(rule.name).pinned(pins);
            let label = format!("{} favoring {} as {form}", spec.name, rule.name);
            faux_check(&label, &g, &data, settings, device);
        }
    }
}
