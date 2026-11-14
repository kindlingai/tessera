//! The per-model suite: what every model test runs alike.

// Linking these registers their lowering rules.
use kernels as _;
use linear as _;

use bundle::names::{repo, resolve};
use checkpoint::{Slice, Stored};
use interp::{Tensor, run};
use lower::form::named;
use lower::{Device, Form, Pins, Plan, Settings, View, lower};
use run::{Run, open};
use settings::read as settings;
use std::collections::BTreeMap;
use std::path::Path;
use testkit::{Noise, argmax, cosine_f32, fill, max_abs_diff};
use trace::DimName::{Context, Seqs};
use trace::{Composite, DimName, Graph, TensorType};

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

/// A recorded reference, and how closely to match it.
pub struct Golden {
    /// The weights and cases, where this host has them.
    pub read: fn(&Graph) -> Option<Reference>,
    /// The form the plan keeps weights in.
    pub form: &'static dyn Form,
    pub tolerance: Tolerance,
    /// Rows are tokens when set, else one row.
    pub per_token: bool,
    /// The output is scaled to unit length first.
    pub unit: bool,
    /// The plan runs on a host with no GPU too.
    pub anywhere: bool,
}

/// What a reference holds: weights, and inputs with outputs.
pub struct Reference {
    pub weights: Vec<Tensor>,
    pub cases: Vec<Case>,
}

/// Sequences of one length, and the output recorded for them.
pub struct Case {
    pub name: String,
    pub seqs: usize,
    pub tokens: Vec<u32>,
    pub want: Vec<f32>,
}

impl Case {
    /// One sequence.
    pub fn one(name: &str, tokens: Vec<u32>, want: Vec<f32>) -> Self {
        let name = name.to_string();
        Self {
            name,
            seqs: 1,
            tokens,
            want,
        }
    }

    pub fn inputs(&self) -> [Tensor; 2] {
        inputs(self.seqs, &self.tokens)
    }

    pub fn sizes(&self) -> [(DimName, u64); 2] {
        let len = self.tokens.len() / self.seqs;
        [(Seqs, self.seqs as u64), (Context, len as u64)]
    }
}

/// Tokens and positions, `[seqs, len]` each.
pub fn inputs(seqs: usize, tokens: &[u32]) -> [Tensor; 2] {
    let len = tokens.len() / seqs;
    let shape = vec![seqs, len];
    let positions = (0..seqs).flat_map(|_| 0..len as u32).collect();
    let ids = |data| Tensor::u32(shape.clone(), data);
    [ids(tokens.to_vec()), ids(positions)]
}

/// How far an output may stray from the reference.
pub struct Tolerance {
    pub max_abs: f32,
    /// The least cosine any row may have.
    pub cos: f32,
    /// Each row's largest entry must match.
    pub top: bool,
}

impl Tolerance {
    pub fn check(&self, label: &str, got: &[f32], want: &[f32], rows: usize) {
        let width = got.len() / rows;
        let (max, mut cos) = (max_abs_diff(got, want), 1.0f32);
        for (g, w) in got.chunks(width).zip(want.chunks(width)) {
            cos = cos.min(cosine_f32(g, w));
            assert!(!self.top || argmax(g) == argmax(w), "{label}: top differs");
        }
        eprintln!("{label}: max_abs {max:.3e} worst cos {cos:.7}");
        assert!(
            max < self.max_abs && cos > self.cos,
            "{label}: {max:.3e} {cos}"
        );
    }
}

impl Golden {
    /// `got` must match what the case recorded.
    pub fn close(&self, label: &str, got: &[f32], case: &Case) {
        let rows = if self.per_token { case.tokens.len() } else { 1 };
        let scaled = if self.unit { unit(got) } else { got.to_vec() };
        self.tolerance.check(label, &scaled, &case.want, rows);
    }
}

pub fn unit(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

/// Ids from a text file of numbers.
pub fn tokens(path: &Path) -> Vec<u32> {
    let text = std::fs::read_to_string(path).unwrap();
    text.split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect()
}

/// Floats from a file of little-endian f32.
pub fn floats(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let wide = |b: &[u8; 4]| f32::from_le_bytes(*b);
    bytes.as_chunks::<4>().0.iter().map(wide).collect()
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

/// The interpreter and the plan against the reference.
pub fn golden(spec: &Spec) {
    let Some(golden) = &spec.golden else { return };
    let g = (spec.release)();
    let Some(reference) = (golden.read)(&g) else {
        return eprintln!("{}: no reference here; skipped", spec.name);
    };
    for case in &reference.cases {
        let out = interpreted(&g, &reference.weights, case.seqs, &case.tokens);
        golden.close(&format!("{} cpu", case.name), &out, case);
    }
    let settings = settings();
    if settings.gpu().is_none() && !golden.anywhere {
        return eprintln!("no GPU here; skipped");
    }
    let device = Device::guess(&settings).pinned(Pins::all(golden.form));
    let mut runner: Option<Box<dyn Run>> = None;
    for case in &reference.cases {
        let plan = lower(&g, 0, &case.sizes(), &device).unwrap();
        let r = match runner.as_mut() {
            Some(r) => r,
            None => runner.insert(loaded(plan.clone(), &g, &reference.weights, &settings)),
        };
        r.replan(&plan).unwrap();
        let inputs = case.inputs();
        let out = r.run(&[inputs[0].view(), inputs[1].view()]).unwrap();
        let label = format!("{} on {}", case.name, r.device());
        golden.close(&label, out[0].floats().unwrap(), case);
    }
}
