//! Each rule's plans, and its launches against the CPU interpreter.

use tessera_attention as _;
use tessera_kernels as _;
use tessera_linear as _;

use tessera_graph::DimName::{Context, HeadDim, Heads, Rows, Seqs};
use tessera_graph::{DType, Dim, DimName, Graph, NodeId, verify};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM};
use tessera_lower::{Device, Limits, Pins, Plan, Platform, Settings, View, cut, lower, rules};
use tessera_run::open;
use tessera_run::testing::{host, runs};
use tessera_testkit::graphs::{self, floats, ids};
use tessera_testkit::{Noise, fill, max_abs_diff};
use tessera_trace::{Act, EntryKind, Model, Pairs, Role};

const SIZES: [(DimName, u64); 3] = [(Seqs, 2), (Rows, 3), (Context, 5)];

/// The bounds every shared graph here is built with.
fn dims() -> Vec<Dim> {
    vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::bounded(Context, 16),
    ]
}

/// Runs entry 0 both ways. Returns the largest difference.
fn gap(g: &Graph, inputs: &[Tensor], device: &Device, settings: &Settings) -> f32 {
    let want = run(g, &[], 0, inputs).unwrap();
    let mut runner = open(lower(g, 0, &SIZES, device).unwrap(), settings).unwrap();
    let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
    let got = runner.run(&feeds).unwrap().remove(0);
    if want[0].dtype() != DType::U32 {
        return max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
    }
    // Indices must match exactly.
    let same = got.indices().unwrap() == want[0].indices().unwrap();
    if same { 0.0 } else { f32::INFINITY }
}

/// The rules this crate registers.
fn ours() -> Vec<&'static tessera_lower::Rule> {
    let ours = rules().into_iter();
    ours.filter(|r| r.module.starts_with("tessera_kernels::"))
        .collect()
}

/// Every rule's launches under `settings`. Returns the rules used.
fn rules_match(settings: &Settings) -> Vec<&'static str> {
    let d = dims();
    // Rows of 600: more entries than threads for the argmax.
    let graphs = [
        graphs::rows(&d, 32),
        graphs::rows(&d, 600),
        graphs::attention(&d, [2, 2, 16]),
        graphs::rotation(&d),
        graphs::partial_rope(&d, 32),
        graphs::table(&d, 32, 16),
        graphs::neighbours(&d, [6, 9, 16]),
        graphs::picks(&d, 16),
        graphs::indexed(&d, [2, 16, 2]),
        graphs::indexed_rows(&d, [2, 16, 2]),
        graphs::residual_norm(&d, 32),
        graphs::residual_norm(&d, 600),
    ];
    let ours = ours();
    let mut used: Vec<&str> = Vec::new();
    for (g, rule) in graphs.iter().flat_map(|g| ours.iter().map(move |r| (g, r))) {
        let mut noise = Noise(7);
        let device = Device::guess(settings).favoring(rule.name);
        let plan = lower(g, 0, &SIZES, &device).unwrap();
        let mut seen: Vec<&[NodeId]> = Vec::new();
        for step in &plan.steps {
            used.extend((step.rule == rule.name).then_some(rule.name));
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
                .map(|p| fill(&sub, &p.ty, &SIZES, &mut noise).unwrap())
                .collect();
            let off = gap(&sub, &inputs, &device, settings);
            assert!(off < 1e-4, "{:?} is off by {off:e}", step.kernel);
        }
    }
    used
}

#[test]
fn rules_match_cpu() {
    for settings in runs() {
        let used = rules_match(&settings);
        // A new rule needs a graph in `testkit::graphs`.
        for rule in ours() {
            let name = rule.name;
            assert!(used.contains(&name), "{name} is never exercised");
        }
    }
}

/// Two rotations by one set of positions, then a map.
fn rotations() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::fixed(Heads, 2),
        Dim::fixed(HeadDim, 16),
    ]);
    let port = || floats([Seqs, Rows, Heads, HeadDim]);
    let [x, y, at] = m.entry(EntryKind::Forward, [port(), port(), ids([Seqs, Rows])]);
    let turn = |t: &tessera_trace::Tensor| t.rope(&at, 1e4, Pairs::Halves);
    m.finish([(Role::Logits, &turn(&x).add(&turn(&y)).act(Act::Silu))])
}

fn count(plan: &Plan, kernel: &str) -> usize {
    let named = |s: &&tessera_lower::Step| s.kernel.name() == kernel;
    plan.steps.iter().filter(named).count()
}

fn names(plan: &Plan) -> Vec<&'static str> {
    plan.steps.iter().map(|s| s.kernel.name()).collect()
}

#[test]
fn rotations_share_angles() {
    let plan = lower(&rotations(), 0, &SIZES, &Device::guess(&host())).unwrap();
    assert_eq!((count(&plan, "angles"), count(&plan, "rope")), (1, 2));
}

#[test]
fn favoring_stays_local() {
    // Rope has no competing rule, so favoring it changes nothing.
    let g = rotations();
    let plain = lower(&g, 0, &SIZES, &Device::guess(&host())).unwrap();
    let favored = Device::guess(&host()).favoring("rope");
    assert!(
        plain == lower(&g, 0, &SIZES, &favored).unwrap(),
        "a favored rule skewed its readers' choice"
    );
}

#[test]
fn trips_are_counted() {
    // Walking a row opens a line every sixteen floats.
    let norm = tessera_kernels::RmsNorm {
        rows: 4,
        width: 64,
        eps: 1e-5,
        weighted: true,
    };
    let trip = tessera_lower::Kernel::load(&norm, 64).unwrap().trip;
    assert_eq!((trip.loads, trip.stores), (3.0, 1.0));
    assert!((trip.fresh - 1.0 / 16.0).abs() < 1e-9, "{trip:?}");
    assert_eq!(trip.special, 0.0, "the root lies outside the loops");
}

#[test]
fn limits_bar_kernels() {
    let g = graphs::rows(&dims(), 32);
    let guess = Device::guess(&host()).favoring("lane_norm");
    assert!(names(&lower(&g, 0, &SIZES, &guess).unwrap()).contains(&"lane_norm"));
    // A 64-lane device runs none of the 32-lane kernels.
    let wide = Limits {
        lanes: 64,
        ..Limits::assumed()
    };
    let narrow = guess.on(Platform::assumed(wide));
    let picked = names(&lower(&g, 0, &SIZES, &narrow).unwrap());
    assert!(picked.contains(&"rms_norm") && !picked.contains(&"lane_norm"));
    // An uncalibrated GPU picks lane_norm on wide rows.
    let gpu = Platform {
        backend: "cuda".into(),
        name: "Test GPU".into(),
        units: 128,
        bandwidth: 1e12,
        ..Platform::assumed(Limits::assumed())
    };
    let wide_rows = graphs::rows(&dims(), 600);
    let plan = lower(&wide_rows, 0, &SIZES, &Device::default().on(gpu)).unwrap();
    assert!(names(&plan).contains(&"lane_norm"));
}

#[test]
fn tables_keep_forms() {
    let g = graphs::table(&dims(), 32, 16);
    let guess = Device::guess(&host());
    let narrow = guess.clone().pinned(Pins::all(BF16_FORM));
    assert!(names(&lower(&g, 0, &SIZES, &narrow).unwrap()).contains(&"gather_bf16"));
    // A table pinned f32 stays f32.
    let pins = Pins::all(BF16_FORM).pin("embed", F32_FORM);
    let plan = lower(&g, 0, &SIZES, &guess.pinned(pins)).unwrap();
    assert_eq!(names(&plan), ["gather"]);
}
