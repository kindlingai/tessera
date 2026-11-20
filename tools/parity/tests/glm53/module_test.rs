//! The glm53-moe recipe's kernel: its plan, and its numbers.

use sensai_codegen::trellis::packed;
use sensai_engine::Engine;
use sensai_engine::wasm::Module;
use sensai_engine_settings::{Backend, Pick, Settings};
use sensai_graph::DimName::{Context, Hidden, Picks, Rows, Segments, Seqs};
use sensai_graph::{Composite, F16};
use sensai_interp::{Tensor, run};
use sensai_lower::form::{Load, named};
use sensai_lower::{Device, Limits, Pins, Platform, Rule, Weight, lower};
use sensai_models::mla::{Config, RotatedMoe, Routed, stepper};
use sensai_recipe::Recipe;
use sensai_run::open;
use sensai_run::testing::{host, runs};
use sensai_testkit::{Noise, cosine_f32, max_abs_diff, weights};
use sensai_trace::{DType, Dim, EntryKind, Graph, Model, Port, Role, TensorType, Weights};
use std::path::Path;

const NAME: &str = "glm53_moe";
const TEXT: &str = include_str!("../../../../examples/recipes/glm53-moe/model/src/moe.cu");
const ROWS: usize = 3;

fn small() -> Config {
    sensai_models::small::glm53()
}

fn recipe() -> Recipe {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/recipes/glm53-moe");
    Recipe::read(&dir).unwrap()
}

/// The recipe's rule, from its module built to wasm.
fn rule() -> &'static Rule {
    let recipe = recipe();
    let cache = sensai_testkit::scratch("recipes");
    let bytes = recipe.module_bytes(Some(&cache)).unwrap().unwrap();
    let module = Module::open(&bytes).unwrap();
    assert_eq!(module.rules.len(), 1);
    module.rules[0]
}

/// Experts as the release stores them, in codebook `book`.
fn pins(book: &str) -> Pins {
    let [codes, turned] = ["", "_rot"].map(|end| named(&format!("trellis_k4{book}{end}")).unwrap());
    Pins::default()
        .pin("*moe*.gate", codes)
        .pin("*moe*.up", codes)
        .pin("*moe*.down", codes)
        .pin("*shared*", turned)
}

/// The lowering choices serve hands a generate engine.
fn choices() -> sensai_lower::Recipe {
    let mut lower = recipe().lower.unwrap();
    lower.rules.push(rule());
    lower.pins = pins("");
    lower
}

#[derive(Weights)]
struct Block {
    moe: RotatedMoe,
}

/// One routed block alone, over rows.
fn block() -> Graph {
    let c = small();
    let rows = Dim::bounded(Rows, 16);
    let m = Model::new(c.dims().into_iter().chain([rows]).collect());
    let w: Block = sensai_trace::declare(&m);
    let ty = TensorType::new([Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let [x] = m.entry(EntryKind::Forward, [Port { role, ty }]);
    let [y] = m.composite(Composite::Moe, || [w.moe.apply(&m, &c, &x)]);
    m.finish([(Role::Logits, &y)])
}

/// Random codes for `[.., out, inp]`, with scales if rotated.
fn codes(shape: &[usize], form: &dyn Load, draws: &mut Noise) -> Vec<u8> {
    let [.., out, inp] = *shape else {
        panic!("a matrix, or a set of them");
    };
    let count = form.bytes(shape.iter().product()) - form.bytes(0);
    let entries: Vec<u8> = (0..count)
        .map(|_| ((draws.draw() + 1.0) * 128.0) as u8)
        .collect();
    let mut bytes = packed(inp, out, 1, &entries);
    if form.name().ends_with("_rot") {
        let scale = |i: usize| F16::from(1.0 + (i % 7) as f32 / 8.0).0.to_le_bytes();
        bytes.extend((0..inp + out).flat_map(scale));
    }
    bytes
}

/// Each weight as floats, and the pinned ones as stored.
fn stored(g: &Graph, draws: &mut Noise, book: &str) -> Vec<(Tensor, Option<Weight>)> {
    let pins = pins(book);
    let floats = weights(g, draws);
    let mut out = Vec::new();
    for (w, t) in g.weights.iter().zip(floats) {
        let form = pins.form(&w.path);
        if !form.name().starts_with("trellis") {
            out.push((t, None));
            continue;
        }
        let bytes = codes(&t.shape, form, draws);
        let decoded = Tensor::f32(t.shape, form.unpack_all(&bytes));
        out.push((decoded, Some(Weight::Packed(form, bytes))));
    }
    out
}

/// Parameters of the kernel's entry, by its commas.
fn parameters() -> usize {
    let from = TEXT.find(&format!("void {NAME}(")).unwrap();
    let upto = TEXT[from..].find(") {").unwrap();
    TEXT[from..from + upto].matches(',').count() + 1
}

#[test]
fn module_takes_routed() {
    for (book, id) in [("", 0), ("_mul1", 1)] {
        takes_routed(book, id);
    }
}

/// Each routed block takes the kernel, codebook in its counts.
fn takes_routed(book: &str, id: u32) {
    let c = small();
    let step = stepper::<RotatedMoe>(&c);
    // A compiler is assumed, so the CPU can plan.
    let mut device = Device::default()
        .adding(rule())
        .pinned(pins(book))
        .favoring(NAME);
    device.platform = Platform::assumed(Limits::assumed());
    let sizes = [(Seqs, 2), (Rows, ROWS as u64), (Context, 16)];
    let plan = lower(&step, 0, &sizes, &device).unwrap();
    let taken: Vec<_> = plan.steps.iter().filter(|s| s.rule == NAME).collect();
    // The dense layer keeps the linked rules.
    assert_eq!(taken.len() as u64, c.stack.layers - c.dense_layers);
    for step in taken {
        let launches = step.launches();
        let [launch] = launches.as_slice() else {
            panic!("one launch a block");
        };
        // Reads, the output, then the counts.
        assert_eq!(step.reads.len(), 23);
        assert_eq!(launch.args.len(), parameters());
        let counts: Vec<u32> = launch.args[24..].iter().filter_map(|a| a.u32()).collect();
        assert_eq!(counts, [ROWS as u32, 128, 128, 32, 2, 4, 4, id, id]);
    }
}

/// A generate engine plans the recipe's kernel.
#[test]
fn module_reaches_engine() {
    let c = small();
    let step = stepper::<RotatedMoe>(&c);
    let choices = choices();
    assert_eq!(choices.favor.as_deref(), Some(NAME));
    let routed = (c.stack.layers - c.dense_layers) as usize;
    // The engine's first launch, planned for an assumed GPU.
    let cuda = Settings {
        backend: Pick::Gpu(Backend::Cuda),
        ..host()
    };
    let device = Device::guess(&cuda).with(&choices);
    let trimmed = sensai_graph::trim(&step, 0).unwrap();
    let slots = trimmed.dims.iter().find(|d| d.name == Seqs).unwrap();
    let slots = slots.size.max();
    let sizes = [
        (Seqs, slots),
        (Rows, 1),
        (Context, 1),
        (Segments, 1),
        (Picks, 1),
    ];
    let plan = lower(&trimmed, 0, &sizes, &device).unwrap();
    let named = |s: &&sensai_lower::Step| s.kernel.name() == NAME;
    assert_eq!(plan.steps.iter().filter(named).count(), routed);
    // The host plans without it: the kernel is CUDA text.
    for settings in runs() {
        let engine = Engine::open(step.clone(), 16, &choices, &settings).unwrap();
        let wanted = (settings.gpu() == Some(Backend::Cuda)).then_some(&routed);
        assert_eq!(engine.kernels().get(NAME), wanted, "{}", engine.device());
    }
}

/// On a CUDA GPU alone: the kernel is CUDA text.
#[test]
fn module_matches_interp() {
    for book in ["", "_mul1"] {
        matches_interp(book);
    }
}

/// The kernel against the interpreter, with codes in codebook `book`.
fn matches_interp(book: &str) {
    let g = block();
    let mut draws = Noise(7);
    let stored = stored(&g, &mut draws, book);
    let floats: Vec<Tensor> = stored.iter().map(|(t, _)| t.clone()).collect();
    let hidden = small().stack.hidden as usize;
    let noise = (0..ROWS * hidden).map(|_| draws.draw()).collect();
    let x = Tensor::f32(vec![ROWS, hidden], noise);
    let want = run(&g, &floats, 0, std::slice::from_ref(&x)).unwrap();
    let want = want[0].floats().unwrap();
    for settings in runs().iter().filter(|s| s.gpu() == Some(Backend::Cuda)) {
        let device = Device::guess(settings).adding(rule()).pinned(pins(book));
        let plan = lower(&g, 0, &[(Rows, ROWS as u64)], &device).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.rule).collect();
        assert_eq!(names, [NAME]);
        let mut runner = open(plan, settings).unwrap();
        for (at, (floats, packed)) in stored.iter().enumerate() {
            match packed {
                Some(p) => runner.load_weight(at as u32, p).unwrap(),
                None => runner.load(at as u32, floats.view()).unwrap(),
            }
        }
        let got = runner.run(&[x.view()]).unwrap().remove(0);
        let got = got.floats().unwrap();
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let (off, cos) = (max_abs_diff(got, want) / scale, cosine_f32(got, want));
        eprintln!("{}{book}: off {off:e} cos {cos}", runner.device());
        assert!(off < 1e-3 && cos > 0.9999, "off {off:e} cos {cos}");
    }
}
