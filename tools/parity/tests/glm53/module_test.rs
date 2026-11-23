//! The glm53-moe recipe's kernel: its plan, and its numbers.

use std::path::Path;
use tessera_codegen::trellis::packed;
use tessera_engine::Engine;
use tessera_engine::wasm::Module;
use tessera_engine_settings::{Backend, Pick, Settings};
use tessera_graph::DimName::{Context, Hidden, Picks, Rows, Segments, Seqs};
use tessera_graph::Parts;
use tessera_graph::{Composite, F16};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{Load, named};
use tessera_lower::{Cluster, Device, Limits, Pins, Platform, Rule, Weight, lower};
use tessera_models::mla::{Config, RotatedMoe, Routed, stepper};
use tessera_recipe::Recipe;
use tessera_run::Loopback;
use tessera_run::testing::{host, runs};
use tessera_testkit::{Noise, cosine_f32, max_abs_diff, weights};
use tessera_trace::{DType, Dim, EntryKind, Graph, Model, Port, Role, TensorType, Weights};

const NAME: &str = "glm53_moe";
const KERNELS: [&str; 5] = [
    "glm53_route",
    "glm53_plan",
    "glm53_fc1",
    "glm53_fc2",
    "glm53_finish",
];
const TEXT: &str = include_str!("../../../../examples/recipes/glm53-moe/model/src/moe.cu");
const ROWS: usize = 3;

fn small() -> Config {
    tessera_models::small::glm53()
}

fn recipe() -> Recipe {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/recipes/glm53-moe");
    Recipe::read(&dir).unwrap()
}

/// The recipe's rule, from its module built to wasm.
fn rule() -> &'static Rule {
    let recipe = recipe();
    let cache = tessera_testkit::scratch("recipes");
    let bytes = recipe.module_bytes(Some(&cache)).unwrap().unwrap();
    let module = Module::open(&bytes).unwrap();
    assert_eq!(module.rules.len(), 1);
    module.rules[0]
}

/// Experts in the release's layout, in codebook `book`.
fn pins(book: &str) -> Pins {
    let [codes, turned] = ["", "_rot"].map(|end| named(&format!("trellis_k4{book}{end}")).unwrap());
    Pins::default()
        .pin("*moe*.gate", codes)
        .pin("*moe*.up", codes)
        .pin("*moe*.down", codes)
        .pin("*shared*", turned)
}

/// The lowering choices serve hands a generate engine.
fn choices() -> tessera_lower::Recipe {
    let mut lower = recipe().lower.unwrap();
    lower.rules.push(rule());
    lower.pins = pins("");
    lower
}

#[derive(Weights)]
struct Block {
    moe: RotatedMoe,
}

/// One routed block of `c` alone, over rows.
fn block_of(c: &Config) -> Graph {
    let rows = Dim::bounded(Rows, 16);
    let m = Model::new(c.dims().into_iter().chain([rows]).collect());
    let w: Block = tessera_trace::declare(&m);
    let ty = TensorType::new([Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let [x] = m.entry(EntryKind::Forward, [Port { role, ty }]);
    let [y] = m.composite(Composite::Moe, || [w.moe.apply(&m, c, &x)]);
    m.finish([(Role::Logits, &y)])
}

/// Random codes for `[.., out, inp]`, with scales if rotated.
pub(super) fn codes(shape: &[usize], form: &dyn Load, draws: &mut Noise) -> Vec<u8> {
    codes_side(shape, form, draws, 1)
}

fn codes_side(shape: &[usize], form: &dyn Load, draws: &mut Noise, side: u32) -> Vec<u8> {
    let [.., out, inp] = *shape else {
        panic!("a matrix, or a set of matrices");
    };
    let count = form.bytes(shape.iter().product()) - form.bytes(0);
    let entries: Vec<u8> = (0..count)
        .map(|_| ((draws.draw() + 1.0) * 128.0) as u8)
        .collect();
    let mut bytes = packed(inp, out, side, &entries);
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
        let side = if matches!(w.path.last(), Some(tessera_graph::PathSeg::Field(f)) if f == "down")
            && t.shape.len() == 3
        {
            tessera_codegen::trellis::INPUT_SIDE
        } else {
            1
        };
        let bytes = codes_side(&t.shape, form, draws, side);
        let decoded = Tensor::f32(t.shape, form.unpack_all(&bytes));
        out.push((decoded, Some(Weight::Packed(form, bytes))));
    }
    out
}

/// Parameters of a kernel's entry, by its commas.
fn parameters(entry: &str) -> usize {
    let from = TEXT.find(&format!("void {entry}(")).unwrap();
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
    let kernels: Vec<&str> = taken.iter().map(|s| s.kernel.name()).collect();
    // The dense layer keeps the linked rules.
    let routed = (c.stack.layers - c.dense_layers) as usize;
    assert_eq!(kernels, KERNELS.repeat(routed));
    for step in taken.iter().filter(|s| s.kernel.name() == KERNELS[2]) {
        let launches = step.launches();
        let [launch] = launches.as_slice() else {
            panic!("one launch a step");
        };
        assert_eq!(launch.args.len(), parameters(KERNELS[2]));
        let counts: Vec<u32> = launch.args.iter().filter_map(|a| a.u32()).collect();
        let sizes = [ROWS as u32, 128, 128, 2, 4, 4, id, id, 0, 128, 1];
        assert_eq!(counts, sizes);
    }
}

/// Each rank's plan takes its share of every routed block.
#[test]
fn module_takes_shards() {
    for (ffn, tp) in [(256, false), (512, true)] {
        let mut c = small();
        c.stack.ffn = ffn;
        let step = stepper::<RotatedMoe>(&c);
        let sharded = tessera_graph::shard_mode(&step, &Parts::even(2), &|_, _, _, _| true, tp)
            .unwrap()
            .graph;
        let mut device = Device::default()
            .adding(rule())
            .pinned(pins("_mul1"))
            .favoring(NAME)
            .zoned(Cluster::two("a", "b"))
            .ranked(vec![0, 1]);
        device.platform = Platform::assumed(Limits::assumed());
        let routed = (c.stack.layers - c.dense_layers) as usize;
        let sizes = [(Seqs, 2), (Rows, ROWS as u64), (Context, 16)];
        for rank in 0..2u32 {
            let plan = lower(&sharded, 0, &sizes, &device.clone().as_rank(rank)).unwrap();
            let planned = plan.steps.iter().filter(|s| s.kernel.name() == KERNELS[1]);
            let mut taken = 0;
            for step in planned {
                let launches = step.launches();
                let counts: Vec<u32> = launches[0].args.iter().filter_map(|a| a.u32()).collect();
                let want = if tp { [0, 32] } else { [16 * rank, 16] };
                assert_eq!(counts[3..], want);
                taken += 1;
            }
            assert_eq!(taken, routed);
            let sums = plan
                .steps
                .iter()
                .filter(|s| s.kernel.name() == "all_reduce");
            assert!(sums.count() >= routed);
        }
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
    let trimmed = tessera_graph::trim(&step, 0).unwrap();
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
    let named = |s: &&tessera_lower::Step| s.kernel.name() == KERNELS[1];
    assert_eq!(plan.steps.iter().filter(named).count(), routed);
    // The host plans without it: the kernel is CUDA text.
    for settings in runs() {
        let engine = Engine::open(step.clone(), 16, &choices, &settings).unwrap();
        let wanted = (settings.gpu() == Some(Backend::Cuda)).then_some(&routed);
        assert_eq!(
            engine.kernels().get(KERNELS[1]),
            wanted,
            "{}",
            engine.device()
        );
    }
}

const CASES: [(&str, bool, bool); 4] = [
    ("", false, false),
    ("_mul1", false, false),
    ("_mul1", true, false),
    ("_mul1", false, true),
];

/// On a CUDA GPU alone: the kernel is CUDA text.
#[test]
fn module_matches_interp() {
    // The release stores its router as bf16.
    for (book, bf16, tp) in CASES {
        matches_interp(book, bf16, tp);
    }
}

/// `v` at bf16 precision, rounded to nearest even.
fn bf16(v: f32) -> f32 {
    let bits = v.to_bits();
    f32::from_bits((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000)
}

/// The kernel against the interpreter, with codes in codebook `book`.
fn matches_interp(book: &str, router_bf16: bool, tp: bool) {
    let mut c = small();
    if tp {
        c.stack.hidden = 1024;
        c.stack.ffn = 512;
    }
    let g = block_of(&c);
    let mut draws = Noise(7);
    let mut stored = stored(&g, &mut draws, book);
    let mut pins = pins(book);
    if router_bf16 {
        pins = pins.pin("*moe*.router", named("bf16").unwrap());
        let at = g
            .weights
            .iter()
            .position(|w| pins.form(&w.path).name() == "bf16")
            .unwrap();
        let vals = stored[at]
            .0
            .floats()
            .unwrap()
            .iter()
            .map(|v| bf16(*v))
            .collect();
        stored[at].0 = Tensor::f32(stored[at].0.shape.clone(), vals);
    }
    let floats: Vec<Tensor> = stored.iter().map(|(t, _)| t.clone()).collect();
    let hidden = c.stack.hidden as usize;
    let sharded = tp.then(|| {
        tessera_graph::shard_mode(&g, &Parts::even(2), &|_, _, _, _| true, true)
            .unwrap()
            .graph
    });
    for &rows in if tp {
        &[1usize, 8, 12][..]
    } else {
        &[ROWS][..]
    } {
        let gain = if tp { 0.25 } else { 1.0 };
        let x = Tensor::f32(
            vec![rows, hidden],
            (0..rows * hidden).map(|_| draws.draw() * gain).collect(),
        );
        let want = run(&g, &floats, 0, std::slice::from_ref(&x)).unwrap();
        let want = want[0].floats().unwrap();
        for settings in runs().iter().filter(|s| s.gpu() == Some(Backend::Cuda)) {
            let mut dev = Device::guess(settings).adding(rule()).pinned(pins.clone());
            let mut plans = Vec::new();
            if let Some(ref sg) = sharded {
                dev = dev
                    .favoring(NAME)
                    .zoned(Cluster::two("a", "b"))
                    .ranked(vec![0, 1]);
                for r in 0..2u32 {
                    plans.push(
                        lower(sg, 0, &[(Rows, rows as u64)], &dev.clone().as_rank(r)).unwrap(),
                    );
                }
                assert!(plans.iter().all(|p| p.steps.iter().any(|s| s.rule == NAME)));
            } else {
                let p = lower(&g, 0, &[(Rows, rows as u64)], &dev).unwrap();
                assert_eq!(
                    p.steps.iter().map(|s| s.rule).collect::<Vec<_>>(),
                    [NAME; 5]
                );
                plans.push(p);
            }
            let mut r = Loopback::open(&plans, settings).unwrap();
            for (i, (f, p)) in stored.iter().enumerate() {
                match p {
                    Some(p) => r.load_weight(i as u32, p).unwrap(),
                    None => r.load(i as u32, f.view()).unwrap(),
                }
            }
            let got = r.run(&[x.view()]).unwrap().remove(0);
            let got = got.floats().unwrap();
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let (off, cos) = (max_abs_diff(got, want) / scale, cosine_f32(got, want));
            eprintln!("{book} tp={tp} rows={rows}: off {off:e} cos {cos}");
            assert!(off < 1e-3 && cos > 0.9999, "off {off:e} cos {cos}");
        }
    }
}
