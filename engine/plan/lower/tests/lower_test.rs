//! Plans for a toy model, without a GPU.

use gpukit::testing::scratch;
use graph::DimName::{self, Context, Rows, Seqs};
use graph::{Act, Composite, Graph};
use kernels::Thin;
use linear::SCHEDULES;
use lower::form::{BF16_FORM, F32_FORM};
use lower::kernels::{Map, MapStage};
use lower::sites;
use lower::{
    Arg, Buffer, Device, Dialect, Error, Fit, Held, Kernel, KernelRef, Pins, Plan, Slot, Source,
    Step, T, Tree,
};
use lower::{assign, lower, peak, rules, schedule, select};
use models::qwen3::{Config, program, stepper};
use settings::read as settings;
use sim::Chip;

const SIZES: [(DimName, u64); 2] = [(Seqs, 2), (Context, 5)];

fn toy(device: &Device) -> (Graph, Plan) {
    let g = program(&Config::toy(4, 16));
    let plan = lower(&g, 0, &SIZES, device).unwrap();
    (g, plan)
}

/// The device with every tiled schedule barred.
fn untiled(device: Device) -> Device {
    let names = SCHEDULES.iter().map(|s| s.name);
    names.fold(device, Device::without)
}

fn count(plan: &Plan, kernel: &str) -> usize {
    let named = |s: &&lower::Step| s.kernel.name() == kernel;
    plan.steps.iter().filter(named).count()
}

fn gate() -> Map {
    Map(vec![MapStage::Act(Act::Silu), MapStage::Mul])
}

#[test]
fn gate_fuses() {
    // With the lanes barred, thin takes the gate's stages too.
    let plain = untiled(Device::guess(&settings())).without("lanes");
    let (g, plan) = toy(&plain);
    let alone = KernelRef::new(gate());
    assert!(plan.steps.iter().all(|s| s.kernel != alone));
    let gated = |s: &&lower::Step| s.rule == "thin" && s.covers.len() == 3;
    let steps: Vec<_> = plan.steps.iter().filter(gated).collect();
    assert_eq!(steps.len(), 2, "one per layer");
    for s in steps {
        assert_eq!(
            s.reads.len(),
            3,
            "the rows, the weights and the gate's operand"
        );
        let holds = |r: &&graph::Region| s.covers.iter().all(|c| r.start <= *c && *c < r.end);
        let region = g.regions.iter().find(holds).unwrap();
        assert_eq!(region.kind, Composite::Mlp);
    }
}

#[test]
fn buffers_are_shared() {
    let (g, plan) = toy(&Device::guess(&settings()));
    // Matmul schedules sum in their own orders.
    assert_eq!(plan.order(), graph::Order::PerConfig);
    assert_eq!(Plan::default().order(), graph::Order::Fixed);
    for s in &plan.steps {
        let in_place = s.reads.iter().any(|r| r.buffer == s.write.buffer);
        assert_eq!(in_place, s.kernel.overwrites());
    }
    let values = plan.weights.len() + plan.inputs.len() + plan.steps.len();
    assert!(
        plan.buffers.len() < values,
        "dead values free their buffers"
    );
    assert_eq!(count(&plan, "angles"), 1, "one table serves every rotation");
    let unsized_context = lower(&g, 0, &[(Seqs, 2)], &Device::guess(&settings()));
    assert!(matches!(unsized_context, Err(Error::Size(_))));
}

#[test]
fn cost_picks_the_matmul() {
    // Ten rows fit the thin kernel, so cost alone decides.
    let tiled_steps = |p: &Plan| {
        p.steps
            .iter()
            .filter(|s| s.kernel.name().starts_with("tiled_"))
            .count()
    };
    let (_, tiled) = toy(&Device::guess(&settings()).without("thin").without("lanes"));
    let (_, thin) = toy(&untiled(Device::guess(&settings())).without("lanes"));
    assert_eq!((count(&tiled, "thin"), tiled_steps(&thin)), (0, 0));
    assert_eq!(tiled_steps(&tiled), count(&thin, "thin"));
    let mut measured = Device::guess(&settings()).without("lanes");
    let cheap = thin.steps.iter().map(|s| (s, 1e-6));
    measured.fit(cheap.chain(tiled.steps.iter().map(|s| (s, 1e-3))));
    let (_, fitted) = toy(&measured);
    assert_eq!(tiled_steps(&fitted), 0, "the faster kernel wins");
    assert!(measured.cost(&fitted).seconds < measured.cost(&tiled).seconds);
}

#[test]
fn cost_picks_attention() {
    // Dense attention is unproven, so a guess streams.
    let (_, stream) = toy(&Device::guess(&settings()));
    assert_eq!((count(&stream, "attend"), count(&stream, "scores")), (2, 0));
    // The host cannot run the hand-written dense gemm.
    if settings().gpu().is_none() {
        return;
    }
    let (_, dense) = toy(&Device::guess(&settings()).favoring("dense"));
    assert_eq!((count(&dense, "attend"), count(&dense, "join")), (0, 0));
    let per_layer = ["scores", "causal", "values"].map(|k| count(&dense, k));
    assert_eq!(per_layer, [2, 2, 2]);
    let softmax = dense
        .steps
        .iter()
        .find(|s| s.kernel.name() == "causal")
        .unwrap();
    assert_eq!(softmax.write.buffer, softmax.reads[0].buffer, "in place");
}

#[test]
fn rule_names_differ() {
    // Rules sharing a name would share a cost curve.
    let names: Vec<&str> = rules().iter().map(|r| r.name).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(names, sorted, "rule names are unique and ordered");
    let g = program(&Config::toy(4, 16));
    let picked = select(&g, 0, &SIZES, &Device::guess(&settings())).unwrap();
    assert!(
        picked.contested.len() > 1,
        "some site saw more than one rule"
    );
}

#[test]
fn order_saves_memory() {
    let g = program(&Config::toy(4, 16));
    let picked = select(&g, 0, &SIZES, &Device::guess(&settings())).unwrap();
    let traced: Vec<usize> = (0..picked.ops.len()).collect();
    let order = schedule(&picked);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, traced, "every op runs once");
    assert!(peak(&picked, &order) <= peak(&picked, &traced));
    assert!(assign(&picked, &order).scratch() <= assign(&picked, &traced).scratch());
}

#[test]
fn dialects_agree() {
    let Source::Made(body) = gate().source() else {
        panic!("maps are generated");
    };
    let (metal, cuda) = (
        body.text().render(Dialect::Metal),
        body.text().render(Dialect::Cuda),
    );
    // Buffers, then scalars, in one order for both.
    assert!(metal.contains("device const float *a0 [[buffer(1)]]"));
    assert!(metal.contains("constant u32 &len [[buffer(3)]]"));
    let order = "const float *x,\n    const float *a0,\n    float *out,\n    u32 len\n";
    assert!(cuda.contains(order), "{cuda}");
    assert!(cuda.contains(&format!("__global__ void {}(", body.entry)));
    for body in [&metal, &cuda] {
        assert!(body.contains("v = (v * a0[(u64)gid]);"), "{body}");
    }
}

/// A toy step plan: `rows` new, `all` kept afterwards.
fn step(seqs: u64, rows: u64, all: u64) -> Result<Plan, Error> {
    let sizes = [(Seqs, seqs), (Rows, rows), (Context, all)];
    lower(
        &stepper(&Config::toy(4, 16)),
        0,
        &sizes,
        &Device::guess(&settings()),
    )
}

#[test]
fn kept_rows_fit() {
    let plan = step(1, 2, 5).unwrap();
    assert_eq!(plan.kept.len(), 2 * 2, "keys and values per layer");
    // Two key heads of eight, for up to sixteen rows.
    for (_, slot) in &plan.kept {
        assert_eq!((slot.len, plan.buffers[slot.buffer as usize]), (80, 1024));
    }
    let appends = plan.steps.iter().filter(|s| s.kernel.name() == "scatter");
    for s in appends.clone() {
        assert!(plan.kept.iter().any(|(_, slot)| *slot == s.write));
        let [launch] = &s.launches()[..] else {
            panic!("one scatter");
        };
        // Two rows of sixteen, into slots of sixteen rows.
        assert_eq!(launch.args[4..], [2, 16, 16].map(Arg::count));
    }
    assert_eq!((appends.count(), count(&plan, "argmax")), (4, 1));
    let scratch: usize = plan.buffers.iter().sum::<usize>() - 4 * 1024;
    assert!(plan.scratch() < scratch, "kept rows are not scratch");
    // Several slots fit, each holding the bound of rows.
    assert!(step(2, 2, 5).is_ok());
}

#[test]
fn replans_hold_buffers() {
    let (first, next) = (step(1, 3, 3).unwrap(), step(1, 1, 4).unwrap());
    let mut held = Held::default();
    let mut made = 0;
    let mut alloc = |zone: u32, bytes: usize| {
        made += 1;
        Ok(Buffer::host(zone, bytes))
    };
    held.bind(&first, &mut alloc).unwrap();
    let named = |p: &Plan| -> Vec<(u32, Slot)> {
        let weights = p.weights.iter().map(|l| (l.weight, l.slot));
        weights.chain(p.kept.iter().copied()).collect()
    };
    // The same memory, wherever the buffer list puts it.
    let at = |held: &Held, s: Slot| held.get(s).backing().mapped().unwrap().as_ptr();
    let before: Vec<_> = named(&first).iter().map(|(_, s)| at(&held, *s)).collect();
    held.bind(&next, &mut alloc).unwrap();
    let after: Vec<_> = named(&next).iter().map(|(_, s)| at(&held, *s)).collect();
    assert_eq!(named(&first).len(), named(&next).len());
    let owner = |p: &Plan| -> Vec<u32> { named(p).iter().map(|(index, _)| *index).collect() };
    assert_eq!(owner(&first), owner(&next));
    assert_eq!(before, after, "weights and kept rows stay put");
    assert!(made < first.buffers.len() + next.buffers.len());
    // Binding the same plan again allocates nothing.
    let so_far = made;
    held.bind(&next, |_, _| unreachable!("every buffer is held"))
        .unwrap();
    assert_eq!(made, so_far);
}

/// One thin launch of `m` rows, 64 outputs wide.
fn thin(m: usize) -> Step {
    let (n, k) = (64, 32);
    Step {
        kernel: KernelRef::new(Thin {
            m,
            n,
            k,
            form: lower::form::F32_FORM,
            after: sites::After::default(),
        }),
        rule: "thin",
        reads: Vec::new(),
        write: Slot::floats(0, m * n),
        covers: Vec::new(),
        zone: 0,
    }
}

#[test]
fn near_samples_estimate() {
    let seconds = |d: &Device, m: usize| d.seconds(&*thin(m).kernel, m * 64);
    let mut device = Device::guess(&settings());
    let guess = seconds(&device, 4);
    device.fit([(&thin(1), 1e-3), (&thin(16), 4e-3)]);
    assert_eq!(device.samples(), 2);
    assert!((seconds(&device, 1) - 1e-3).abs() < 1e-9, "an exact match");
    assert!((seconds(&device, 16) - 4e-3).abs() < 1e-9);
    // Between samples, each is scaled by work and they blend.
    let between = seconds(&device, 4);
    assert!(1e-3 < between && between < 4e-3 && between != guess);
    assert!((seconds(&device, 32) - 8e-3).abs() < 1e-4, "scaled by work");
    assert!(device.known(&thin(1)) && !device.known(&thin(4)));
    // A launch timed again replaces its sample.
    device.fit([(&thin(1), 2e-3), (&thin(1), 9e-3), (&thin(1), 3e-3)]);
    assert_eq!(device.samples(), 2);
    assert!((seconds(&device, 1) - 3e-3).abs() < 1e-9, "the median");
    assert!(seconds(&device.clone().without("thin"), 1).is_infinite());
}

#[test]
fn sources_differ() {
    let hash = |stages: Vec<MapStage>| Map(stages).source().hash();
    assert_eq!(hash(vec![MapStage::Add]), hash(vec![MapStage::Add]));
    assert_ne!(hash(vec![MapStage::Add]), hash(vec![MapStage::Mul]));
    assert_ne!(hash(Vec::new()), thin(1).kernel.source().hash());
}

/// Round constants for a made-up chip.
fn chip() -> Fit {
    Fit {
        floor: 2e-6,
        spawn: 1e-10,
        step: 10e-9,
        curve: 30e-9,
        rate: 1e12,
        wait: 300e-9,
        near: 40e-9,
        cache: 4e6,
        cached: 250e9,
        read: 100e9,
        scan: 80e9,
        write: 90e9,
        tile: 0.0,
    }
}

#[test]
fn models_fill_gaps() {
    let (step, fit) = (thin(1), chip());
    let seconds = |d: &Device| d.seconds(&*step.kernel, 64);
    let device = Device::guess(&settings()).fitted(fit);
    let load = step.kernel.load(64).unwrap();
    let modelled = fit.pool("fitted").estimate(&load).seconds;
    assert_eq!(seconds(&device), modelled);
    assert!(
        seconds(&Device::guess(&settings())) != modelled,
        "a bare guess differs"
    );
    let bound = device.bound(&*step.kernel, 64).unwrap();
    assert!(
        bound <= modelled
            && Device::guess(&settings())
                .bound(&*step.kernel, 64)
                .is_none()
    );
    // A sample of the kernel outranks the model.
    let mut timed = device.clone();
    timed.fit([(&step, 5e-3)]);
    assert_eq!(seconds(&timed), 5e-3);
    // The stored fit brings the model back.
    let file = scratch("tessera-lower-fit-test.json");
    device.save(&file).unwrap();
    let loaded = Device::load(&file).unwrap();
    std::fs::remove_file(&file).unwrap();
    assert!(loaded == device && seconds(&loaded) == modelled);
}

#[test]
fn trees_cut_shared() {
    let g = program(&Config::toy(4, 16));
    let tree = Tree::new(&g, 0, &SIZES, &Pins::default(), lower::kept::DENSE).unwrap();
    let chains = |want: &[MapStage]| -> Vec<(T, Vec<T>)> {
        let all = (0..tree.len() as u32).map(T);
        let chain = |t: T| match tree.site(t)? {
            sites::Chain { stages, reads, .. } if stages == want => Some((t, reads)),
            _ => None,
        };
        all.filter_map(chain).collect()
    };
    // The activation has one reader, so it joins the multiply.
    let gates = chains(&[MapStage::Act(Act::Silu), MapStage::Mul]);
    assert_eq!(gates.len(), 2, "one per layer");
    for (gate, reads) in &gates {
        let act = tree.node(*gate).args[0];
        assert!(!tree.root(act) && reads.len() == 2);
        assert_eq!(tree.covered(*gate, reads).len(), 2);
    }
    // The residual stream has two readers: a root.
    let adds = chains(&[MapStage::Add]);
    assert_eq!(adds.len(), 2 * 2, "two per layer");
    for (add, reads) in &adds {
        assert!(tree.root(reads[0]) && tree.covered(*add, reads).len() == 1);
    }
    let (logits, shape) = &tree.outputs[0];
    assert!(tree.root(*logits) && *shape == [2, 5, 32]);
}

#[test]
fn favoring_stays_local() {
    // No rule competes with rope, so favoring it changes nothing.
    let (_, plain) = toy(&Device::guess(&settings()));
    let (_, favored) = toy(&Device::guess(&settings()).favoring("rope"));
    assert!(
        plain == favored,
        "a favored rule skewed its readers' choice"
    );
}

#[test]
fn pins_match_paths() {
    use graph::PathSeg::{Field, Index};
    let path = |s: &str| -> Vec<graph::PathSeg> {
        let seg = |p: &str| p.parse().map_or(Field(p.into()), Index);
        s.split('.').map(seg).collect()
    };
    let pins = Pins::all(BF16_FORM)
        .pin("embed", F32_FORM)
        .pin("layers.*.ln*", F32_FORM)
        .pin("layers.1.*", BF16_FORM);
    let form = |p: &str| pins.form(&path(p)).name();
    assert_eq!(form("embed"), "f32");
    assert_eq!(form("layers.0.ln1"), "f32");
    assert_eq!(form("layers.0.q"), "bf16");
    assert_eq!(form("layers.1.ln1"), "bf16", "the last pin wins");
    assert_eq!(form("norm"), "bf16");
    assert_eq!(Pins::default().form(&path("layers.0.q")).name(), "f32");
    // Pins round trip through their names.
    let text = serde_json::to_string(&pins).unwrap();
    assert_eq!(serde_json::from_str::<Pins>(&text).unwrap(), pins);
}

#[test]
fn pinned_weights_narrow() {
    let g = program(&Config::toy(4, 16));
    let pins = Pins::all(BF16_FORM).pin("embed", F32_FORM);
    let plan = lower(&g, 0, &SIZES, &Device::guess(&settings()).pinned(pins)).unwrap();
    for l in &plan.weights {
        let embed = l.weight == 0;
        assert_eq!(l.slot.form == F32_FORM, embed, "weight {}", l.weight);
    }
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    for narrow in ["lanes_bf16", "widen", "gather"] {
        assert!(
            names.contains(&narrow),
            "{narrow} is missing from {names:?}"
        );
    }
    assert!(!names.contains(&"gather_bf16"), "the table stays f32");
    // The head reads the tied embedding, kept f32.
    let embed = plan.weights.iter().find(|l| l.weight == 0).unwrap();
    for s in plan.steps.iter().filter(|s| s.kernel.name() == "lanes") {
        assert_eq!(s.reads[1], embed.slot);
    }
    // Norm weights widen once each, however often they are read.
    let widened = names.iter().filter(|n| **n == "widen").count();
    let norms = g.weights.iter().filter(|w| w.ty.shape.0.len() == 1).count();
    assert_eq!(widened, norms);
    let (_, wide) = toy(&Device::guess(&settings()));
    assert!(plan.buffers.iter().sum::<usize>() < wide.buffers.iter().sum::<usize>());
}

#[test]
fn plans_are_cached() {
    let g = program(&Config::toy(4, 16));
    let mut cache = lower::Cache::default();
    let device = Device::guess(&settings());
    let first = cache.lower(&g, 0, &SIZES, &device).unwrap();
    let again = cache.lower(&g, 0, &SIZES, &device).unwrap();
    assert!(first == again && cache.hits == 1 && cache.len() == 1);
    // Other sizes, or a device that changed, lower afresh.
    cache
        .lower(&g, 0, &[(Seqs, 1), (Context, 5)], &device)
        .unwrap();
    let favored = device.clone().favoring("tiled");
    assert!(favored.generation() != device.generation());
    // Two devices changed alike are still two devices.
    let twice = device.clone().favoring("tiled");
    assert!(twice.generation() != favored.generation() && twice == favored);
    let tiled = cache.lower(&g, 0, &SIZES, &favored).unwrap();
    assert!(tiled != first && cache.hits == 1 && cache.len() == 3);
    let mut fitted = device.clone();
    fitted.fit([(&tiled.steps[0], 1e-3)]);
    assert!(fitted.generation() != device.generation());
    // Another graph at the same sizes is another plan.
    let step = stepper(&Config::toy(4, 16));
    let sizes = [(Seqs, 1), (Rows, 2), (Context, 5)];
    let stepped = cache.lower(&step, 0, &sizes, &device).unwrap();
    let forward = cache.lower(&g, 0, &sizes, &device).unwrap();
    assert!(stepped != forward && cache.len() == 5);
}

#[test]
fn trips_are_counted() {
    let trip = |k: &dyn lower::Kernel| k.load(64).unwrap().trip;
    // Walking a row opens a line every sixteen floats.
    let norm = trip(&kernels::RmsNorm {
        rows: 4,
        width: 64,
        eps: 1e-5,
    });
    assert_eq!((norm.loads, norm.stores), (3.0, 1.0));
    assert!((norm.fresh - 1.0 / 16.0).abs() < 1e-9, "{norm:?}");
    assert_eq!(norm.special, 0.0, "the root lies outside the loops");
    // One thread per element reads a new line each time.
    let map = trip(&Map(vec![MapStage::Add, MapStage::Act(Act::Silu)]));
    assert_eq!(
        (map.loads, map.stores, map.special, map.fresh),
        (2.0, 1.0, 1.0, 1.0)
    );
    // Thin sums `m` rows a trip, each row apart.
    let thin = trip(&Thin {
        m: 3,
        n: 8,
        k: 64,
        form: lower::form::F32_FORM,
        after: sites::After::default(),
    });
    assert_eq!((thin.loads, thin.float), (4.0, 3.0));
    assert!((thin.fresh - 1.0 / 16.0).abs() < 1e-9, "{thin:?}");
}

#[test]
fn limits_bar_kernels() {
    // A 64-lane device runs none of the 32-lane kernels.
    let wide = lower::Limits {
        lanes: 64,
        ..lower::Limits::assumed()
    };
    let (_, plan) = toy(&Device::guess(&settings()).within(wide));
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    let lanes = |n: &&str| n.starts_with("lanes") || n.starts_with("tiled") || *n == "lane_norm";
    assert!(!names.iter().any(lanes), "{names:?}");
    assert!(
        names.contains(&"thin") && names.contains(&"rms_norm"),
        "{names:?}"
    );
    // A small group memory keeps the tiled schedules that fit.
    let small = lower::Limits {
        group_bytes: 12 * 1024,
        ..lower::Limits::assumed()
    };
    let (g, _) = toy(&Device::guess(&settings()));
    let device = Device::guess(&settings()).within(small).favoring("tiled");
    let plan = lower(&g, 0, &SIZES, &device).unwrap();
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    assert!(names.contains(&"tiled_32x32_w16"), "{names:?}");
    assert!(
        !names.iter().any(|n| n.starts_with("tiled_64")),
        "{names:?}"
    );
}
