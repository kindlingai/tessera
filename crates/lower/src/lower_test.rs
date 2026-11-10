use crate::kernels::{Map, MapStage, Thin};
use crate::{
    Arg, Device, Dialect, Error, Fit, Held, Kernel, KernelRef, Plan, Site, Slot, Source, Step, T,
    Tree,
};
use crate::{assign, lower, peak, rules, schedule, select};
use graph::DimName::{self, Context, Rows, Seqs};
use graph::{Act, Composite, Graph};
use models::qwen3::{Config, program, stepper};
use sim::Chip;

const SIZES: [(DimName, u64); 2] = [(Seqs, 2), (Context, 5)];

fn toy(device: &Device) -> (Graph, Plan) {
    let g = program(&Config::toy(4, 16));
    let plan = lower(&g, 0, &SIZES, device).unwrap();
    (g, plan)
}

fn count(plan: &Plan, kernel: &str) -> usize {
    let named = |s: &&crate::Step| s.kernel.name() == kernel;
    plan.steps.iter().filter(named).count()
}

fn gate() -> Map {
    Map(vec![MapStage::Act(Act::Silu), MapStage::Mul])
}

#[test]
fn gate_fuses() {
    let (g, plan) = toy(&Device::guess());
    let fused = KernelRef::new(gate());
    let steps: Vec<_> = plan.steps.iter().filter(|s| s.kernel == fused).collect();
    assert_eq!(steps.len(), 2, "one per layer");
    for s in steps {
        assert_eq!((s.covers.len(), s.reads.len(), s.rule), (2, 2, "map"));
        let holds = |r: &&graph::Region| s.covers.iter().all(|c| r.start <= *c && *c < r.end);
        let region = g.regions.iter().find(holds).unwrap();
        assert_eq!(region.kind, Composite::Mlp);
    }
}

#[test]
fn buffers_are_shared() {
    let (g, plan) = toy(&Device::guess());
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
    let unsized_context = lower(&g, 0, &[(Seqs, 2)], &Device::guess());
    assert!(matches!(unsized_context, Err(Error::Size(_))));
}

#[test]
fn cost_picks_the_matmul() {
    // Ten rows fit the thin kernel, so cost alone decides.
    let (_, tiled) = toy(&Device::guess().without("thin"));
    let (_, thin) = toy(&Device::guess().without("matmul"));
    assert_eq!((count(&tiled, "thin"), count(&thin, "matmul")), (0, 0));
    assert_eq!(count(&tiled, "matmul"), count(&thin, "thin"));
    let mut measured = Device::guess();
    let cheap = thin.steps.iter().map(|s| (s, 1e-6));
    measured.fit(cheap.chain(tiled.steps.iter().map(|s| (s, 1e-3))));
    let (_, fitted) = toy(&measured);
    assert_eq!(count(&fitted, "matmul"), 0, "the faster kernel wins");
    assert!(measured.cost(&fitted).seconds < measured.cost(&tiled).seconds);
}

#[test]
fn cost_picks_attention() {
    // Dense attention is unproven, so a guess streams.
    let (_, stream) = toy(&Device::guess());
    assert_eq!((count(&stream, "attend"), count(&stream, "scores")), (2, 0));
    let (_, dense) = toy(&Device::guess().favoring("dense"));
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
fn rules_register_themselves() {
    let names: Vec<&str> = rules().iter().map(|r| r.name).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(names, sorted, "rule names are unique and ordered");
    for expected in ["attend", "dense", "map", "matmul", "thin"] {
        assert!(names.contains(&expected), "{expected} is missing");
    }
    let g = program(&Config::toy(4, 16));
    let picked = select(&g, 0, &SIZES, &Device::guess()).unwrap();
    let contested: Vec<&str> = picked.contested.iter().copied().collect();
    assert_eq!(contested, ["attend", "dense", "matmul", "thin"]);
}

#[test]
fn order_saves_memory() {
    let g = program(&Config::toy(4, 16));
    let picked = select(&g, 0, &SIZES, &Device::guess()).unwrap();
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
    let Source::Made(text) = gate().source() else {
        panic!("maps are generated");
    };
    let (metal, cuda) = (text.render(Dialect::Metal), text.render(Dialect::Cuda));
    // Buffers, then scalars, in one order for both.
    assert!(metal.contains("device const float *a0 [[buffer(1)]]"));
    assert!(metal.contains("constant u32 &len [[buffer(3)]]"));
    let order = "const float *x,\n    const float *a0,\n    float *out,\n    u32 len\n";
    assert!(cuda.contains(order), "{cuda}");
    assert!(cuda.contains(&format!("__global__ void {}(", text.entry)));
    for body in [&metal, &cuda] {
        assert!(body.contains("v = v * a0[gid];"));
    }
}

/// A toy step plan: `rows` new, `all` kept afterwards.
fn step(seqs: u64, rows: u64, all: u64) -> Result<Plan, Error> {
    let sizes = [(Seqs, seqs), (Rows, rows), (Context, all)];
    lower(&stepper(&Config::toy(4, 16)), 0, &sizes, &Device::guess())
}

#[test]
fn kept_rows_fit() {
    let plan = step(1, 2, 5).unwrap();
    assert_eq!(plan.kept.len(), 2 * 2, "keys and values per layer");
    // Two key heads of eight, for up to sixteen rows.
    for (_, slot) in &plan.kept {
        assert_eq!((slot.len, plan.buffers[slot.buffer as usize]), (80, 256));
    }
    let appends = plan.steps.iter().filter(|s| s.kernel.name() == "append");
    for s in appends.clone() {
        assert!(plan.kept.iter().any(|(_, slot)| *slot == s.write));
        let [launch] = &s.launches()[..] else {
            panic!("one copy");
        };
        // Three rows are kept, so the copy starts later.
        assert_eq!(launch.args[1..], [Arg::Buffer(s.write, 48), Arg::U32(32)]);
    }
    assert_eq!((appends.count(), count(&plan, "argmax")), (4, 1));
    let scratch: usize = plan.buffers.iter().sum::<usize>() - 4 * 256;
    assert!(plan.scratch() < scratch, "kept rows are not scratch");
    assert!(matches!(step(2, 2, 5), Err(Error::Unsupported(_))));
}

#[test]
fn replans_hold_buffers() {
    let (first, next) = (step(1, 3, 3).unwrap(), step(1, 1, 4).unwrap());
    let mut held: Held<usize> = Held::default();
    let mut made = 0;
    let mut alloc = |_: usize| {
        made += 1;
        Ok(made)
    };
    held.bind(&first, &mut alloc).unwrap();
    let named = |p: &Plan| -> Vec<_> { p.weights.iter().chain(&p.kept).copied().collect() };
    let before: Vec<usize> = named(&first).iter().map(|(_, s)| *held.get(*s)).collect();
    held.bind(&next, &mut alloc).unwrap();
    let after: Vec<usize> = named(&next).iter().map(|(_, s)| *held.get(*s)).collect();
    assert_eq!(named(&first).len(), named(&next).len());
    let owner = |p: &Plan| -> Vec<u32> { named(p).iter().map(|(index, _)| *index).collect() };
    assert_eq!(owner(&first), owner(&next));
    assert_eq!(before, after, "weights and kept rows stay put");
    assert!(made < first.buffers.len() + next.buffers.len());
    // Binding the same plan again allocates nothing.
    let so_far = made;
    held.bind(&next, |_| unreachable!("every buffer is held"))
        .unwrap();
    assert_eq!(made, so_far);
}

/// One thin launch of `m` rows, 64 outputs wide.
fn thin(m: usize) -> Step {
    let (n, k) = (64, 32);
    Step {
        kernel: KernelRef::new(Thin { m, n, k }),
        rule: "thin",
        reads: Vec::new(),
        write: Slot {
            buffer: 0,
            len: m * n,
        },
        covers: Vec::new(),
    }
}

#[test]
fn near_samples_estimate() {
    let seconds = |d: &Device, m: usize| d.seconds(&*thin(m).kernel, m * 64);
    let mut device = Device::guess();
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
    }
}

#[test]
fn models_fill_gaps() {
    let (step, fit) = (thin(1), chip());
    let seconds = |d: &Device| d.seconds(&*step.kernel, 64);
    let device = Device::guess().fitted(fit);
    let load = step.kernel.load(64).unwrap();
    let modelled = fit.pool("fitted").estimate(&load).seconds;
    assert_eq!(seconds(&device), modelled);
    assert!(
        seconds(&Device::guess()) != modelled,
        "a bare guess differs"
    );
    let bound = device.bound(&*step.kernel, 64).unwrap();
    assert!(bound <= modelled && Device::guess().bound(&*step.kernel, 64).is_none());
    // A sample of the kernel outranks the model.
    let mut timed = device.clone();
    timed.fit([(&step, 5e-3)]);
    assert_eq!(seconds(&timed), 5e-3);
    // The stored fit brings the model back.
    let file = std::env::temp_dir().join("tessera-lower-fit-test.json");
    device.save(&file).unwrap();
    let loaded = Device::load(&file).unwrap();
    std::fs::remove_file(&file).unwrap();
    assert!(loaded == device && seconds(&loaded) == modelled);
}

#[test]
fn trees_cut_shared() {
    let g = program(&Config::toy(4, 16));
    let tree = Tree::new(&g, 0, &SIZES).unwrap();
    let chains = |want: &[MapStage]| -> Vec<(T, Vec<T>)> {
        let all = (0..tree.len() as u32).map(T);
        let chain = |t: T| match tree.site(t)? {
            Site::Map { stages, reads, .. } if stages == want => Some((t, reads)),
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
    // No rule competes with maps, so favoring them changes nothing.
    let (_, plain) = toy(&Device::guess());
    let (_, favored) = toy(&Device::guess().favoring("map"));
    assert!(
        plain == favored,
        "a favored rule skewed its readers' choice"
    );
}
