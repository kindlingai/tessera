//! Plans for small graphs, with lower's own kernels.

use sensai_graph::DimName::{self, Context, Hidden, Rows, Seqs};
use sensai_graph::{Act, DType, Dim, Graph, TensorType};
use sensai_interp::{Tensor, run};
use sensai_lower::form::{BF16_FORM, F32_FORM};
use sensai_lower::kernels::{Map, MapStage, Widen};
use sensai_lower::sites;
use sensai_lower::{
    Arg, Buffer, Device, Dialect, Draft, Error, Fit, Kernel, KernelRef, Pins, Plan, Slot, Source,
    Step, T, Tree,
};
use sensai_lower::{assign, lower, peak, rules, schedule, select};
use sensai_run::testing::{host, runs};
use sensai_run::{Held, open};
use sensai_sim::Chip;
use sensai_testkit::{Noise, fill, max_abs_diff, scratch};
use sensai_trace::{EntryKind, Model, Port, Role, Vector, Weights, declare};

// Lowering needs one rule from outside lower.
fn nothing(_: &Tree, _: T) -> Option<Vec<Draft>> {
    None
}

sensai_lower::rule!("nothing", nothing);

const SIZES: [(DimName, u64); 2] = [(Seqs, 2), (Rows, 3)];

#[derive(Weights)]
struct Gains {
    #[shape(Hidden)]
    a: Vector,
    #[shape(Hidden)]
    b: Vector,
}

fn floats<const N: usize>(shape: [DimName; N]) -> Port {
    let ty = TensorType::new(shape, DType::F32);
    let role = Role::Logits;
    Port { role, ty }
}

fn ids<const N: usize>(shape: [DimName; N]) -> Port {
    let ty = TensorType::new(shape, DType::U32);
    let role = Role::Slots;
    Port { role, ty }
}

/// Maps over rows, where two values have two readers each.
fn ladder() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::fixed(Hidden, 16),
    ]);
    let w: Gains = declare(&m);
    let port = || floats([Seqs, Rows, Hidden]);
    let [x, y] = m.entry(EntryKind::Forward, [port(), port()]);
    let h1 = x.add(&y);
    let h2 = h1.act(Act::Silu).mul(&y.mul_vector(&w.a)).mul(&x);
    let out = h2.mul_vector(&w.b).act(Act::Silu).add(&h1).add(&h2);
    m.finish([(Role::Logits, &out)])
}

/// New rows into kept keys and values, summed.
fn appended() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 16),
        Dim::bounded(Context, 16),
        Dim::fixed(Hidden, 16),
    ]);
    let w: Gains = declare(&m);
    let ports = [
        floats([Rows, Hidden]),
        floats([Rows, Hidden]),
        ids([Rows]),
        ids([Rows]),
    ];
    let [k, v, slots, positions] = m.entry(EntryKind::Step, ports);
    let kept = TensorType::new([Seqs, Context, Hidden], DType::F32);
    let (keys, values) = (m.kept(kept.clone()), m.kept(kept));
    let by = [&slots, &positions];
    let out = k.append(&keys, by).add(&v.append(&values, by));
    m.finish([(Role::Logits, &out.mul_vector(&w.a))])
}

fn count(plan: &Plan, kernel: &str) -> usize {
    let named = |s: &&sensai_lower::Step| s.kernel.name() == kernel;
    plan.steps.iter().filter(named).count()
}

fn names(plan: &Plan) -> Vec<&'static str> {
    plan.steps.iter().map(|s| s.kernel.name()).collect()
}

fn gate() -> Map {
    Map::plain(vec![MapStage::Act(Act::Silu), MapStage::Mul])
}

#[test]
fn maps_match_cpu() {
    let g = ladder();
    let mut noise = Noise(5);
    let weights: Vec<Tensor> = (g.weights.iter())
        .map(|w| fill(&g, &w.ty, &SIZES, &mut noise).unwrap())
        .collect();
    let ports = &g.entries[0].inputs;
    let inputs: Vec<Tensor> = ports
        .iter()
        .map(|p| fill(&g, &p.ty, &SIZES, &mut noise).unwrap())
        .collect();
    let forms = [F32_FORM, BF16_FORM];
    for (settings, form) in runs().iter().flat_map(|s| forms.map(|f| (s, f))) {
        // The reference sees the weights in the plan's stored form.
        let kept: Vec<Tensor> = (weights.iter())
            .map(|t| Tensor::f32(t.shape.clone(), form.roundtrip(t.floats().unwrap())))
            .collect();
        let want = run(&g, &kept, 0, &inputs).unwrap();
        let device = Device::guess(settings).pinned(Pins::all(form));
        let plan = lower(&g, 0, &SIZES, &device).unwrap();
        assert!(names(&plan).contains(&"map"));
        let mut runner = open(plan, settings).unwrap();
        for (at, w) in kept.iter().enumerate() {
            runner.load(at as u32, w.view()).unwrap();
        }
        let feeds: Vec<_> = inputs.iter().map(Tensor::view).collect();
        let got = runner.run(&feeds).unwrap().remove(0);
        let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(off < 1e-5, "as {}: off by {off:e}", form.name());
    }
}

#[test]
fn buffers_are_shared() {
    let g = ladder();
    let plan = lower(&g, 0, &SIZES, &Device::guess(&host())).unwrap();
    assert_eq!(plan.order(), sensai_graph::Order::Fixed);
    for s in &plan.steps {
        let in_place = s.reads.iter().any(|r| r.buffer == s.write.buffer);
        assert_eq!(in_place, s.kernel.overwrites());
    }
    let values = plan.weights.len() + plan.inputs.len() + plan.steps.len();
    assert!(
        plan.buffers.len() < values,
        "dead values free their buffers"
    );
    let unsized_rows = lower(&g, 0, &[(Seqs, 2)], &Device::guess(&host()));
    assert!(matches!(unsized_rows, Err(Error::Size(_))));
}

#[test]
fn rule_names_differ() {
    // Rules sharing a name would share a cost curve.
    let names: Vec<&str> = rules().iter().map(|r| r.name).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(names, sorted, "rule names are unique and ordered");
}

#[test]
fn order_saves_memory() {
    let g = ladder();
    let picked = select(&g, 0, &SIZES, &Device::guess(&host())).unwrap();
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
    lower(&appended(), 0, &sizes, &Device::guess(&host()))
}

#[test]
fn kept_rows_fit() {
    let plan = step(1, 2, 5).unwrap();
    assert_eq!(plan.kept.len(), 2, "keys and values");
    // Sixteen rows of sixteen, for up to sixteen rows.
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
    assert_eq!(appends.count(), 2);
    let scratch: usize = plan.buffers.iter().sum::<usize>() - 2 * 1024;
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

/// One map launch writing `len` elements.
fn mapped(len: usize) -> Step {
    Step {
        kernel: KernelRef::new(gate()),
        rule: "map",
        reads: Vec::new(),
        write: Slot::floats(0, len),
        covers: Vec::new(),
        zone: 0,
        table: None,
    }
}

#[test]
fn near_samples_estimate() {
    let seconds = |d: &Device, len: usize| d.seconds(&*mapped(len).kernel, len);
    let mut device = Device::guess(&host());
    let guess = seconds(&device, 256);
    device.fit([(&mapped(64), 1e-3), (&mapped(1024), 4e-3)]);
    assert_eq!(device.samples(), 2);
    assert!((seconds(&device, 64) - 1e-3).abs() < 1e-9, "an exact match");
    assert!((seconds(&device, 1024) - 4e-3).abs() < 1e-9);
    // Between samples, each is scaled by work and they blend.
    let between = seconds(&device, 256);
    assert!(1e-3 < between && between < 4e-3 && between != guess);
    assert!(
        (seconds(&device, 2048) - 8e-3).abs() < 1e-4,
        "scaled by work"
    );
    assert!(device.known(&mapped(64)) && !device.known(&mapped(256)));
    // A launch timed again replaces its sample.
    device.fit([
        (&mapped(64), 2e-3),
        (&mapped(64), 9e-3),
        (&mapped(64), 3e-3),
    ]);
    assert_eq!(device.samples(), 2);
    assert!((seconds(&device, 64) - 3e-3).abs() < 1e-9, "the median");
    assert!(seconds(&device.clone().without("map"), 64).is_infinite());
}

#[test]
fn sources_differ() {
    let hash = |stages: Vec<MapStage>| Map::plain(stages).source().hash();
    assert_eq!(hash(vec![MapStage::Add]), hash(vec![MapStage::Add]));
    assert_ne!(hash(vec![MapStage::Add]), hash(vec![MapStage::Mul]));
    let widen = Widen {
        len: 4,
        form: BF16_FORM,
    };
    assert_ne!(hash(Vec::new()), widen.source().hash());
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
        ..Fit::default()
    }
}

#[test]
fn models_fill_gaps() {
    let (step, fit) = (mapped(64), chip());
    let seconds = |d: &Device| d.seconds(&*step.kernel, 64);
    let device = Device::guess(&host()).fitted(fit);
    let load = step.kernel.load(64).unwrap();
    let modelled = fit.pool("fitted").estimate(&load).seconds;
    assert_eq!(seconds(&device), modelled);
    assert!(
        seconds(&Device::guess(&host())) != modelled,
        "a bare guess differs"
    );
    let bound = device.bound(&*step.kernel, 64).unwrap();
    assert!(bound <= modelled && Device::guess(&host()).bound(&*step.kernel, 64).is_none());
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
    let g = ladder();
    let tree = Tree::new(&g, 0, &SIZES, &Pins::default(), sensai_lower::kept::DENSE).unwrap();
    let chains = |want: &[MapStage]| -> Vec<(T, Vec<T>)> {
        let all = (0..tree.len() as u32).map(T);
        let chain = |t: T| match tree.site(t)? {
            sites::Chain { stages, reads, .. } if stages == want => Some((t, reads)),
            _ => None,
        };
        all.filter_map(chain).collect()
    };
    // The activation has one reader, so it joins the multiply.
    let gates = chains(&[MapStage::Act(Act::Silu), MapStage::Mul, MapStage::Mul]);
    assert_eq!(gates.len(), 1);
    for (gate, reads) in &gates {
        let act = tree.node(*gate).args[0];
        assert!(!tree.root(act) && reads.len() == 3);
        assert_eq!(tree.covered(*gate, reads).len(), 3);
    }
    // The sum of x and y has two readers.
    let sums = chains(&[MapStage::Add]);
    assert_eq!(sums.len(), 1);
    for (sum, reads) in &sums {
        assert!(tree.covered(*sum, reads).len() == 1);
    }
    let (out, shape) = &tree.outputs[0];
    assert!(tree.root(*out) && *shape == [2, 3, 16]);
}

#[test]
fn pins_match_paths() {
    use sensai_graph::PathSeg::{Field, Index};
    let path = |s: &str| -> Vec<sensai_graph::PathSeg> {
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
    let g = ladder();
    let pins = Pins::all(BF16_FORM).pin("a", F32_FORM);
    let plan = lower(&g, 0, &SIZES, &Device::guess(&host()).pinned(pins)).unwrap();
    for l in &plan.weights {
        let kept = l.weight == 0;
        assert_eq!(l.slot.form == F32_FORM, kept, "weight {}", l.weight);
    }
    // Maps read floats, so the narrow weight widens once.
    assert_eq!(count(&plan, "widen"), 1);
    let both = lower(
        &g,
        0,
        &SIZES,
        &Device::guess(&host()).pinned(Pins::all(BF16_FORM)),
    );
    assert_eq!(count(&both.unwrap(), "widen"), 2);
    let wide = lower(&g, 0, &SIZES, &Device::guess(&host())).unwrap();
    assert_eq!(count(&wide, "widen"), 0);
}

#[test]
fn plans_are_cached() {
    let g = ladder();
    let mut cache = sensai_lower::Cache::default();
    let device = Device::guess(&host());
    let first = cache.lower(&g, 0, &SIZES, &device).unwrap();
    let again = cache.lower(&g, 0, &SIZES, &device).unwrap();
    assert!(first == again && cache.hits == 1 && cache.len() == 1);
    // Other sizes, or a device that changed, lower afresh.
    cache
        .lower(&g, 0, &[(Seqs, 1), (Rows, 3)], &device)
        .unwrap();
    let favored = device.clone().favoring("map");
    assert!(favored.generation() != device.generation());
    // Two devices changed alike are still two devices.
    let twice = device.clone().favoring("map");
    assert!(twice.generation() != favored.generation() && twice == favored);
    let narrow = device.clone().pinned(Pins::all(BF16_FORM));
    let widened = cache.lower(&g, 0, &SIZES, &narrow).unwrap();
    assert!(widened != first && cache.hits == 1 && cache.len() == 3);
    let mut fitted = device.clone();
    fitted.fit([(&widened.steps[0], 1e-3)]);
    assert!(fitted.generation() != device.generation());
    // Another graph at the same sizes is another plan.
    let kept = appended();
    let sizes = [(Seqs, 1), (Rows, 2), (Context, 5)];
    let stepped = cache.lower(&kept, 0, &sizes, &device).unwrap();
    let forward = cache.lower(&g, 0, &sizes, &device).unwrap();
    assert!(stepped != forward && cache.len() == 5);
}

#[test]
fn trips_are_counted() {
    let trip = |k: &dyn Kernel| k.load(64).unwrap().trip;
    // One thread per element reads a new line each time.
    let map = trip(&Map::plain(vec![MapStage::Add, MapStage::Act(Act::Silu)]));
    assert_eq!(
        (map.loads, map.stores, map.special, map.fresh),
        (2.0, 1.0, 1.0, 1.0)
    );
}

#[test]
fn uncovered_is_named() {
    use sensai_graph::DimName::Vocab;
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 4),
        Dim::fixed(Vocab, 64),
    ]);
    let [x] = m.entry(EntryKind::Forward, [floats([Seqs, Rows, Vocab])]);
    // No rule takes a lone center, so the act fails.
    let g = m.finish([(Role::Logits, &x.center().act(Act::Silu))]);
    let Err(Error::Unsupported(what)) = lower(&g, 0, &SIZES, &Device::guess(&host())) else {
        panic!("a lone center lowered");
    };
    let what = what.to_string();
    assert!(what.starts_with("Stage(Center)"), "{what}");
}

#[test]
fn unlinked_rules_fail() {
    let all = rules();
    assert!(sensai_lower::linked(&all).is_ok());
    let own: Vec<_> = all
        .into_iter()
        .filter(|r| r.module.starts_with("sensai_lower::"))
        .collect();
    let Err(Error::Unsupported(what)) = sensai_lower::linked(&own) else {
        panic!("only this crate's rules passed as linked");
    };
    assert!(what.to_string().contains("use rules as _;"), "{what}");
}
