//! Attention rules against the interpreter, and the rule plans pick.

use tessera_attention as _;
use tessera_kernels as _;

use tessera_graph::DimName::{Context, HeadDim, Heads, Hidden, KvHeads, Rows, Segments, Seqs};
use tessera_graph::{DType, Dim, Graph, Size, TensorType};
use tessera_interp::{Store, Tensor, run, run_kept};
use tessera_lower::form::{BF16_FORM, FP8_FORM};
use tessera_lower::kept::{self, Layout};
use tessera_lower::{Cache, Device, Pick, Pins, Settings, View, lower, rules};
use tessera_run::testing::{host, runs};
use tessera_run::{Run, open};
use tessera_testkit::{Noise, fill, graphs, max_abs_diff};
use tessera_trace::{EntryKind, Mask, Model, PathSeg, Port, Role, Taps, Weights};

/// Attention under `mask` over `heads` query heads of `width`.
fn attention(mask: Mask, heads: u64, width: u64) -> Graph {
    let dim = |name, size| Dim { name, size };
    let bounded = [(Seqs, 4), (Rows, 128), (Context, 256)];
    let mut dims: Vec<Dim> = bounded.map(|(n, max)| dim(n, Size::Bounded(max))).into();
    dims.extend([
        dim(Heads, Size::Fixed(heads)),
        dim(HeadDim, Size::Fixed(width)),
    ]);
    let m = Model::new(dims);
    let port = |rows| Port {
        role: Role::Tokens,
        ty: TensorType::new([Seqs, rows, Heads, HeadDim], DType::F32),
    };
    let [q, k, v] = m.entry(
        EntryKind::Forward,
        [port(Rows), port(Context), port(Context)],
    );
    let out = m.attend(mask, 0.125, [&q, &k, &v], None);
    m.finish([(Role::Logits, &out)])
}

fn noise(shape: Vec<usize>, noise: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
    Tensor::f32(shape, data)
}

#[test]
fn flash_matches_cpu() {
    let mut draws = Noise(41);
    // Rows off, on and past the block, with older keys.
    let cases = [
        (1, 1, 1),
        (1, 1, 38),
        (2, 7, 7),
        (1, 33, 33),
        (2, 70, 93),
        (1, 128, 128),
    ];
    let masks = [
        Mask::Causal,
        Mask::Bidirectional,
        Mask::Within(5),
        Mask::Within(40),
        Mask::Back(5),
    ];
    let shapes = [(2u64, 16u64), (4, 64), (2, 128)];
    for (mask, (heads, width)) in masks.into_iter().flat_map(|m| shapes.map(|s| (m, s))) {
        let g = attention(mask, heads, width);
        for (settings, (seqs, rows, keys)) in runs().iter().flat_map(|s| cases.map(|c| (s, c))) {
            let (h, w) = (heads as usize, width as usize);
            let q = noise(vec![seqs, rows, h, w], &mut draws);
            let k = noise(vec![seqs, keys, h, w], &mut draws);
            let v = noise(vec![seqs, keys, h, w], &mut draws);
            let inputs = [q, k, v];
            let want = run(&g, &[], 0, &inputs).unwrap();
            let sizes = [
                (Seqs, seqs as u64),
                (Rows, rows as u64),
                (Context, keys as u64),
            ];
            let views: Vec<_> = inputs.iter().map(|t| t.view()).collect();
            let mut device = Device::guess(settings).favoring("flash");
            // Each way in turn. The runner names the platform.
            let ways = ["flash", "flash_k32", "flash_held", "flash_k32_held"];
            for way in ways {
                let others = ways.into_iter().filter(|o| *o != way);
                let barred = others.fold(device.clone(), |d, o| d.without(o));
                let plan = lower(&g, 0, &sizes, &barred).unwrap();
                let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                let label = format!("{mask:?} {seqs}x{rows}x{keys} at {width}, {way}");
                // A way that does not fit leaves the first.
                if names[1] != way && way != ways[0] {
                    continue;
                }
                assert_eq!(names, ["map", way], "{label}");
                let mut runner = open(plan, settings).unwrap();
                let got = runner.run(&views).unwrap().remove(0);
                let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                assert!(off < 1e-3, "{label}: off by {off:e}");
                if settings.gpu().is_some() {
                    device = device.on(runner.platform());
                }
            }
        }
    }
}

/// Ragged rows over kept keys: `heads` query heads over `kv_heads`.
fn ragged(mask: Mask, heads: u64, kv_heads: u64, width: u64) -> Graph {
    let dim = |name, size| Dim { name, size };
    let bounded = [(Seqs, 4), (Rows, 128), (Context, 256)];
    let mut dims: Vec<Dim> = bounded.map(|(n, max)| dim(n, Size::Bounded(max))).into();
    dims.extend([
        dim(Heads, Size::Fixed(heads)),
        dim(KvHeads, Size::Fixed(kv_heads)),
        dim(HeadDim, Size::Fixed(width)),
    ]);
    let m = Model::new(dims);
    let floats = |heads| TensorType::new([Rows, heads, HeadDim], DType::F32);
    let port = |role, ty| Port { role, ty };
    let ids = TensorType::new([Rows], DType::U32);
    let [q, k, v, slots, positions] = m.entry(
        EntryKind::Step,
        [
            port(Role::Tokens, floats(Heads)),
            port(Role::Tokens, floats(KvHeads)),
            port(Role::Tokens, floats(KvHeads)),
            port(Role::Slots, ids.clone()),
            port(Role::Positions, ids),
        ],
    );
    let kept = TensorType::new([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let (keys, values) = (m.kept(kept.clone()), m.kept(kept));
    let by = [&slots, &positions];
    let (k, v) = (k.append(&keys, by), v.append(&values, by));
    let out = m.attend_rows(mask, 0.125, [&q, &k, &v], by, None);
    m.finish([(Role::Logits, &out)])
}

/// Rows for `runs` of (slot, first position, count).
fn rows(runs: &[(u32, u32, u32)]) -> (Vec<u32>, Vec<u32>) {
    let slots = runs.iter().flat_map(|(s, _, n)| (0..*n).map(move |_| *s));
    let positions = runs
        .iter()
        .flat_map(|(_, p, n)| (0..*n).map(move |i| p + i));
    (slots.collect(), positions.collect())
}

/// Runs one plan on the runner.
fn step_plan(
    runner: &mut Option<Box<dyn Run>>,
    plan: tessera_lower::Plan,
    settings: &Settings,
    inputs: &[Tensor],
) -> Tensor {
    let r = match runner {
        Some(r) => {
            r.replan(&plan).unwrap();
            r
        }
        None => {
            let table = plan.table;
            let r = runner.insert(open(plan, settings).unwrap());
            if let Some(table) = table {
                let pages: Vec<u8> = (0..table.len as u32)
                    .rev()
                    .flat_map(u32::to_le_bytes)
                    .collect();
                r.held_mut().upload(table, &pages).unwrap();
            }
            r
        }
    };
    let views: Vec<_> = inputs.iter().map(|t| t.view()).collect();
    r.run(&views).unwrap().remove(0)
}

#[test]
fn ragged_rows_match_cpu() {
    let mut draws = Noise(43);
    // A block over three slots, then rows past kept ones.
    let launches: [&[(u32, u32, u32)]; 2] = [
        &[(0, 0, 20), (1, 0, 5), (2, 0, 15)],
        &[(0, 20, 1), (1, 5, 1), (3, 0, 3), (2, 15, 1)],
    ];
    // Reversed pages: no slot's rows lie where dense ones would.
    let layouts: [&'static dyn Layout; 2] = [kept::DENSE, &kept::PAGED16];
    let shapes = [
        (Mask::Causal, 2u64, 2u64, 16u64),
        (Mask::Causal, 4, 2, 64),
        (Mask::Back(4), 4, 2, 64),
    ];
    for (mask, heads, kv_heads, width) in shapes {
        let g = ragged(mask, heads, kv_heads, width);
        let mut cases = Vec::new();
        for settings in runs() {
            // The flash rows kernel is causal only.
            let favors = match mask {
                Mask::Causal => &["flash", "attend"][..],
                _ => &["attend"],
            };
            for &favored in favors {
                for layout in layouts {
                    cases.push((settings.clone(), favored, layout));
                }
            }
        }
        for &(ref settings, favored, layout) in &cases {
            let (mut store, mut runner): (Store, Option<Box<dyn Run>>) = (Store::default(), None);
            for runs in launches {
                let (slots, positions) = rows(runs);
                let (n, h, kv, w) = (
                    slots.len(),
                    heads as usize,
                    kv_heads as usize,
                    width as usize,
                );
                let furthest = *positions.iter().max().unwrap() as u64 + 1;
                let inputs = [
                    noise(vec![n, h, w], &mut draws),
                    noise(vec![n, kv, w], &mut draws),
                    noise(vec![n, kv, w], &mut draws),
                    Tensor::u32(vec![n], slots),
                    Tensor::u32(vec![n], positions),
                ];
                let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
                let sizes = [
                    (Seqs, 4),
                    (Rows, n as u64),
                    (Context, furthest),
                    (Segments, runs.len() as u64),
                ];
                let mut device = Device::guess(settings).favoring(favored);
                device.kept = layout;
                let plan = lower(&g, 0, &sizes, &device).unwrap();
                let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                let kernel = format!("{favored}_rows");
                assert!(names.contains(&kernel.as_str()), "{names:?}");
                let got = step_plan(&mut runner, plan, settings, &inputs);
                let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                assert!(
                    off < 1e-3,
                    "{favored} {layout:?} on {runs:?} at {width}: off by {off:e}"
                );
            }
        }
    }
}

#[test]
fn attention_streams() {
    let bounded = [
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::bounded(Context, 16),
    ];
    let g = graphs::attention(&bounded, [2, 2, 16]);
    let sizes = [(Seqs, 2), (Rows, 3), (Context, 5)];
    let plan = lower(&g, 0, &sizes, &Device::guess(&host())).unwrap();
    let streams = |s: &&tessera_lower::Step| matches!(s.kernel.name(), "attend" | "flash");
    assert_eq!(plan.steps.iter().filter(streams).count(), 1);
}

/// Largest gap between the interpreter and rule `name`.
fn gap(g: &Graph, inputs: &[Tensor], name: &str, settings: &Settings) -> f32 {
    let sizes = [(Seqs, 2), (Rows, 3), (Context, 5)];
    let want = run(g, &[], 0, inputs).unwrap();
    let device = Device::guess(settings).favoring(name);
    let mut runner = open(lower(g, 0, &sizes, &device).unwrap(), settings).unwrap();
    let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
    let got = runner.run(&feeds).unwrap().remove(0);
    max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap())
}

#[test]
fn gpu_rules_match() {
    let dims = [
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::bounded(Context, 16),
    ];
    let g = graphs::attention(&dims, [2, 2, 16]);
    let mut noise = Noise(7);
    let inputs: Vec<Tensor> = g.entries[0]
        .inputs
        .iter()
        .map(|p| fill(&g, &p.ty, &[(Seqs, 2), (Rows, 3), (Context, 5)], &mut noise).unwrap())
        .collect();
    // The CPU rules need a CPU zone.
    let gpu =
        |m: &str| m.starts_with("tessera_attention::") && !m.starts_with("tessera_attention::cpu");
    for rule in rules().into_iter().filter(|r| gpu(r.module)) {
        for settings in runs() {
            let off = gap(&g, &inputs, rule.name, &settings);
            assert!(off < 1e-4, "{} is off by {off:e}", rule.name);
        }
    }
}

/// Filtered values, then six value heads over two key heads.
#[test]
fn delta_matches_interp() {
    let dim = |name, size| Dim { name, size };
    let m = Model::new(vec![
        dim(Seqs, Size::Bounded(2)),
        dim(Context, Size::Bounded(64)),
        dim(KvHeads, Size::Fixed(2)),
        dim(Heads, Size::Fixed(6)),
        dim(HeadDim, Size::Fixed(16)),
        dim(Hidden, Size::Fixed(8)),
    ]);
    let port = |shape: tessera_graph::Shape| Port {
        role: Role::Tokens,
        ty: TensorType::new(shape, DType::F32),
    };
    let (keys, rows) = ([Seqs, Context, KvHeads, HeadDim], [Seqs, Context, Heads]);
    let channels = vec![Seqs.into(), Context.into(), 48u64.into()];
    let ports = [
        keys.into(),
        keys.into(),
        channels.into(),
        rows.into(),
        rows.into(),
    ]
    .map(port);
    let [q, k, x, decay, rate] = m.entry(EntryKind::Forward, ports);
    let mut path = vec![PathSeg::Field("taps".into())];
    let taps = Taps::declare(&m, &mut path, &[48u64.into(), 4u64.into()]);
    let v = x.conv(&taps).reshape([Seqs, Context, Heads, Hidden]);
    let out = m.delta(0.25, [&q, &k, &v], [&decay, &rate]);
    let g = m.finish([(Role::Logits, &out)]);
    let mut draws = Noise(7);
    let taps = noise(vec![48, 4], &mut draws);
    let native = Settings {
        backend: Pick::Cpu,
        ..host()
    };
    for settings in runs().into_iter().chain([native]) {
        let device = Device::guess(&settings);
        let inputs = delta_inputs(50, &mut draws);
        let want = run(&g, std::slice::from_ref(&taps), 0, &inputs).unwrap();
        let plan = lower(&g, 0, &[(Seqs, 2), (Context, 50)], &device).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        let ours = |n: &&str| n.ends_with("delta") || n.ends_with("conv");
        assert!(names.iter().all(ours), "{names:?}");
        let mut runner = open(plan, &settings).unwrap();
        runner.load(0, taps.view()).unwrap();
        let views: Vec<_> = inputs.iter().map(|t| t.view()).collect();
        let got = runner.run(&views).unwrap().remove(0);
        let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(off < 1e-4, "{:?}: off by {off:e}", settings.backend);
        // One 64-row plan. A cache launch binds the bucket.
        let mut plans = Cache::default();
        for len in [50, 33] {
            let inputs = delta_inputs(len, &mut draws);
            let want = run(&g, std::slice::from_ref(&taps), 0, &inputs).unwrap();
            let sizes = [(Seqs, 2), (Context, len as u64)];
            let (plan, bound) = plans.upto(&g, 0, &sizes, &[Context], &device).unwrap();
            let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
            assert!(names.iter().all(ours), "{names:?}");
            let slot = plan.outputs[0].slot;
            runner.replan(plan).unwrap();
            runner.bind_all(&bound);
            // Bound below the bucket, the kernels read the count.
            let rows = if len == 33 { len } else { bound[0].1 as usize };
            runner.bind(Context, rows as u32);
            let fed: Vec<Tensor> = (inputs.iter().zip(&runner.plan().inputs))
                .map(|(t, s)| flat(&padded(t, rows), s.len))
                .collect();
            let views: Vec<_> = fed.iter().map(|t| t.view()).collect();
            runner.feed(&views).unwrap();
            runner.submit(0..runner.plan().steps.len()).unwrap();
            let out = runner.read(slot).unwrap().floats().unwrap()[..2 * rows * 48].to_vec();
            let got = padded(&Tensor::f32(vec![2, rows, 48], out), len);
            let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
            assert!(
                off < 1e-4,
                "{len} rows on {:?}: off by {off:e}",
                settings.backend
            );
        }
        assert_eq!(plans.len(), 1);
    }
}

/// `t`, `[seqs, len, ..]`, cut or zero filled to `rows`.
fn padded(t: &Tensor, rows: usize) -> Tensor {
    let [seqs, len] = [t.shape[0], t.shape[1]];
    let row = t.len() / (seqs * len);
    let x = t.floats().unwrap();
    let mut out = vec![0.0; seqs * rows * row];
    let kept = len.min(rows) * row;
    for s in 0..seqs {
        out[s * rows * row..][..kept].copy_from_slice(&x[s * len * row..][..kept]);
    }
    let mut shape = t.shape.clone();
    shape[1] = rows;
    Tensor::f32(shape, out)
}

/// `t`'s values, zero filled to `len`.
fn flat(t: &Tensor, len: usize) -> Tensor {
    let mut x = t.floats().unwrap().to_vec();
    x.resize(len, 0.0);
    Tensor::f32(vec![len], x)
}

/// Two sequences of `len` rows: unit keys, decays and rates.
fn delta_inputs(len: usize, draws: &mut Noise) -> [Tensor; 5] {
    let seqs = 2;
    // Unit keys and rates below one keep the state bounded.
    let unit = |draws: &mut Noise| {
        let t = noise(vec![seqs, len, 2, 16], draws);
        let mut x = t.floats().unwrap().to_vec();
        for row in x.chunks_mut(16) {
            let norm = row.iter().map(|a| a * a).sum::<f32>().sqrt();
            row.iter_mut().for_each(|a| *a /= norm);
        }
        Tensor::f32(t.shape.clone(), x)
    };
    let gate = |draws: &mut Noise, f: fn(f32) -> f32| {
        let x = (0..seqs * len * 6).map(|_| f(draws.draw())).collect();
        Tensor::f32(vec![seqs, len, 6], x)
    };
    [
        unit(draws),
        unit(draws),
        noise(vec![seqs, len, 48], draws),
        gate(draws, |x| -x.abs()),
        gate(draws, |x| (x + 1.0) / 2.0),
    ]
}

/// Narrow KV caches shrink kept buffers and attend natively.
#[test]
fn narrow_kept_matches() {
    let g = ragged(Mask::Causal, 4, 2, 32);
    let layouts: [&'static dyn Layout; 4] = [
        kept::DENSE,
        kept::INTERLEAVED,
        kept::named("paged").unwrap(),
        &kept::PAGED16,
    ];
    let forms = [(BF16_FORM, 2e-2f32), (FP8_FORM, 0.25f32)];
    let mut draws = Noise(91);
    for (form, tol) in forms {
        let pins = Pins::default().activations(BF16_FORM).kept(form);
        for &layout in &layouts {
            for favored in ["attend", "flash"] {
                for settings in runs() {
                    let (mut store, mut runner): (Store, Option<Box<dyn Run>>) =
                        (Store::default(), None);
                    for runs in [&[(0u32, 0u32, 6u32)][..], &[(0, 6, 1)][..]] {
                        let (slots, positions) = rows(runs);
                        let n = slots.len();
                        let furthest = *positions.iter().max().unwrap() as u64 + 1;
                        let inputs = [
                            noise(vec![n, 4, 32], &mut draws),
                            noise(vec![n, 2, 32], &mut draws),
                            noise(vec![n, 2, 32], &mut draws),
                            Tensor::u32(vec![n], slots),
                            Tensor::u32(vec![n], positions),
                        ];
                        let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
                        let sizes = [
                            (Seqs, 4),
                            (Rows, n as u64),
                            (Context, furthest),
                            (Segments, runs.len() as u64),
                        ];
                        let mut device = Device::guess(&settings)
                            .favoring(favored)
                            .pinned(pins.clone());
                        device.kept = layout;
                        let plan = lower(&g, 0, &sizes, &device).unwrap();
                        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                        assert!(!names.contains(&"widen"), "{names:?}");
                        assert!(!names.contains(&"narrow"), "{names:?}");
                        assert!(plan.kept.iter().all(|k| k.1.form == form));
                        let got = step_plan(&mut runner, plan, &settings, &inputs);
                        let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                        assert!(
                            off < tol,
                            "{} {favored} {layout:?} on {:?}: off by {off:e}",
                            form.name(),
                            settings.backend
                        );
                    }
                }
            }
        }
    }
}
