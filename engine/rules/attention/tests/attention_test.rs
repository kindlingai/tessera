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
use tessera_trace::{Act, EntryKind, Mask, Model, PathSeg, Port, Role, Taps, Weights};

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
            for favored in ["attend", "flash", "listed_fused", "listed"] {
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
                        if favored == "listed_fused" {
                            assert!(names.contains(&"listed_attend"), "{names:?}");
                        } else if favored == "listed" {
                            assert!(names.contains(&"listed_mix"), "{names:?}");
                        }
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

/// Ragged Lightning Indexer over kept keys, causal or unmasked.
fn indexer_ragged(heads: u64, dim: u64, max_ctx: u64, causal: bool) -> Graph {
    let dim_of = |name, size| Dim { name, size };
    let mut dims = vec![
        dim_of(Seqs, Size::Bounded(4)),
        dim_of(Rows, Size::Bounded(64)),
        dim_of(Context, Size::Bounded(max_ctx)),
        dim_of(Heads, Size::Fixed(heads)),
        dim_of(HeadDim, Size::Fixed(dim)),
    ];
    dims.push(dim_of(Hidden, Size::Fixed(1)));
    let m = Model::new(dims);
    let port = |role, ty| Port { role, ty };
    let ids = TensorType::new([Rows], DType::U32);
    let [q, k, w, slots, positions] = m.entry(
        EntryKind::Step,
        [
            port(
                Role::Tokens,
                TensorType::new([Rows, Heads, HeadDim], DType::F32),
            ),
            port(Role::Tokens, TensorType::new([Rows, HeadDim], DType::F32)),
            port(
                Role::Tokens,
                TensorType::new([Rows, Hidden, Heads], DType::F32),
            ),
            port(Role::Slots, ids.clone()),
            port(Role::Positions, ids),
        ],
    );
    let kept = m.kept(TensorType::new([Seqs, Context, HeadDim], DType::F32));
    let k = k.append(&kept, [&slots, &positions]).gather(&slots);
    let hits = q.matmul_rows(&k).act(Act::Relu);
    let scores = hits.permute(&[0, 2, 1]).matmul_rows(&w);
    let out = match causal {
        true => scores.reshape([Rows, Context]).causal(&positions),
        false => scores,
    };
    m.finish([(Role::Logits, &out)])
}

/// Whole-sequence Lightning Indexer, with optional mask.
fn indexer_whole(heads: u64, dim: u64, mask: Option<Mask>) -> Graph {
    let dim_of = |name, size| Dim { name, size };
    let dims = vec![
        dim_of(Seqs, Size::Bounded(4)),
        dim_of(Context, Size::Bounded(64)),
        dim_of(Heads, Size::Fixed(heads)),
        dim_of(HeadDim, Size::Fixed(dim)),
        dim_of(Hidden, Size::Fixed(1)),
    ];
    let m = Model::new(dims);
    let port = |ty| Port {
        role: Role::Tokens,
        ty,
    };
    let [q, k, w] = m.entry(
        EntryKind::Forward,
        [
            port(TensorType::new([Seqs, Context, Heads, HeadDim], DType::F32)),
            port(TensorType::new([Seqs, Context, HeadDim], DType::F32)),
            port(TensorType::new([Seqs, Context, Hidden, Heads], DType::F32)),
        ],
    );
    let hits = q.matmul_rows(&k).act(Act::Relu);
    let scores = hits.permute(&[0, 1, 3, 2]).matmul_rows(&w);
    let out = match mask {
        Some(kind) => scores.reshape([Seqs, Context, Context]).mask(kind),
        None => scores,
    };
    m.finish([(Role::Logits, &out)])
}

/// Checks score parity, requiring exact -inf on masked entries.
fn check_scores(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: len mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if w == f32::NEG_INFINITY {
            assert_eq!(g, f32::NEG_INFINITY, "{label}[{i}]: expected -inf, got {g}");
        } else {
            let diff = (g - w).abs();
            assert!(
                diff < tol,
                "{label}[{i}]: got {g}, want {w} (diff {diff:e})"
            );
        }
    }
}

#[test]
fn indexer_ragged_matches() {
    let mut draws = Noise(53);
    let launches: [&[(u32, u32, u32)]; 3] = [
        &[(0, 0, 1)],
        &[(0, 1, 12), (1, 0, 5), (2, 0, 9)],
        &[(0, 13, 1), (1, 5, 1), (3, 0, 4), (2, 9, 1)],
    ];
    let layouts: [&'static dyn Layout; 3] =
        [kept::DENSE, kept::named("paged").unwrap(), &kept::PAGED16];
    // Includes odd width to cover plain fallback.
    let shapes = [
        (2u64, 32u64, true),
        (8, 128, true),
        (4, 6, true),
        (4, 32, false),
    ];
    for (heads, dim, causal) in shapes {
        let g = indexer_ragged(heads, dim, 128, causal);
        for settings in runs() {
            for &layout in &layouts {
                for force_plain in [false, true] {
                    if force_plain && dim == 6 {
                        continue;
                    }
                    let (mut store, mut runner): (Store, Option<Box<dyn Run>>) =
                        (Store::default(), None);
                    for runs in launches {
                        let (slots, positions) = rows(runs);
                        let (n, h, d) = (slots.len(), heads as usize, dim as usize);
                        let furthest = *positions.iter().max().unwrap() as u64 + 1;
                        let inputs = [
                            noise(vec![n, h, d], &mut draws),
                            noise(vec![n, d], &mut draws),
                            noise(vec![n, 1, h], &mut draws),
                            Tensor::u32(vec![n], slots),
                            Tensor::u32(vec![n], positions),
                        ];
                        let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
                        let sizes = [(Seqs, 4), (Rows, n as u64), (Context, furthest)];
                        let mut device = Device::guess(&settings);
                        device.kept = layout;
                        if force_plain {
                            device = device.without("indexer_warp");
                        }
                        let plan = lower(&g, 0, &sizes, &device).unwrap();
                        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                        let expect = if force_plain || dim == 6 {
                            "indexer"
                        } else {
                            "indexer_warp"
                        };
                        assert_eq!(names, ["scatter", expect], "{layout:?} {heads}x{dim}");
                        let got = step_plan(&mut runner, plan, &settings, &inputs);
                        let label = format!("{expect} {layout:?} {heads}x{dim} causal={causal}");
                        check_scores(
                            got.floats().unwrap(),
                            want[0].floats().unwrap(),
                            1e-4,
                            &label,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn indexer_narrow_kept() {
    let g = indexer_ragged(4, 128, 128, true);
    let layouts: [&'static dyn Layout; 3] = [kept::DENSE, kept::INTERLEAVED, &kept::PAGED16];
    let forms = [(BF16_FORM, 5e-2f32), (FP8_FORM, 0.5f32)];
    let mut draws = Noise(97);
    for (form, tol) in forms {
        let pins = Pins::default().activations(BF16_FORM).kept(form);
        for &layout in &layouts {
            for settings in runs() {
                let (mut store, mut runner): (Store, Option<Box<dyn Run>>) =
                    (Store::default(), None);
                for runs in [
                    &[(0u32, 0u32, 8u32), (1, 0, 5)][..],
                    &[(0, 8, 1), (1, 5, 1)][..],
                ] {
                    let (slots, positions) = rows(runs);
                    let n = slots.len();
                    let furthest = *positions.iter().max().unwrap() as u64 + 1;
                    let inputs = [
                        noise(vec![n, 4, 128], &mut draws),
                        noise(vec![n, 128], &mut draws),
                        noise(vec![n, 1, 4], &mut draws),
                        Tensor::u32(vec![n], slots),
                        Tensor::u32(vec![n], positions),
                    ];
                    let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
                    let sizes = [(Seqs, 4), (Rows, n as u64), (Context, furthest)];
                    let mut device = Device::guess(&settings).pinned(pins.clone());
                    device.kept = layout;
                    let plan = lower(&g, 0, &sizes, &device).unwrap();
                    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                    assert_eq!(names, ["scatter", "indexer_warp"], "{names:?}");
                    assert!(plan.kept.iter().all(|k| k.1.form == form));
                    let got = step_plan(&mut runner, plan, &settings, &inputs);
                    let label = format!("{} {layout:?}", form.name());
                    check_scores(
                        got.floats().unwrap(),
                        want[0].floats().unwrap(),
                        tol,
                        &label,
                    );
                }
            }
        }
    }
}

#[test]
fn indexer_whole_matches() {
    let mut draws = Noise(67);
    let cases = [
        (4u64, 32u64, Some(Mask::Causal)),
        (4, 128, Some(Mask::Back(3))),
        (2, 16, None),
    ];
    for (heads, dim, mask) in cases {
        let g = indexer_whole(heads, dim, mask);
        let (seqs, keys, h, d) = (2usize, 9usize, heads as usize, dim as usize);
        let inputs = [
            noise(vec![seqs, keys, h, d], &mut draws),
            noise(vec![seqs, keys, d], &mut draws),
            noise(vec![seqs, keys, 1, h], &mut draws),
        ];
        let want = run(&g, &[], 0, &inputs).unwrap();
        let views: Vec<_> = inputs.iter().map(|t| t.view()).collect();
        let sizes = [(Seqs, seqs as u64), (Context, keys as u64)];
        for settings in runs() {
            for force_plain in [false, true] {
                let mut device = Device::guess(&settings);
                if force_plain {
                    device = device.without("indexer_warp");
                }
                let plan = lower(&g, 0, &sizes, &device).unwrap();
                let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                let expect = if force_plain {
                    "indexer"
                } else {
                    "indexer_warp"
                };
                assert_eq!(names, [expect]);
                let mut runner = open(plan, &settings).unwrap();
                let got = runner.run(&views).unwrap().remove(0);
                check_scores(
                    got.floats().unwrap(),
                    want[0].floats().unwrap(),
                    1e-4,
                    expect,
                );
            }
        }
    }
}

#[test]
fn indexer_128k_scratch() {
    let g = indexer_ragged(32, 128, 131_072, true);
    let device = Device::guess(&host());
    let bare = device.clone().without("indexer_warp").without("indexer");
    for n in [1u64, 64] {
        let sizes = [(Seqs, 1), (Rows, n), (Context, 131_072)];
        let fused = lower(&g, 0, &sizes, &device).unwrap();
        let unfused = lower(&g, 0, &sizes, &bare).unwrap();
        let f_cost = device.cost(&fused);
        let u_cost = bare.cost(&unfused);
        assert_eq!(fused.steps.len(), 2);
        assert!(fused.scratch() * 100 < unfused.scratch());
        assert!(f_cost.bytes * 2 < u_cost.bytes);
    }
}

#[test]
fn sliding_ring_wraps() {
    let dim = |name, size| Dim { name, size };
    let dims = vec![
        dim(Seqs, Size::Bounded(2)),
        dim(Rows, Size::Bounded(4)),
        dim(Context, Size::Bounded(24)),
        dim(Heads, Size::Fixed(4)),
        dim(KvHeads, Size::Fixed(2)),
        dim(HeadDim, Size::Fixed(16)),
    ];
    let m = Model::new(dims);
    let floats = |h| TensorType::new([Rows, h, HeadDim], DType::F32);
    let ids = TensorType::new([Rows], DType::U32);
    let [q, k, v, slots, positions] = m.entry(
        EntryKind::Step,
        [
            Port {
                role: Role::Tokens,
                ty: floats(Heads),
            },
            Port {
                role: Role::Tokens,
                ty: floats(KvHeads),
            },
            Port {
                role: Role::Tokens,
                ty: floats(KvHeads),
            },
            Port {
                role: Role::Slots,
                ty: ids.clone(),
            },
            Port {
                role: Role::Positions,
                ty: ids,
            },
        ],
    );
    let kept = TensorType::new([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let (keys, values) = (m.kept(kept.clone()), m.kept(kept));
    let by = [&slots, &positions];
    let (k, v) = (k.append(&keys, by), v.append(&values, by));
    let out = m.attend_rows(Mask::Back(3), 0.25, [&q, &k, &v], by, None);
    let g = m.finish([(Role::Logits, &out)]);
    let mut draws = Noise(71);
    let mut batches = Vec::new();
    let mut store = Store::default();
    for step in 0..6u32 {
        let pos = step * 4;
        let (s, p) = rows(&[(0, pos, 2), (1, pos, 2)]);
        let (s2, p2) = rows(&[(0, pos + 2, 2), (1, pos + 2, 2)]);
        for (sl, po) in [(s, p), (s2, p2)] {
            let n = sl.len();
            let feeds = vec![
                noise(vec![n, 4, 16], &mut draws),
                noise(vec![n, 2, 16], &mut draws),
                noise(vec![n, 2, 16], &mut draws),
                Tensor::u32(vec![n], sl),
                Tensor::u32(vec![n], po),
            ];
            let want = run_kept(&g, &[], &mut store, 0, &feeds).unwrap().remove(0);
            batches.push((feeds, want));
        }
    }
    for settings in runs() {
        for layout in [kept::DENSE, &kept::PAGED16] {
            let mut device = Device::guess(&settings);
            device.kept = layout;
            let mut runner: Option<Box<dyn Run>> = None;
            for (feeds, want) in &batches {
                let sizes = [(Seqs, 2), (Rows, 4), (Context, 24)];
                let plan = lower(&g, 0, &sizes, &device).unwrap();
                assert_eq!(plan.kept_info[0].context, 8);
                assert!(plan.table.is_none());
                let got = step_plan(&mut runner, plan, &settings, feeds);
                let off = max_abs_diff(got.floats().unwrap(), want.floats().unwrap());
                assert!(
                    off < 1e-4,
                    "{:?} {}: off {off:e}",
                    settings.backend,
                    layout.name()
                );
            }
        }
    }
}

#[test]
fn attend_flash_knobs() {
    let g = attention(Mask::Bidirectional, 4, 64);
    let mut dev = Device::guess(&host()).favoring("attend");
    let base = lower(&g, 0, &[(Seqs, 1), (Rows, 2), (Context, 16)], &dev).unwrap();
    dev.knobs.parse("attend:threads=128").unwrap();
    let tuned = lower(&g, 0, &[(Seqs, 1), (Rows, 2), (Context, 16)], &dev).unwrap();
    let attend_id = |p: &tessera_lower::Plan| {
        p.steps
            .iter()
            .find(|s| s.kernel.name() == "attend")
            .unwrap()
            .kernel
            .identity()
    };
    assert_ne!(attend_id(&base), attend_id(&tuned));

    let mut fdev = Device::guess(&host()).favoring("flash");
    let full =
        tessera_lower::select(&g, 0, &[(Seqs, 1), (Rows, 64), (Context, 64)], &fdev).unwrap();
    fdev.knobs.parse("flash:bk=16").unwrap();
    let sub = tessera_lower::select(&g, 0, &[(Seqs, 1), (Rows, 64), (Context, 64)], &fdev).unwrap();
    assert!(sub.ways["flash"].len() < full.ways["flash"].len());
}

/// Each row's own taps, over whole sequences and ragged rows.
#[test]
fn dynamic_conv_matches() {
    let dims = || {
        vec![
            Dim::bounded(Seqs, 4),
            Dim::bounded(Rows, 16),
            Dim::bounded(Context, 16),
            Dim::fixed(Hidden, 6),
        ]
    };
    let port = |role, shape: tessera_graph::Shape, dtype| Port {
        role,
        ty: TensorType::new(shape, dtype),
    };
    let floats = |shape: Vec<tessera_graph::DimExpr>| port(Role::Tokens, shape.into(), DType::F32);
    let (m, mut draws) = (Model::new(dims()), Noise(11));
    let taps = vec![Seqs.into(), Context.into(), Hidden.into(), 3u64.into()];
    let [x, w] = m.entry(
        EntryKind::Forward,
        [
            floats(vec![Seqs.into(), Context.into(), Hidden.into()]),
            floats(taps),
        ],
    );
    let whole = m.finish([(Role::Logits, &x.conv_dynamic(&w, 2))]);
    let m = Model::new(dims());
    let ids = |role| port(role, [Rows].into(), DType::U32);
    let ports = [
        floats(vec![Rows.into(), Hidden.into()]),
        floats(vec![Rows.into(), Hidden.into(), 3u64.into()]),
        ids(Role::Slots),
        ids(Role::Positions),
    ];
    let [x, w, slots, positions] = m.entry(EntryKind::Step, ports);
    let state = TensorType::new(vec![Seqs.into(), 4u64.into(), Hidden.into()], DType::F32);
    let out = x.conv_rows_dynamic(&w, 2, &m.kept(state), [&slots, &positions]);
    let ragged = m.finish([(Role::Logits, &out)]);
    let launches: [&[(u32, u32, u32)]; 2] =
        [&[(0, 0, 7), (1, 0, 3)], &[(0, 7, 2), (1, 3, 1), (2, 0, 4)]];
    for settings in runs() {
        let device = Device::guess(&settings);
        let inputs = [
            noise(vec![2, 9, 6], &mut draws),
            noise(vec![2, 9, 6, 3], &mut draws),
        ];
        let want = run(&whole, &[], 0, &inputs).unwrap();
        let plan = lower(&whole, 0, &[(Seqs, 2), (Context, 9)], &device).unwrap();
        assert_eq!(plan.steps[0].kernel.name(), "conv_dynamic");
        let got = step_plan(&mut None, plan, &settings, &inputs);
        let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(off < 1e-5, "whole: off by {off:e}");
        // Slot 0 of the ragged steps is sequence 0 again.
        let (mut store, mut runner): (Store, Option<Box<dyn Run>>) = (Store::default(), None);
        let (xs, ws) = (inputs[0].floats().unwrap(), inputs[1].floats().unwrap());
        for launch in launches {
            let (slots, positions) = rows(launch);
            let n = slots.len();
            let (mut x, mut w) = (
                noise(vec![n, 6], &mut draws),
                noise(vec![n, 6, 3], &mut draws),
            );
            let (first, count) = (launch[0].1 as usize, launch[0].2 as usize);
            x = Tensor::f32(vec![n, 6], splice(&x, &xs[first * 6..][..count * 6]));
            w = Tensor::f32(vec![n, 6, 3], splice(&w, &ws[first * 18..][..count * 18]));
            let feeds = [
                x,
                w,
                Tensor::u32(vec![n], slots),
                Tensor::u32(vec![n], positions),
            ];
            let want_rows = run_kept(&ragged, &[], &mut store, 0, &feeds).unwrap();
            let want_rows = want_rows[0].floats().unwrap();
            let same = &want[0].floats().unwrap()[first * 6..][..count * 6];
            assert!(max_abs_diff(&want_rows[..count * 6], same) < 1e-6);
            let sizes = [(Seqs, 4), (Rows, n as u64)];
            let plan = lower(&ragged, 0, &sizes, &device).unwrap();
            assert_eq!(plan.steps[0].kernel.name(), "conv_rows");
            let got = step_plan(&mut runner, plan, &settings, &feeds);
            let off = max_abs_diff(got.floats().unwrap(), want_rows);
            assert!(off < 1e-5, "ragged {launch:?}: off by {off:e}");
        }
    }
}

/// `t`'s values with `head` over the first ones.
fn splice(t: &Tensor, head: &[f32]) -> Vec<f32> {
    let mut out = t.floats().unwrap().to_vec();
    out[..head.len()].copy_from_slice(head);
    out
}
