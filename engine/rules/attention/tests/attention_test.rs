//! Attention rules against the interpreter, and the rule plans pick.

use tessera_attention as _;
use tessera_kernels as _;

use tessera_graph::DimName::{Context, HeadDim, Heads, KvHeads, Rows, Segments, Seqs};
use tessera_graph::{DType, Dim, Graph, Size, TensorType};
use tessera_interp::{Store, Tensor, run, run_kept};
use tessera_lower::kept::{self, Layout};
use tessera_lower::{Device, Settings, View, lower, rules};
use tessera_run::testing::{host, runs};
use tessera_run::{Run, open};
use tessera_testkit::{Noise, fill, graphs, max_abs_diff};
use tessera_trace::{EntryKind, Mask, Model, Port, Role};

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
            // Each key block in turn. The runner names the platform.
            for (way, other) in [("flash", "flash_k32"), ("flash_k32", "flash")] {
                let plan = lower(&g, 0, &sizes, &device.clone().without(other)).unwrap();
                let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                let label = format!("{mask:?} {seqs}x{rows}x{keys} at {width}, {way}");
                // Where 32 keys do not fit, 16 keys stay.
                if names[1] != way && way == "flash_k32" {
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
fn ragged(heads: u64, kv_heads: u64, width: u64) -> Graph {
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
    let out = m.attend_rows(0.125, [&q, &k, &v], by, None);
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
    let reversed = |n: usize| {
        (0..n as u32)
            .rev()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<u8>>()
    };
    for (heads, kv_heads, width) in [(2u64, 2u64, 16u64), (4, 2, 64)] {
        let g = ragged(heads, kv_heads, width);
        let mut cases = Vec::new();
        for settings in runs() {
            for favored in ["flash", "attend"] {
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
                let r = match &mut runner {
                    Some(r) => {
                        r.replan(&plan).unwrap();
                        r
                    }
                    None => {
                        let table = plan.table;
                        let r = runner.insert(open(plan, settings).unwrap());
                        if let Some(table) = table {
                            r.held_mut().upload(table, &reversed(table.len)).unwrap();
                        }
                        r
                    }
                };
                let views: Vec<_> = inputs.iter().map(|t| t.view()).collect();
                let got = r.run(&views).unwrap().remove(0);
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
