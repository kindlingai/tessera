//! The tiled and split sink rules against the interpreter.

use tessera_attention as _;
use tessera_kernels as _;

use tessera_graph::DimName::{self, Context, HeadDim, Heads, KvHeads, Rows, Segments, Seqs};
use tessera_graph::{DType, Dim, Graph, Size, TensorType};
use tessera_interp::{Store, Tensor, run_kept};
use tessera_lower::{Device, Dialect, Limits, Platform, Source, lower};
use tessera_run::testing::{host, runs};
use tessera_run::{Run, open};
use tessera_testkit::{Noise, cosine_f32, max_abs_diff};
use tessera_trace::{EntryKind, Mask, Model, Port, Role};

const RULE: &str = "flash_sunk";

const SPLIT: &str = "listed_split";

const METAL: &str = "flash_sunk_metal";

/// One launch: runs of (slot, first position, count).
type Runs = &'static [(u32, u32, u32)];

/// One ragged site: heads, key heads, width, sinks, bias.
#[derive(Clone, Copy, Debug)]
struct Case {
    mask: Mask,
    heads: usize,
    kv_heads: usize,
    width: usize,
    /// 0: none. 1: per head. 2: per row and head.
    sinks: usize,
    /// Nearest keys the bias covers, or 0.
    win: usize,
}

fn rel_win() -> DimName {
    DimName::named("RelWin")
}

/// Ragged rows over kept keys, with the case's extras.
fn graph(c: Case) -> Graph {
    let dim = |name, size| Dim { name, size };
    let bounded = [(Seqs, 4), (Rows, 128), (Context, 1024)];
    let mut dims: Vec<Dim> = bounded.map(|(n, max)| dim(n, Size::Bounded(max))).into();
    let fixed = [(Heads, c.heads), (KvHeads, c.kv_heads), (HeadDim, c.width)];
    dims.extend(fixed.map(|(n, size)| dim(n, Size::Fixed(size as u64))));
    dims.push(dim(rel_win(), Size::Fixed(c.win.max(1) as u64)));
    let m = Model::new(dims);
    let tokens = |shape: tessera_graph::Shape| Port {
        role: Role::Tokens,
        ty: TensorType::new(shape, DType::F32),
    };
    let ids = |role| Port {
        role,
        ty: TensorType::new([Rows], DType::U32),
    };
    let sink_shape = match c.sinks {
        2 => [Rows, Heads].into(),
        _ => [Heads].into(),
    };
    let [q, k, v, slots, positions, sinks, rel] = m.entry(
        EntryKind::Step,
        [
            tokens([Rows, Heads, HeadDim].into()),
            tokens([Rows, KvHeads, HeadDim].into()),
            tokens([Rows, KvHeads, HeadDim].into()),
            ids(Role::Slots),
            ids(Role::Positions),
            tokens(sink_shape),
            tokens([Rows, Heads, rel_win()].into()),
        ],
    );
    let kept = TensorType::new([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let (keys, values) = (m.kept(kept.clone()), m.kept(kept));
    let by = [&slots, &positions];
    let (k, v) = (k.append(&keys, by), v.append(&values, by));
    let extra = [(c.sinks > 0).then_some(&sinks), (c.win > 0).then_some(&rel)];
    let out = m.attend_rows_biased(c.mask, 0.125, [&q, &k, &v], by, None, extra);
    m.finish([(Role::Logits, &out)])
}

fn noise(shape: Vec<usize>, draws: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| draws.draw()).collect();
    Tensor::f32(shape, data)
}

/// Sink logits, the last one `-inf`: no sink there.
fn sink_logits(shape: Vec<usize>, draws: &mut Noise) -> Tensor {
    let mut data: Vec<f32> = (0..shape.iter().product()).map(|_| draws.draw()).collect();
    *data.last_mut().unwrap() = f32::NEG_INFINITY;
    Tensor::f32(shape, data)
}

/// Feeds for `runs` of (slot, first position, count).
fn feeds(c: Case, runs: &[(u32, u32, u32)], draws: &mut Noise) -> Vec<Tensor> {
    let slots: Vec<u32> = runs.iter().flat_map(|r| vec![r.0; r.2 as usize]).collect();
    let positions: Vec<u32> = runs.iter().flat_map(|r| r.1..r.1 + r.2).collect();
    let n = slots.len();
    let sink_shape = match c.sinks {
        2 => vec![n, c.heads],
        _ => vec![c.heads],
    };
    vec![
        noise(vec![n, c.heads, c.width], draws),
        noise(vec![n, c.kv_heads, c.width], draws),
        noise(vec![n, c.kv_heads, c.width], draws),
        Tensor::u32(vec![n], slots),
        Tensor::u32(vec![n], positions),
        sink_logits(sink_shape, draws),
        noise(vec![n, c.heads, c.win.max(1)], draws),
    ]
}

/// The sizes of a launch of `runs`.
fn sizes(runs: &[(u32, u32, u32)]) -> [(DimName, u64); 4] {
    let rows: u32 = runs.iter().map(|r| r.2).sum();
    let furthest = runs.iter().map(|r| r.1 + r.2).max().unwrap();
    [
        (Seqs, 4),
        (Rows, rows as u64),
        (Context, furthest as u64),
        (Segments, runs.len() as u64),
    ]
}

/// A device of `backend` with the assumed limits.
fn platform(backend: &str, arch: &str) -> Platform {
    Platform {
        backend: backend.into(),
        arch: arch.into(),
        name: arch.into(),
        ..Platform::assumed(Limits::assumed())
    }
}

fn kernels(plan: &tessera_lower::Plan) -> Vec<&'static str> {
    plan.steps.iter().map(|s| s.kernel.name()).collect()
}

/// A case from mask, heads, key heads, width, sinks, bias.
const fn case(mask: Mask, [heads, kv_heads, width, sinks, win]: [usize; 5]) -> Case {
    Case {
        mask,
        heads,
        kv_heads,
        width,
        sinks,
        win,
    }
}

const CASES: [Case; 6] = [
    case(Mask::Causal, [4, 2, 64, 1, 3]),
    case(Mask::Back(20), [4, 2, 64, 2, 5]),
    case(Mask::Back(20), [6, 2, 128, 0, 4]),
    case(Mask::Causal, [2, 2, 16, 2, 0]),
    case(Mask::Back(37), [4, 4, 32, 1, 0]),
    case(Mask::Causal, [3, 1, 128, 0, 0]),
];

/// A prefill over two slots, then rows wrapping the ring.
const LAUNCHES: [Runs; 5] = [
    &[(0, 0, 40), (1, 0, 24)],
    &[(0, 40, 33), (1, 24, 17), (2, 0, 16)],
    &[(2, 16, 50), (0, 73, 20)],
    &[(0, 93, 100), (1, 41, 28)],
    &[(0, 193, 61), (2, 66, 19), (1, 69, 30)],
];

/// Runs `plan` on the open runner, or opens one.
fn advance(
    runner: &mut Option<Box<dyn Run>>,
    plan: tessera_lower::Plan,
    opens: impl FnOnce(tessera_lower::Plan) -> Box<dyn Run>,
    inputs: &[Tensor],
) -> Tensor {
    let r = match runner.as_mut() {
        Some(r) => {
            r.replan(&plan).unwrap();
            r
        }
        None => runner.insert(opens(plan)),
    };
    let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
    r.run(&views).unwrap().remove(0)
}

/// Runs `launches` under `rule`. It must take those `takes` names.
fn parity(c: Case, launches: &[Runs], rule: &str, takes: fn(Runs) -> bool, draws: &mut Noise) {
    let settings = host();
    let g = graph(c);
    let (mut store, mut runner): (Store, Option<Box<dyn Run>>) = (Store::default(), None);
    for runs in launches {
        let inputs = feeds(c, runs, draws);
        let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
        let device = Device::guess(&settings).favoring(rule);
        let plan = lower(&g, 0, &sizes(runs), &device).unwrap();
        let took = kernels(&plan).contains(&rule);
        assert_eq!(took, takes(runs), "{c:?}: {:?}", kernels(&plan));
        let ring = plan.kept_info[0].context;
        let got = advance(&mut runner, plan, |p| open(p, &settings).unwrap(), &inputs);
        let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(off < 1e-4, "{c:?} ring {ring} on {runs:?}: off {off:e}");
    }
}

#[test]
fn sunk_matches_interp() {
    let mut draws = Noise(57);
    for c in CASES {
        parity(c, &LAUNCHES, RULE, |_| true, &mut draws);
    }
}

const SPLIT_CASES: [Case; 5] = [
    case(Mask::Causal, [4, 2, 64, 1, 3]),
    case(Mask::Back(300), [4, 2, 64, 2, 5]),
    case(Mask::Back(300), [6, 2, 128, 0, 4]),
    case(Mask::Causal, [2, 2, 16, 2, 0]),
    case(Mask::Back(270), [4, 4, 32, 1, 0]),
];

/// Prefills, then steps of few rows past the ring's wrap.
const STEPS: [Runs; 10] = [
    &[(0, 0, 128)],
    &[(0, 128, 128)],
    &[(0, 256, 100), (1, 0, 20)],
    &[(0, 356, 3), (1, 20, 2)],
    &[(0, 359, 1)],
    &[(0, 360, 128)],
    &[(0, 488, 128)],
    &[(0, 616, 60), (2, 0, 30)],
    &[(0, 676, 5), (1, 22, 7), (2, 30, 1)],
    &[(1, 29, 1)],
];

/// Whether a launch is a step over enough keys.
fn steps(runs: Runs) -> bool {
    let rows: u32 = runs.iter().map(|r| r.2).sum();
    rows < 16 && sizes(runs)[2].1 >= 256
}

#[test]
fn split_matches_interp() {
    let mut draws = Noise(91);
    for c in SPLIT_CASES {
        parity(c, &STEPS, SPLIT, steps, &mut draws);
    }
}

/// Metal steps split each list. Other backends do not.
#[test]
fn metal_takes_split() {
    let apple = Device::guess(&host()).on(platform("metal", "apple9"));
    let step: Runs = &[(0, 900, 8)];
    for c in SPLIT_CASES.into_iter().filter(|c| c.width >= 64) {
        let g = graph(c);
        let names = kernels(&lower(&g, 0, &sizes(step), &apple).unwrap());
        assert!(names.contains(&SPLIT), "{c:?}: {names:?}");
        assert!(names.contains(&"listed_merge"), "{c:?}: {names:?}");
        assert!(!names.contains(&"listed_attend"), "{c:?}: {names:?}");
        // Few keys stay whole.
        let near = kernels(&lower(&g, 0, &sizes(&[(0, 100, 8)]), &apple).unwrap());
        assert!(!near.contains(&SPLIT), "{c:?}: {near:?}");
        for (backend, arch) in [("cuda", "sm_121"), ("rocm", "gfx11_5")] {
            let other = Device::guess(&host()).on(platform(backend, arch));
            let names = kernels(&lower(&g, 0, &sizes(step), &other).unwrap());
            assert!(!names.contains(&SPLIT), "{c:?} on {backend}: {names:?}");
        }
    }
}

/// A window keeps its keys on a ring.
#[test]
fn sunk_reads_rings() {
    let g = graph(CASES[1]);
    let device = Device::guess(&host()).favoring(RULE);
    let plan = lower(&g, 0, &sizes(LAUNCHES[0]), &device).unwrap();
    let step = plan.steps.iter().find(|s| s.kernel.name() == RULE).unwrap();
    let entry = step.kernel.identity().0;
    assert_eq!(&*entry, "flash_sunk_64_ring");
    assert_eq!(plan.kept_info[0].context, 192);
    // Metal gets the tiles and the ring's scalar.
    let Source::Made(body) = step.kernel.source() else {
        panic!("the tiles are generated");
    };
    let text = body.text().render(Dialect::Metal);
    assert!(text.contains("flash_sunk_64_ring") && text.contains("kept_ring"));
    let whole = lower(&graph(CASES[0]), 0, &sizes(LAUNCHES[0]), &device).unwrap();
    let step = whole
        .steps
        .iter()
        .find(|s| s.kernel.name() == RULE)
        .unwrap();
    assert_eq!(&*step.kernel.identity().0, "flash_sunk_64_dense");
}

/// Metal prefill takes the tiles. Few rows do not.
#[test]
fn metal_takes_sunk() {
    let apple = Device::guess(&host()).on(m4_pro());
    for c in CASES {
        let g = graph(c);
        let plan = lower(&g, 0, &sizes(LAUNCHES[0]), &apple).unwrap();
        let names = kernels(&plan);
        // Width 128 goes to the bf16 tiles.
        let tiles = if c.width == 128 { METAL } else { RULE };
        assert!(names.contains(&tiles), "{c:?}: {names:?}");
        assert!(!names.contains(&"listed_attend"), "{c:?}: {names:?}");
        // A decode step of a few rows stays split-K.
        let few = lower(&g, 0, &sizes(&[(0, 64, 4)]), &apple).unwrap();
        assert!(!kernels(&few).contains(&RULE), "{c:?}");
        for (backend, arch) in [("cuda", "sm_121"), ("rocm", "gfx11_5")] {
            let other = Device::guess(&host()).on(platform(backend, arch));
            let plan = lower(&g, 0, &sizes(LAUNCHES[0]), &other).unwrap();
            assert!(!kernels(&plan).contains(&RULE), "{c:?} on {backend}");
        }
    }
}

/// The units and bandwidth an M4 Pro reports.
fn m4_pro() -> Platform {
    let mut plat = Platform {
        name: "Apple M4 Pro".into(),
        units: 20,
        bandwidth: 273e9,
        ..platform("metal", "apple9")
    };
    plat.limits.msl_bfloat = true;
    plat
}

/// Wide Metal prefill takes the bf16 tiles, unfavored.
#[test]
fn sunk_metal_is_picked() {
    let apple = Device::guess(&host()).on(m4_pro());
    // A window keeps a ring. Causal rows keep all.
    for (c, kept) in [(CASES[2], "ring"), (CASES[5], "dense")] {
        let g = graph(c);
        let plan = lower(&g, 0, &sizes(LAUNCHES[0]), &apple).unwrap();
        assert!(kernels(&plan).contains(&METAL), "{kept}: {c:?}");
        let step = plan
            .steps
            .iter()
            .find(|s| s.kernel.name() == METAL)
            .unwrap();
        assert_eq!(step.rule, RULE);
        let Source::Fixed { entry, .. } = step.kernel.source() else {
            panic!("a fixed kernel");
        };
        assert_eq!(entry, METAL);
        assert_eq!(step.kernel.needs().group_bytes, 25_600);
        let few = lower(&g, 0, &sizes(&[(0, 64, 4)]), &apple).unwrap();
        assert!(!kernels(&few).contains(&METAL), "{c:?}");
        let mut old_msl = m4_pro();
        old_msl.limits.msl_bfloat = false;
        let fallback = lower(
            &g,
            0,
            &sizes(LAUNCHES[0]),
            &Device::guess(&host()).on(old_msl),
        )
        .unwrap();
        assert!(
            kernels(&fallback).contains(&RULE) && !kernels(&fallback).contains(&METAL),
            "{kept}: should fall back to {RULE} without MSL 3.1 bfloat"
        );
        for (backend, arch) in [("cuda", "sm_121"), ("rocm", "gfx11_5")] {
            let other = Device::guess(&host()).on(platform(backend, arch));
            let plan = lower(&g, 0, &sizes(LAUNCHES[0]), &other).unwrap();
            assert!(!kernels(&plan).contains(&METAL), "{c:?} on {backend}");
        }
    }
    // Narrower heads keep the f32 tiles.
    let narrow = lower(&graph(CASES[0]), 0, &sizes(LAUNCHES[0]), &apple).unwrap();
    assert!(kernels(&narrow).contains(&RULE) && !kernels(&narrow).contains(&METAL));
}

/// Zero queries and keys over `n` rows of one value.
fn flat(c: Case, n: usize, val: f32, sink: f32, shift: f32) -> Vec<Tensor> {
    let win = c.win.max(1);
    vec![
        Tensor::f32(vec![n, c.heads, c.width], vec![0.0; n * c.heads * c.width]),
        Tensor::f32(
            vec![n, c.kv_heads, c.width],
            vec![0.0; n * c.kv_heads * c.width],
        ),
        Tensor::f32(
            vec![n, c.kv_heads, c.width],
            vec![val; n * c.kv_heads * c.width],
        ),
        Tensor::u32(vec![n], vec![0; n]),
        Tensor::u32(vec![n], (0..n as u32).collect()),
        Tensor::f32(vec![c.heads], vec![sink; c.heads]),
        Tensor::f32(vec![n, c.heads, win], vec![shift; n * c.heads * win]),
    ]
}

/// Gives `check` each host and Metal tile output.
fn tiled(c: Case, n: usize, inputs: &[Tensor], label: &str, check: impl Fn(&[f32], &str)) {
    let g = graph(c);
    let run = [(0u32, 0u32, n as u32)];
    for settings in runs() {
        let dev = Device::guess(&settings).favoring(RULE);
        if dev.platform.backend != "host" && dev.platform.backend != "metal" {
            continue;
        }
        let bf16 = dev.platform.backend == "metal" && dev.platform.limits.msl_bfloat;
        let plan = lower(&g, 0, &sizes(&run), &dev).unwrap();
        let names = kernels(&plan);
        let expect_kernel = if c.width == 128 && bf16 { METAL } else { RULE };
        assert!(names.contains(&expect_kernel), "picked {names:?}");
        let mut runner = open(plan, &settings).unwrap();
        let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
        let got = runner.run(&views).unwrap().remove(0);
        let on = format!("{label} on {} ({expect_kernel})", runner.device());
        check(got.floats().unwrap(), &on);
    }
}

/// Host and Metal tiles give finite rows near `expected`.
fn near(c: Case, n: usize, inputs: &[Tensor], expected: &[f32], label: &str) {
    tiled(c, n, inputs, label, |got, on| {
        assert!(got.iter().all(|v| v.is_finite()), "{on} non-finite");
        let rel = got
            .iter()
            .zip(expected)
            .map(|(a, b)| (*a as f64 - *b as f64).abs() / (b.abs() as f64).max(1.0))
            .fold(0.0f64, f64::max);
        assert!(rel < 2e-2, "{on}: rel {rel:e}");
    });
}

/// Large values, sinks and shifted logits stay finite and accurate.
#[test]
fn sink_stays_finite() {
    let near_max = f32::from_bits(0x7f7f_fffe);
    let cases = [
        (16usize, 0.0f32, 90.0f32, 1e37f32),
        (64, 0.0, 90.0, 1e37),
        (64, 0.0, 64.0, 1e37),
        (64, 536_870_912.0, 64.0, 1e37),
        (64, 0.0, 0.5, 1e37),
        (64, 0.0, 90.0, near_max),
        (64, 0.0, 0.5, near_max),
        (64, 0.0, 90.0, f32::MAX),
        (64, 0.0, 0.5, f32::MAX),
    ];
    for (n, shift, gap, val) in cases {
        for width in [64usize, 128] {
            let win = if shift != 0.0 { n } else { 0 };
            let c = case(Mask::Causal, [2, 1, width, 1, win]);
            let inputs = flat(c, n, val, shift + gap, shift);
            let mut store = Store::default();
            let want = run_kept(&graph(c), &[], &mut store, 0, &inputs).unwrap();
            let expected = want[0].floats().unwrap();
            let label = format!("n={n} w={width} shift={shift} gap={gap}");
            assert!(
                expected[0] > 1e-3 && expected.iter().all(|v| v.is_finite()),
                "{label} expected[0]={:?}",
                expected[0]
            );
            near(c, n, &inputs, expected, &label);
        }
    }
}

/// A weak or absent sink leaves the largest values finite.
#[test]
fn weak_sink_stays_finite() {
    let n = 64usize;
    let near_max = f32::from_bits(0x7f7f_fffe);
    // (sink mode, sink logit): weak, absent, then no sink.
    let sinks = [(1, -30.0f32), (1, -6.0), (1, f32::NEG_INFINITY), (0, 0.0)];
    for (mode, sink) in sinks {
        for val in [f32::MAX, near_max, -f32::MAX] {
            for width in [64usize, 128] {
                let c = case(Mask::Causal, [2, 1, width, mode, 0]);
                let inputs = flat(c, n, val, sink, 0.0);
                // The interpreter overflows here: weigh by hand.
                let share = |t: f64| match mode {
                    1 => t / (t + (sink as f64).exp()),
                    _ => 1.0,
                };
                let row = |r: usize| (val as f64 * share((r + 1) as f64)) as f32;
                let expected: Vec<f32> =
                    (0..n).flat_map(|r| vec![row(r); c.heads * width]).collect();
                let label = format!("w={width} mode={mode} sink={sink} val={val:e}");
                near(c, n, &inputs, &expected, &label);
            }
        }
    }
}

/// Masked non-finite values stay out. Seen ones pass through.
#[test]
fn odd_values_masked() {
    let n = 64usize;
    // The first block, then a block after finite ones.
    for pos in [1usize, 33] {
        for width in [64usize, 128] {
            let c = case(Mask::Causal, [2, 1, width, 1, 0]);
            let mut inputs = flat(c, n, 1.0, -30.0, 0.0);
            let mut v = vec![1.0f32; n * width];
            let odd = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
            v[pos * width..pos * width + 3].copy_from_slice(&odd);
            inputs[2] = Tensor::f32(vec![n, 1, width], v);
            let mut store = Store::default();
            let want = run_kept(&graph(c), &[], &mut store, 0, &inputs).unwrap();
            let want = want[0].floats().unwrap();
            let label = format!("pos={pos} w={width}");
            tiled(c, n, &inputs, &label, |got, on| {
                for (i, (g, e)) in got.iter().zip(want).enumerate() {
                    let (row, ch) = (i / (c.heads * width), i % width);
                    if row >= pos && ch < 3 {
                        let same = g.is_nan() && odd[ch].is_nan() || *g == odd[ch];
                        assert!(same, "{on}: row {row} ch {ch} got {g}");
                    } else {
                        let off = (g - e).abs();
                        assert!(off < 2e-2, "{on}: row {row} ch {ch} got {g}, not {e}");
                    }
                }
            });
        }
    }
}

/// A seen NaN key turns every later row NaN.
#[test]
fn odd_keys_propagate() {
    let n = 64usize;
    for width in [64usize, 128] {
        let c = case(Mask::Causal, [2, 1, width, 1, 0]);
        let mut inputs = flat(c, n, 1.0, -30.0, 0.0);
        inputs[0] = Tensor::f32(vec![n, 2, width], vec![1.0; n * 2 * width]);
        let mut k = vec![0.0f32; n * width];
        k[width] = f32::NAN;
        inputs[1] = Tensor::f32(vec![n, 1, width], k);
        tiled(c, n, &inputs, &format!("w={width}"), |got, on| {
            for (i, g) in got.iter().enumerate() {
                let row = i / (c.heads * width);
                let fits = if row == 0 {
                    (g - 1.0).abs() < 2e-2
                } else {
                    g.is_nan()
                };
                assert!(fits, "{on}: row {row} got {g}");
            }
        });
    }
}

/// Prefills 255 zero-score keys, then checks one decode row.
fn decoded(
    c: Case,
    rule: &'static str,
    fill: impl Fn(&mut [Tensor], usize, usize),
    check: impl Fn(&[f32], &[f32], &str),
) {
    let g = graph(c);
    for settings in runs() {
        let dev = Device::guess(&settings).favoring(rule);
        if dev.platform.backend != "host" && dev.platform.backend != "metal" {
            continue;
        }
        let (mut store, mut runner) = (Store::default(), None);
        for (start, n) in [(0usize, 128usize), (128, 127), (255, 1)] {
            let run = [(0u32, start as u32, n as u32)];
            let mut inputs = flat(c, n, 1.0, f32::NEG_INFINITY, 0.0);
            inputs[4] = Tensor::u32(vec![n], (start as u32..(start + n) as u32).collect());
            fill(&mut inputs, start, n);
            let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
            let plan = lower(&g, 0, &sizes(&run), &dev).unwrap();
            let took = kernels(&plan).contains(&rule);
            let got = advance(&mut runner, plan, |p| open(p, &settings).unwrap(), &inputs);
            if n == 1 {
                let on = format!("{rule} on {}", dev.platform.backend);
                assert!(took, "{on}: not taken");
                check(got.floats().unwrap(), want[0].floats().unwrap(), &on);
            }
        }
    }
}

/// Split parts of huge opposite values merge to zero.
#[test]
fn split_cancels_large() {
    let c = case(Mask::Causal, [2, 1, 64, 1, 0]);
    for big in [f32::from_bits(0x7d80_0000), f32::MAX] {
        // Eight keys of one sign, then eight of the other.
        let sign = |t: usize| if (t / 8).is_multiple_of(2) { big } else { -big };
        let fill = |inputs: &mut [Tensor], start: usize, n: usize| {
            let vs = (start..start + n).flat_map(|t| vec![sign(t); 64]);
            inputs[2] = Tensor::f32(vec![n, 1, 64], vs.collect());
        };
        decoded(c, SPLIT, fill, |got, want, on| {
            assert!(got.iter().all(|v| v.abs() < 1e-4), "{on}: {:?}", got[0]);
            // The interpreter loses the largest values to rounding.
            assert!(want[0] == 0.0 || big == f32::MAX, "{on}: interp {want:?}");
        });
    }
}

/// Drops keys whose relative bias is -inf.
#[test]
fn barred_key_drops() {
    let c = case(Mask::Causal, [2, 1, 64, 1, 256]);
    for rule in [SPLIT, "listed_attend"] {
        let fill = |inputs: &mut [Tensor], _: usize, n: usize| {
            let mut bias = vec![0.0f32; n * 2 * 256];
            if n == 1 {
                bias[255] = f32::NEG_INFINITY;
                bias[511] = f32::NEG_INFINITY;
            }
            inputs[6] = Tensor::f32(vec![n, 2, 256], bias);
        };
        decoded(c, rule, fill, |got, want, on| {
            let off = max_abs_diff(got, want);
            let one = got.iter().all(|v| (v - 1.0).abs() < 1e-4);
            assert!(one && off < 1e-4, "{on}: got {:?}, off {off:e}", got[0]);
        });
    }
}

/// The largest query component times zero keys stays zero.
#[test]
fn huge_query_stays_finite() {
    let n = 64usize;
    for width in [64usize, 128] {
        let c = case(Mask::Causal, [2, 1, width, 1, 0]);
        let mut inputs = flat(c, n, 1.0, -30.0, 0.0);
        let mut q = vec![0.0f32; n * 2 * width];
        q[5 * 2 * width] = f32::MAX;
        inputs[0] = Tensor::f32(vec![n, 2, width], q);
        let expected = vec![1.0f32; n * 2 * width];
        near(c, n, &inputs, &expected, &format!("w={width}"));
    }
}

/// Whether each value matches, NaN for NaN.
fn alike(got: &[f32], want: &[f32], tol: f32) -> bool {
    let fits = |(g, w): (&f32, &f32)| g.is_nan() && w.is_nan() || (g - w).abs() < tol;
    got.iter().zip(want).all(fits)
}

/// Checks the last row under the tiled and listed plans.
fn across(c: Case, fill: impl Fn(&mut [Tensor], usize, usize), check: impl Fn(&[f32], &str)) {
    let n = 64usize;
    let mut inputs = flat(c, n, 1.0, f32::NEG_INFINITY, 0.0);
    fill(&mut inputs, 0, n);
    let want = run_kept(&graph(c), &[], &mut Store::default(), 0, &inputs).unwrap();
    let at = (n - 1) * c.heads * c.width;
    let want = &want[0].floats().unwrap()[at..];
    tiled(c, n, &inputs, "last row", |got, on| {
        assert!(alike(&got[at..], want, 2e-2), "{on}: got {:?}", &got[at..]);
        check(&got[at..], on);
    });
    for rule in [SPLIT, "listed_attend"] {
        decoded(c, rule, &fill, |got, want, on| {
            assert!(alike(got, want, 1e-4), "{on}: got {got:?}");
            check(got, on);
        });
    }
}

/// A NaN sink logit makes its whole head NaN.
#[test]
fn odd_sink_propagates() {
    for width in [64usize, 128] {
        let c = case(Mask::Causal, [2, 1, width, 1, 0]);
        let fill = |inputs: &mut [Tensor], _: usize, _: usize| {
            inputs[5] = Tensor::f32(vec![2], vec![f32::NAN, f32::NEG_INFINITY]);
        };
        across(c, fill, |got, on| {
            let (odd, plain) = got.split_at(width);
            let ones = plain.iter().all(|v| (v - 1.0).abs() < 2e-2);
            assert!(odd.iter().all(|v| v.is_nan()) && ones, "{on}: {got:?}");
        });
    }
}

/// A key barred by bias still shows its NaN value.
#[test]
fn barred_value_shows() {
    for (width, odd) in [(64usize, true), (64, false), (128, true)] {
        let c = case(Mask::Causal, [2, 1, width, 1, 256]);
        let fill = |inputs: &mut [Tensor], start: usize, n: usize| {
            let mut bias = vec![0.0f32; n * 2 * 256];
            // The first head of each row bars key 0.
            for r in 0..n {
                bias[r * 512 + start + r] = f32::NEG_INFINITY;
            }
            inputs[6] = Tensor::f32(vec![n, 2, 256], bias);
            let mut v = vec![1.0f32; n * width];
            if start == 0 && odd {
                v[0] = f32::NAN;
            }
            inputs[2] = Tensor::f32(vec![n, 1, width], v);
        };
        across(c, fill, |got, on| {
            let from = if odd { 1 } else { 0 };
            let ones = got[from..width].iter().all(|v| (v - 1.0).abs() < 2e-2);
            let first = got[0].is_nan() == odd;
            assert!(first && ones, "{on}: {:?}", &got[..2]);
        });
    }
}

/// Odd sinks and barred rows end alike on every plan.
#[test]
fn endpoints_align() {
    let (ni, pi, na) = (f32::NEG_INFINITY, f32::INFINITY, f32::NAN);
    // Per head: NaN, 0 or 1. Head 7 follows interp.
    let ends = [na, na, na, 0.0, na, na, 1.0];
    for width in [64usize, 128] {
        let c = case(Mask::Causal, [8, 1, width, 1, 256]);
        let fill = |inputs: &mut [Tensor], start: usize, n: usize| {
            inputs[5] = Tensor::f32(vec![8], vec![pi, ni, na, 0.0, ni, 0.0, ni, 0.0]);
            let mut bias = vec![0.0f32; n * 8 * 256];
            for r in 0..n {
                // Heads 1 to 3 bar each key.
                bias[(r * 8 + 1) * 256..(r * 8 + 4) * 256].fill(ni);
                bias[(r * 8 + 4) * 256 + start + r] = pi;
                bias[(r * 8 + 5) * 256 + start + r] = na;
            }
            inputs[6] = Tensor::f32(vec![n, 8, 256], bias);
        };
        across(c, fill, |got, on| {
            for (h, end) in ends.iter().enumerate() {
                let head = &got[h * width..(h + 1) * width];
                assert!(alike(head, &vec![*end; width], 2e-2), "{on}: head {h}");
            }
        });
    }
}

/// Runs on a Metal GPU alone: bf16 rounds each operand.
#[test]
fn sunk_metal_matches() {
    let Some(settings) = runs().into_iter().find(|s| s.gpu().is_some()) else {
        return;
    };
    let plat = Device::guess(&settings).platform;
    if plat.backend != "metal" || !plat.limits.msl_bfloat {
        return;
    }
    let mut draws = Noise(73);
    let wide = [
        case(Mask::Back(20), [6, 2, 128, 0, 4]),
        case(Mask::Causal, [3, 1, 128, 0, 0]),
        case(Mask::Causal, [4, 2, 128, 1, 3]),
        case(Mask::Back(37), [4, 4, 128, 2, 5]),
    ];
    for c in wide {
        let g = graph(c);
        let (mut store, mut runner): (Store, Option<Box<dyn Run>>) = (Store::default(), None);
        for launch in LAUNCHES {
            let inputs = feeds(c, launch, &mut draws);
            let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
            let device = Device::guess(&settings).favoring(RULE);
            let plan = lower(&g, 0, &sizes(launch), &device).unwrap();
            assert!(kernels(&plan).contains(&METAL), "{c:?}: {launch:?}");
            let r = match runner.as_mut() {
                Some(r) => {
                    r.replan(&plan).unwrap();
                    r
                }
                None => runner.insert(open(plan, &settings).unwrap()),
            };
            let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
            let got = r.run(&views).unwrap().remove(0);
            let cos = cosine_f32(got.floats().unwrap(), want[0].floats().unwrap());
            assert!(cos > 0.9999, "{c:?} on {launch:?}: cos {cos}");
        }
    }
}
