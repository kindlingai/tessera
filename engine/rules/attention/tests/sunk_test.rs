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
        let r = match runner.as_mut() {
            Some(r) => {
                r.replan(&plan).unwrap();
                r
            }
            None => runner.insert(open(plan, &settings).unwrap()),
        };
        let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
        let got = r.run(&views).unwrap().remove(0);
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
            let g = graph(c);
            let run = [(0u32, 0u32, n as u32)];
            let bias = vec![shift; n * c.heads * win.max(1)];
            let inputs = vec![
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
                Tensor::f32(vec![c.heads], vec![shift + gap; c.heads]),
                Tensor::f32(vec![n, c.heads, win.max(1)], bias),
            ];
            let mut store = Store::default();
            let want = run_kept(&g, &[], &mut store, 0, &inputs).unwrap();
            let expected = want[0].floats().unwrap();
            assert!(
                expected[0] > 1e-3 && expected.iter().all(|v| v.is_finite()),
                "n={n} w={width} shift={shift} gap={gap} expected[0]={:?}",
                expected[0]
            );
            for settings in runs() {
                let dev = Device::guess(&settings).favoring(RULE);
                if dev.platform.backend != "host" && dev.platform.backend != "metal" {
                    continue;
                }
                let plan = lower(&g, 0, &sizes(&run), &dev).unwrap();
                let names = kernels(&plan);
                let expect_kernel = if width == 128
                    && dev.platform.backend == "metal"
                    && dev.platform.limits.msl_bfloat
                {
                    METAL
                } else {
                    RULE
                };
                assert!(names.contains(&expect_kernel), "picked {names:?}");
                let mut runner = open(plan, &settings).unwrap();
                let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
                let got = runner.run(&views).unwrap().remove(0);
                let got = got.floats().unwrap();
                assert!(
                    got.iter().all(|v| v.is_finite()),
                    "n={n} w={width} shift={shift} gap={gap} on {} ({expect_kernel}) non-finite",
                    runner.device()
                );
                let rel = got
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
                    .fold(0.0f32, f32::max);
                assert!(
                    rel < 2e-2,
                    "n={n} w={width} shift={shift} gap={gap} on {} ({expect_kernel}): rel {rel:e}",
                    runner.device()
                );
            }
        }
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
