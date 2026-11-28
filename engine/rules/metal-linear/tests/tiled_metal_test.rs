//! Where the Metal bf16 matmul takes dense prefill.

use tessera_kernels as _;
use tessera_linear as _;
use tessera_metal_linear as _;

use tessera_graph::DimName::{Ffn, Hidden, Rows};
use tessera_interp::{Tensor, run};
use tessera_lower::form::BF16_FORM;
use tessera_lower::{Device, Limits, Pins, Plan, Platform, Shape, Source, lower};
use tessera_run::open;
use tessera_run::testing::{host, runs};
use tessera_testkit::{Noise, cosine_f32};
use tessera_trace::{DType, Dim, EntryKind, Graph, Linear, Model, Port, Role, TensorType, Weights};

const RULE: &str = "tiled_metal";

/// Dense prefill products as rows, outputs and inputs.
const PREFILL: [(u64, u64, u64); 4] = [
    (2048, 3072, 2304),
    (1472, 3072, 2304),
    (2048, 2304, 3072),
    (2048, 1024, 2304),
];

#[derive(Weights)]
struct Layer {
    #[shape(Hidden, Ffn)]
    w: Linear,
}

/// `k` inputs to `n` outputs, over prefill rows.
fn dense(k: u64, n: u64) -> Graph {
    let rows = Dim::bounded(Rows, 4096);
    let m = Model::new(vec![rows, Dim::fixed(Hidden, k), Dim::fixed(Ffn, n)]);
    let w: Layer = tessera_trace::declare(&m);
    let ty = TensorType::new([Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let [x] = m.entry(EntryKind::Forward, [Port { role, ty }]);
    m.finish([(Role::Logits, &x.matmul(&w.w))])
}

fn platform(backend: &str, arch: &str, name: &str) -> Platform {
    Platform {
        backend: backend.into(),
        arch: arch.into(),
        name: name.into(),
        units: 20,
        bandwidth: 273e9,
        ..Platform::assumed(Limits::assumed())
    }
}

fn device(on: Platform) -> Device {
    Device::guess(&host()).on(on).pinned(Pins::all(BF16_FORM))
}

fn plan(on: Platform, (m, n, k): (u64, u64, u64)) -> Plan {
    lower(&dense(k, n), 0, &[(Rows, m)], &device(on)).unwrap()
}

fn m4_pro() -> Platform {
    let mut p = platform("metal", "apple9", "Apple M4 Pro");
    p.limits.msl_bfloat = true;
    p
}

fn rules(plan: &Plan) -> Vec<&str> {
    plan.steps.iter().map(|s| s.rule).collect()
}

#[test]
fn prefill_wins_unfavored() {
    for shape in PREFILL {
        let plan = plan(m4_pro(), shape);
        assert_eq!(rules(&plan), [RULE], "{shape:?}");
        assert_eq!(plan.steps[0].kernel.name(), "tiled_metal_bf16");
    }
}

#[test]
fn launch_covers_tiles() {
    let plan = plan(m4_pro(), (1472, 1000, 2304));
    let step = &plan.steps[0];
    assert_eq!(step.rule, RULE);
    let Source::Fixed { entry, .. } = step.kernel.source() else {
        panic!("a fixed kernel");
    };
    assert_eq!(entry, "tiled_metal_bf16");
    let launch = step.kernel.launches(step).remove(0);
    let counts: Vec<_> = launch.args[3..].iter().map(|a| a.u32()).collect();
    assert_eq!(counts, [Some(1472), Some(1000), Some(2304)]);
    let Shape::Grid {
        across,
        down,
        threads,
    } = launch.shape
    else {
        panic!("a grid launch");
    };
    assert_eq!((across, down, threads), (23 * 16, 1, 256));
    assert_eq!(step.kernel.needs().group_bytes, 10_240);
}

#[test]
fn other_backends_skip() {
    for (backend, arch) in [("cuda", "sm_121"), ("rocm", "gfx1151")] {
        let plan = plan(platform(backend, arch, arch), PREFILL[0]);
        assert!(!rules(&plan).contains(&RULE), "{backend}");
    }
    let mut old_msl = m4_pro();
    old_msl.limits.msl_bfloat = false;
    let fallback = plan(old_msl, PREFILL[0]);
    assert!(!rules(&fallback).contains(&RULE), "needs MSL 3.1 bfloat");
}

#[test]
fn few_rows_skip() {
    for rows in [1, 8, 31] {
        let plan = plan(m4_pro(), (rows, 3072, 2304));
        assert!(!rules(&plan).contains(&RULE), "{rows} rows");
    }
    let odd = plan(m4_pro(), (2048, 3072, 2300));
    assert!(!rules(&odd).contains(&RULE));
    let least = plan(m4_pro(), (32, 3072, 2304));
    let mut favored = device(m4_pro()).favoring(RULE);
    favored = favored.requiring_favored(true);
    let forced = lower(&dense(2304, 3072), 0, &[(Rows, 32)], &favored);
    assert!(forced.is_ok(), "32 rows match: {:?}", rules(&least));
}

fn fill(shape: Vec<usize>, draws: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| draws.draw()).collect();
    Tensor::f32(shape, data)
}

#[test]
fn metal_tiled_matches_cpu() {
    let Some(settings) = runs().into_iter().find(|s| s.gpu().is_some()) else {
        return;
    };
    let mut draws = Noise(97);
    let cases = [(64u64, 128u64, 64u64), (37, 100, 40), (95, 70, 72)];
    for (m, n, k) in cases {
        let g = dense(k, n);
        let w = fill(vec![n as usize, k as usize], &mut draws);
        let rounded = BF16_FORM.roundtrip(w.floats().unwrap());
        let weights = [Tensor::f32(w.shape.clone(), rounded)];
        let x = fill(vec![m as usize, k as usize], &mut draws);
        let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
        let dev = Device::guess(&settings)
            .favoring(RULE)
            .pinned(Pins::all(BF16_FORM));
        let plan = lower(&g, 0, &[(Rows, m)], &dev).unwrap();
        if dev.platform.backend == "metal" && dev.platform.limits.msl_bfloat {
            assert_eq!(rules(&plan), [RULE], "{m}x{n}x{k}");
        }
        let mut runner = open(plan, &settings).unwrap();
        runner.load(0, weights[0].view()).unwrap();
        let got = runner.run(&[x.view()]).unwrap().remove(0);
        let cos = cosine_f32(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(cos > 0.9999, "{m}x{n}x{k}: cos {cos}");
    }
}
