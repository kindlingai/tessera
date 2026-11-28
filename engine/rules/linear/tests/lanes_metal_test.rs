//! Single-row Metal GEMV tests across all regimes.

use tessera_kernels as _;
use tessera_linear as _;

#[allow(dead_code, reason = "only a subset of common helpers is used here")]
mod common;
use common::{Finish, finished, linear, noise};
use tessera_graph::DimName::Rows;
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM};
use tessera_lower::{Backend, Device, Limits, Pins, Platform, lower};
use tessera_run::open;
use tessera_run::testing::{emulations as runs, host};
use tessera_testkit::{Noise, max_abs_diff};
use tessera_trace::Act;

#[test]
fn lanes_metal_matches_cpu() {
    let mut draws = Noise(71);
    // Small-K, small-N and wide-N shapes (pair, non-pair, unaligned).
    let shapes = [
        (64, 2304),
        (16, 19),
        (512, 24),
        (512, 128),
        (512, 384),
        (256, 516),
        (260, 515),
    ];
    for (k, n) in shapes {
        let g = linear(k, n);
        let w = noise(vec![n as usize, k as usize], &mut draws);
        for form in [BF16_FORM, F32_FORM] {
            let rounded = form.roundtrip(w.floats().unwrap());
            let weights = [Tensor::f32(w.shape.clone(), rounded)];
            let x = noise(vec![1, k as usize], &mut draws);
            let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
            for settings in runs() {
                let device = Device::guess(&settings)
                    .favoring("lanes_metal")
                    .pinned(Pins::all(form));
                let plan = lower(&g, 0, &[(Rows, 1)], &device).unwrap();
                let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                if settings.gpu().is_none() || settings.gpu() == Some(Backend::Metal) {
                    assert_eq!(names, [form.suffixed("lanes_metal")]);
                } else {
                    assert!(
                        names.iter().all(|n| !n.starts_with("lanes_metal")),
                        "non-Metal runner picked lanes_metal: {names:?}"
                    );
                }
                let mut runner = open(plan, &settings).unwrap();
                runner.load(0, weights[0].view()).unwrap();
                let got = runner.run(&[x.view()]).unwrap().remove(0);
                let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                assert!(
                    off < 5e-3,
                    "1x{k}->{n} {} on {}: off {off:e}",
                    form.name(),
                    runner.device()
                );
            }
        }
    }
}

#[test]
fn lanes_metal_fuses_after() {
    let mut draws = Noise(89);
    let tanh_finish: Finish = |_x, product| product.act(Act::Tanh);
    let add_finish: Finish = |x, product| x.add(&product);
    let cases: [(u64, u64, Finish); 3] = [
        (64, 128, tanh_finish),
        (256, 256, add_finish),
        (256, 520, tanh_finish),
    ];
    for (k, n, fin) in cases {
        let g = finished(k, n, fin);
        let w = noise(vec![n as usize, k as usize], &mut draws);
        let rounded = BF16_FORM.roundtrip(w.floats().unwrap());
        let weights = [Tensor::f32(w.shape.clone(), rounded)];
        let x = noise(vec![1, k as usize], &mut draws);
        let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
        let settings = host();
        let device = Device::guess(&settings)
            .favoring("lanes_metal")
            .pinned(Pins::all(BF16_FORM));
        let plan = lower(&g, 0, &[(Rows, 1)], &device).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        assert_eq!(names, ["lanes_metal_bf16"]);
        let mut runner = open(plan, &settings).unwrap();
        runner.load(0, weights[0].view()).unwrap();
        let got = runner.run(&[x.view()]).unwrap().remove(0);
        let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(off < 5e-3, "fused 1x{k}->{n}: off {off:e}");
    }
}

#[test]
fn metal_picks_lanes_metal() {
    let apple = Device::guess(&host())
        .on(Platform {
            backend: "metal".into(),
            arch: "apple9".into(),
            name: "Apple M4 Pro".into(),
            units: 20,
            bandwidth: 273e9,
            ..Platform::assumed(Limits::assumed())
        })
        .pinned(Pins::all(BF16_FORM));
    let shapes = [
        (64, 2304),
        (2304, 24),
        (4608, 64),
        (2304, 128),
        (2304, 1024),
        (3072, 2304),
        (2304, 3072),
        (2304, 201024),
    ];
    let mut missed = Vec::new();
    for (k, n) in shapes {
        let g = linear(k, n);
        let plan = lower(&g, 0, &[(Rows, 1)], &apple).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        if names != ["lanes_metal_bf16"] {
            missed.push((k, n, names));
        }
    }
    assert!(missed.is_empty(), "missed: {missed:?}");
    for (backend, arch) in [("cuda", "sm_121"), ("rocm", "gfx11_5")] {
        let other = Device::guess(&host())
            .on(Platform {
                backend: backend.into(),
                arch: arch.into(),
                name: arch.into(),
                ..Platform::assumed(Limits::assumed())
            })
            .favoring("lanes_metal")
            .pinned(Pins::all(BF16_FORM));
        for (k, n) in [(64, 2304), (2304, 1024)] {
            let plan = lower(&linear(k, n), 0, &[(Rows, 1)], &other).unwrap();
            assert!(
                plan.steps
                    .iter()
                    .all(|s| !s.kernel.name().starts_with("lanes_metal")),
                "{backend} should not pick lanes_metal for 1x{k}->{n}"
            );
        }
    }
}

#[test]
fn multi_row_exact() {
    let mut draws = Noise(97);
    let settings = host();
    let device = Device::guess(&settings)
        .favoring("lanes_metal")
        .pinned(Pins::all(BF16_FORM));
    for (k, n) in [(64, 32), (256, 128), (256, 520)] {
        let g = linear(k, n);
        let w = noise(vec![n as usize, k as usize], &mut draws);
        let rounded = BF16_FORM.roundtrip(w.floats().unwrap());
        let weights = [Tensor::f32(w.shape.clone(), rounded)];
        let x4 = noise(vec![4, k as usize], &mut draws);
        let x1 = Tensor::f32(
            vec![1, k as usize],
            x4.floats().unwrap()[..k as usize].to_vec(),
        );
        let plan4 = lower(&g, 0, &[(Rows, 4)], &device).unwrap();
        let plan1 = lower(&g, 0, &[(Rows, 1)], &device).unwrap();
        assert_eq!(plan4.steps[0].kernel.name(), "lanes_metal_bf16");
        assert_eq!(plan1.steps[0].kernel.name(), "lanes_metal_bf16");
        let mut run4 = open(plan4, &settings).unwrap();
        let mut run1 = open(plan1, &settings).unwrap();
        run4.load(0, weights[0].view()).unwrap();
        run1.load(0, weights[0].view()).unwrap();
        let out4 = run4.run(&[x4.view()]).unwrap().remove(0);
        let out1 = run1.run(&[x1.view()]).unwrap().remove(0);
        let bits4: Vec<u32> = out4.floats().unwrap()[..n as usize]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let bits1: Vec<u32> = out1.floats().unwrap().iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits4, bits1, "m=4 row 0 diverged from m=1 on {k}->{n}");
    }
}
