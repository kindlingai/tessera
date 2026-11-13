//! The tiled matmul against the CPU interpreter, on the GPU.

// The other matmul rules come from here.
use kernels as _;

use gpukit::testing::{Noise, max_abs_diff};
use graph::DimName::Rows;
use interp::{Tensor, run};

mod common;
use common::{Finish, finished, linear, noise};
use linear::{SCHEDULES, Schedule};
use lower::{Device, Dialect, Pins, Source, lower, open, round, rules};
use trace::{Act, DType};

/// A device that can only choose schedule `at`.
fn only(at: usize) -> Device {
    let others = SCHEDULES.iter().enumerate().filter(|(i, _)| *i != at);
    let bar = |d: Device, (_, s): (usize, &Schedule)| d.without(s.names[0]).without(s.names[1]);
    others.fold(Device::guess().favoring("tiled"), bar)
}

#[test]
fn rule_registers() {
    assert!(rules().iter().any(|r| r.name == "tiled"));
    let g = linear(72, 50);
    let plan = lower(&g, 0, &[(Rows, 37)], &Device::guess().favoring("tiled")).unwrap();
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    assert_eq!(names.len(), 1, "{names:?}");
    assert!(names[0].starts_with("tiled_"), "{names:?}");
    let Source::Made(body) = plan.steps[0].kernel.source() else {
        panic!("tiled is generated");
    };
    for dialect in [Dialect::Metal, Dialect::Cuda] {
        let text = body.text().render(dialect);
        assert!(text.contains("TILE_MMA(acc_t[i][j], ta[i], tb[j])"));
    }
}

#[test]
fn tiles_match_cpu() {
    let mut draws = Noise(29);
    // Sizes off the block, on it, and beyond it.
    let sizes = [
        (1, 3, 4),
        (37, 50, 72),
        (64, 64, 64),
        (130, 96, 1024),
        (256, 512, 1024),
    ];
    for (m, n, k) in sizes {
        let g = linear(k, n);
        let x = noise(vec![m as usize, k as usize], &mut draws);
        for form in [DType::F32, DType::Bf16] {
            // The reference sees the weights as the kernel stores them.
            let w = noise(vec![n as usize, k as usize], &mut draws);
            let rounded = w.floats().unwrap().iter().map(|x| round(form, *x));
            let weights = [Tensor::f32(w.shape.clone(), rounded.collect())];
            let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
            for (at, schedule) in SCHEDULES.iter().enumerate() {
                let device = only(at).pinned(Pins::all(form));
                let plan = lower(&g, 0, &[(Rows, m)], &device).unwrap();
                let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                let name = schedule.names[(form == DType::Bf16) as usize];
                assert_eq!(names, [name]);
                let mut runner = open(plan).unwrap();
                runner.load(0, weights[0].view()).unwrap();
                let got = runner.run(&[x.view()]).unwrap().remove(0);
                let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                let seconds = runner.run_timed(&[x.view()]).unwrap()[0];
                let rate = 2.0 * (m * n * k) as f64 / seconds / 1e12;
                eprintln!("{m}x{n}x{k} {name}: {rate:.2} TFLOP/s, off by {off:.2e}");
                assert!(off < 1e-3, "{m}x{n}x{k} {name}: off by {off:e}");
            }
        }
    }
}

#[test]
fn stages_ride_along() {
    let mut draws = Noise(31);
    // The sum as an operand, then as the chain's input.
    let residual: Finish = |x, product| x.add(&product);
    let gate: Finish = |x, product| product.act(Act::Silu).mul(x);
    for finish in [residual, gate] {
        let g = finished(48, 48, finish);
        let weights = [noise(vec![48, 48], &mut draws)];
        let x = noise(vec![70, 48], &mut draws);
        let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
        let device = Device::guess().favoring("tiled");
        let plan = lower(&g, 0, &[(Rows, 70)], &device).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        assert!(
            names.len() == 1 && names[0].starts_with("tiled_"),
            "{names:?}"
        );
        assert_eq!(plan.steps[0].covers.len(), g.nodes.len() - 2);
        let mut runner = open(plan).unwrap();
        runner.load(0, weights[0].view()).unwrap();
        let got = runner.run(&[x.view()]).unwrap().remove(0);
        let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(off < 1e-3, "fused chain off by {off:e}");
    }
}
