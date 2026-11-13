//! The tiled matmul against the CPU interpreter, on the GPU.

// The other matmul rules come from here.
use kernels as _;

use gpukit::testing::{Noise, max_abs_diff};
use graph::DimName::Rows;
use interp::{Tensor, run};

mod common;
use common::{Finish, finished, linear, noise, runs, terms};
use linear::{SCHEDULES, Schedule};
use lower::form::{BF16_FORM, F32_FORM};
use lower::{Device, Dialect, Pins, Source, lower, open, round, rules};
use settings::read as settings;
use trace::Act;

/// A device that can only choose schedule `at`.
fn only(at: usize) -> Device {
    let others = SCHEDULES.iter().enumerate().filter(|(i, _)| *i != at);
    let bar = |d: Device, (_, s): (usize, &Schedule)| d.without(s.names[0]).without(s.names[1]);
    others.fold(Device::guess(&settings()).favoring("tiled"), bar)
}

#[test]
fn rule_registers() {
    assert!(rules().iter().any(|r| r.name == "tiled"));
    let g = linear(72, 50);
    let plan = lower(
        &g,
        0,
        &[(Rows, 37)],
        &Device::guess(&settings()).favoring("tiled"),
    )
    .unwrap();
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
    // The host skips the largest size and the timing.
    let runs = runs();
    let host = runs[0].gpu().is_none();
    for (m, n, k) in &sizes[..sizes.len() - host as usize] {
        let (m, n, k) = (*m, *n, *k);
        let g = linear(k, n);
        let x = noise(vec![m as usize, k as usize], &mut draws);
        for form in [F32_FORM, BF16_FORM] {
            // The reference sees the weights as the kernel stores them.
            let w = noise(vec![n as usize, k as usize], &mut draws);
            let rounded = w.floats().unwrap().iter().map(|x| round(form.dtype(), *x));
            let weights = [Tensor::f32(w.shape.clone(), rounded.collect())];
            let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
            for ((at, schedule), settings) in SCHEDULES.iter().enumerate().zip(runs.iter().cycle())
            {
                let device = only(at).pinned(Pins::all(form));
                let plan = lower(&g, 0, &[(Rows, m)], &device).unwrap();
                let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                let name = schedule.names[(form.name() == "bf16") as usize];
                assert_eq!(names, [name]);
                let terms = terms(&plan);
                let mut runner = open(plan, settings).unwrap();
                runner.load(0, weights[0].view()).unwrap();
                let got = runner.run(&[x.view()]).unwrap().remove(0);
                let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                let (xs, ws) = (x.floats().unwrap(), weights[0].floats().unwrap());
                let bound = terms.matmul(&runner.precision(), xs, ws, k as usize);
                let label = format!("{m}x{n}x{k} {name} on {}", runner.device());
                assert!(off <= bound, "{label}: off by {off:e}, allowed {bound:e}");
                // A bound past the outputs themselves says nothing.
                let most = want[0]
                    .floats()
                    .unwrap()
                    .iter()
                    .fold(0.0f32, |m, v| m.max(v.abs()));
                assert!(
                    bound < most / 10.0,
                    "{label}: {bound:e} against outputs of {most:e}"
                );
                if host {
                    eprintln!("{label}: off by {off:.2e} of {bound:.2e}");
                    continue;
                }
                let seconds = runner.run_timed(&[x.view()]).unwrap()[0];
                let rate = 2.0 * (m * n * k) as f64 / seconds / 1e12;
                eprintln!("{label}: {rate:.2} TFLOP/s, off by {off:.2e} of {bound:.2e}");
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
        let device = Device::guess(&settings()).favoring("tiled");
        let plan = lower(&g, 0, &[(Rows, 70)], &device).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        assert!(
            names.len() == 1 && names[0].starts_with("tiled_"),
            "{names:?}"
        );
        assert_eq!(plan.steps[0].covers.len(), g.nodes.len() - 2);
        let mut runner = open(plan, &settings()).unwrap();
        runner.load(0, weights[0].view()).unwrap();
        let got = runner.run(&[x.view()]).unwrap().remove(0);
        let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(off < 1e-3, "fused chain off by {off:e}");
    }
}
