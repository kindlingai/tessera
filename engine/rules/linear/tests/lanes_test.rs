//! The lane schedule against the CPU interpreter, on the GPU.

// The other matmul rules come from here.
use kernels as _;
use linear as _;

use gpukit::testing::{Noise, max_abs_diff};
use graph::DimName::{self, Rows};
use interp::run;

mod common;
use common::{Finish, finished, linear, noise, runs, terms};
use interp::Tensor;
use lower::Pins;
use lower::{Device, Dialect, Source, lower, open, rules};
use settings::read as settings;
use trace::Act;

#[test]
fn rule_registers() {
    assert!(rules().iter().any(|r| r.name == "lanes"));
    let g = linear(52, 7);
    let sizes = |m| [(Rows, m)];
    let plan = lower(
        &g,
        0,
        &sizes(3),
        &Device::guess(&settings()).favoring("lanes"),
    )
    .unwrap();
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    assert_eq!(names, ["lanes"]);
    // Past its row limit, another rule takes the site.
    let wide = lower(
        &g,
        0,
        &sizes(33),
        &Device::guess(&settings()).favoring("lanes"),
    )
    .unwrap();
    assert!(wide.steps.iter().all(|s| s.kernel.name() != "lanes"));
    // A favored rule takes the chain that finishes its product.
    let residual = finished(48, 48, |x, product| x.add(&product));
    let thin = Device::guess(&settings()).favoring("thin");
    let kept = lower(&residual, 0, &sizes(3), &thin).unwrap();
    let names: Vec<&str> = kept.steps.iter().map(|s| s.kernel.name()).collect();
    assert_eq!(names, ["thin"]);
    // Weights come in wide loads, under a hashed entry name.
    let Source::Made(body) = plan.steps[0].kernel.source() else {
        panic!("lanes are generated");
    };
    for dialect in [Dialect::Metal, Dialect::Cuda] {
        let text = body.text().render(dialect);
        assert!(text.contains("f_f32x4(w + (at + (u64)i))") && text.contains("linear_f32_"));
    }
}

#[test]
fn lanes_match_cpu() {
    let mut draws = Noise(23);
    // Widths that lanes do not divide, and one they do.
    let plain = [(52, 7), (4, 3), (1024, 96)].map(|(k, n)| (k, n, linear(k, n), true));
    // The sum as an operand, then as the chain's input.
    let residual: Finish = |x, product| x.add(&product);
    let gate: Finish = |x, product| product.act(Act::Silu).mul(x);
    let fused = [residual, gate].map(|finish| (64, 64, finished(64, 64, finish), false));
    for (k, n, g, bare) in plain.into_iter().chain(fused) {
        let rows: DimName = Rows;
        let w = noise(vec![n as usize, k as usize], &mut draws);
        let forms = common::forms(k as usize).into_iter();
        for ((m, settings), form) in [1usize, 5]
            .into_iter()
            .cycle()
            .zip(runs().iter().cycle())
            .zip(forms)
        {
            // The reference sees the weights as the kernel stores them.
            let rounded = form.roundtrip(w.floats().unwrap());
            let weights = [Tensor::f32(w.shape.clone(), rounded)];
            let x = noise(vec![m, k as usize], &mut draws);
            let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
            let device = Device::guess(settings)
                .favoring("lanes")
                .pinned(Pins::all(form));
            let plan = lower(&g, 0, &[(rows, m as u64)], &device).unwrap();
            // Stages after the matmul ride in the same launch.
            let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
            assert_eq!(names, [form.suffixed("lanes")]);
            assert_eq!(plan.steps[0].covers.len(), g.nodes.len() - 2);
            let terms = terms(&plan);
            let mut runner = open(plan, settings).unwrap();
            runner.load(0, weights[0].view()).unwrap();
            let got = runner.run(&[x.view()]).unwrap().remove(0);
            let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
            // The dot's bound leaves out a fused chain's stages.
            let (xs, ws) = (x.floats().unwrap(), weights[0].floats().unwrap());
            let bound = match bare {
                true => terms.matmul(&runner.precision(), xs, ws, k as usize),
                false => 1e-3,
            };
            let label = format!("{m}x{k}->{n} as {} on {}", form.name(), runner.device());
            assert!(off <= bound, "{label}: off by {off:e}, allowed {bound:e}");
            eprintln!("{label}: off by {off:.2e} of {bound:.2e}");
            // Rows picking single weights read them back exactly.
            let (k, n) = (k as usize, n as usize);
            let pick = |r: usize| 13 * r % k;
            let basis = Vec::from_iter((0..m * k).map(|i| f32::from(i % k == pick(i / k))));
            let x = Tensor::f32(vec![m, k], basis);
            let picked = runner.run(&[x.view()]).unwrap();
            let rows = picked[0].floats().unwrap().chunks(n).enumerate();
            for (r, out) in rows.filter(|_| bare) {
                let want: Vec<f32> = (0..n).map(|c| ws[c * k + pick(r)]).collect();
                assert_eq!(out, &want[..], "{label}: row {r}");
            }
        }
    }
}
