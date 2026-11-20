//! Lane and picked kernels against the CPU interpreter.

// The other matmul rules come from here.
use sensai_kernels as _;
use sensai_linear as _;

use sensai_graph::Dim;
use sensai_graph::DimName::{self, Rows, Seqs};
use sensai_interp::run;

mod common;
use common::{Finish, finished, linear, noise, terms};
use sensai_interp::Tensor;
use sensai_lower::Pins;
use sensai_lower::form::named;
use sensai_lower::{Device, Dialect, Source, lower};
use sensai_run::open;
use sensai_run::testing::{emulations as runs, host};
use sensai_testkit::{Noise, cosine_f32, fill, graphs, max_abs_diff, weights};
use sensai_trace::Act;

#[test]
fn rule_registers() {
    let g = linear(52, 7);
    let sizes = |m| [(Rows, m)];
    let device = Device::guess(&host()).favoring("lanes");
    let plan = lower(&g, 0, &sizes(3), &device).unwrap();
    // Past its row limit, another rule takes the site.
    let wide = lower(&g, 0, &sizes(33), &device).unwrap();
    assert!(wide.steps.iter().all(|s| s.kernel.name() != "lanes"));
    // A favored rule takes the chain that finishes its product.
    let residual = finished(48, 48, |x, product| x.add(&product));
    let thin = Device::guess(&host()).favoring("thin");
    let kept = lower(&residual, 0, &sizes(3), &thin).unwrap();
    let names: Vec<&str> = kept.steps.iter().map(|s| s.kernel.name()).collect();
    assert_eq!(names, ["thin"]);
    // Weights come in wide loads, under a hashed entry name.
    let Source::Made(body) = plan.steps[0].kernel.source() else {
        panic!("lanes are generated");
    };
    for dialect in [Dialect::Metal, Dialect::Cuda] {
        let text = body.text().render(dialect);
        assert!(text.contains("f_f32x4(w + (at + (u64)o))") && text.contains("linear_f32_"));
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
            // The reference sees the weights in the kernel's stored form.
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

/// Two of four experts per row, from one stacked set.
#[test]
fn picked_matches_cpu() {
    let sizes: [(DimName, u64); 2] = [(Seqs, 2), (Rows, 5)];
    let bounded = vec![Dim::bounded(Seqs, 4), Dim::bounded(Rows, 8)];
    let g = graphs::routed(&bounded, [4, 64, 32]);
    let mut noise = Noise(11);
    let drawn = weights(&g, &mut noise);
    let input = fill(&g, &g.entries[0].inputs[0].ty, &sizes, &mut noise).unwrap();
    for settings in runs() {
        for form in ["f32", "bf16", "q8_0"] {
            let form = named(form).unwrap();
            let device = Device::guess(&settings)
                .favoring("picked")
                .pinned(Pins::all(form));
            // The interpreter sees the weights after the form's round trip.
            let kept: Vec<Tensor> = (drawn.iter())
                .map(|t| Tensor::f32(t.shape.clone(), form.roundtrip(t.floats().unwrap())))
                .collect();
            let want = run(&g, &kept, 0, std::slice::from_ref(&input)).unwrap();
            let plan = lower(&g, 0, &sizes, &device).unwrap();
            let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
            assert!(names.iter().any(|n| n.starts_with("picked")), "{names:?}");
            let mut runner = open(plan, &settings).unwrap();
            for (at, w) in kept.iter().enumerate() {
                runner.load(at as u32, w.view()).unwrap();
            }
            let got = runner.run(&[input.view()]).unwrap().remove(0);
            let (got, want) = (got.floats().unwrap(), want[0].floats().unwrap());
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let (off, cos) = (max_abs_diff(got, want) / scale, cosine_f32(got, want));
            assert!(
                off < 1e-3 && cos > 0.9999,
                "{}: off {off:e} cos {cos}",
                form.name()
            );
        }
    }
}
