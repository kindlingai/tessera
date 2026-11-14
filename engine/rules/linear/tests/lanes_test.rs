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
    for name in ["lanes", "lanes_q8_0", "lanes_q4_0", "lanes_nvfp4"] {
        assert!(rules().iter().any(|r| r.name == name));
    }
    for form in common::forms(64).into_iter().filter(|f| f.block() > 1) {
        let device = Device::guess(&settings()).pinned(Pins::all(form));
        let g = linear(64, 7);
        for (device, name) in [
            (device.clone(), form.suffixed("lanes")),
            (device.favoring("lanes"), form.suffixed("lanes_generic")),
        ] {
            let plan = lower(&g, 0, &[(Rows, 1)], &device).unwrap();
            let kernel = &plan.steps[0].kernel;
            assert_eq!(kernel.name(), name);
            if name.contains("generic") {
                continue;
            }
            let Source::Made(body) = kernel.source() else {
                panic!("lanes are generated");
            };
            let text = body.text();
            assert!(!text.body.contains("form_"));
            let load = if form.name() == "nvfp4" {
                "f_pack4("
            } else {
                "f_pack4_aligned("
            };
            assert!(text.body.contains(load));
            for dialect in [Dialect::Metal, Dialect::Cuda] {
                assert!(text.render(dialect).contains(load));
            }
            compile_cuda(&text.render(Dialect::Cuda), name);
        }
    }
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
    let plain = [(52, 7), (4, 3), (48, 15), (1024, 96)].map(|(k, n)| (k, n, linear(k, n), true));
    // The sum as an operand, then as the chain's input.
    let residual: Finish = |x, product| x.add(&product);
    let gate: Finish = |x, product| product.act(Act::Silu).mul(x);
    let fused = [residual, gate].map(|finish| (64, 64, finished(64, 64, finish), false));
    for (k, n, g, bare) in plain.into_iter().chain(fused) {
        let rows: DimName = Rows;
        let w = noise(vec![n as usize, k as usize], &mut draws);
        let platforms = &runs();
        let cases = common::forms(k as usize).into_iter().flat_map(|form| {
            [1usize, 5]
                .into_iter()
                .flat_map(move |m| platforms.iter().map(move |settings| (m, settings, form)))
        });
        for (m, settings, form) in cases {
            // The reference sees the weights as the kernel stores them.
            let rounded = form.roundtrip(w.floats().unwrap());
            let weights = [Tensor::f32(w.shape.clone(), rounded)];
            let x = noise(vec![m, k as usize], &mut draws);
            let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
            let rule = if form.block() > 1 {
                form.suffixed("lanes")
            } else {
                "lanes"
            };
            let device = Device::guess(settings)
                .favoring(rule)
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
            let label = format!(
                "{m} rows of {k} to {n} as {} on {}",
                form.name(),
                runner.device()
            );
            assert!(off <= bound, "{label}: off by {off:e}, allowed {bound:e}");
            eprintln!("{label}: off by {off:.2e} of {bound:.2e}");
        }
    }
}

fn compile_cuda(text: &str, name: &str) {
    use std::path::Path;
    use std::process::Command;
    let local = "/usr/local/cuda/bin/nvcc";
    let nvcc = if Path::new(local).exists() {
        local
    } else {
        "nvcc"
    };
    if Command::new(nvcc).arg("--version").output().is_err() {
        eprintln!("{name}: nvcc not installed, skipped");
        return;
    }
    let dir = gpukit::testing::scratch(name);
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("kernel.cu");
    let binary = dir.join("kernel.cubin");
    std::fs::write(&source, text).unwrap();
    let result = Command::new(nvcc)
        .args(["-arch=sm_90", "-cubin"])
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    std::fs::remove_file(source).unwrap();
    if binary.exists() {
        std::fs::remove_file(binary).unwrap();
    }
    assert!(
        result.status.success(),
        "{name}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    eprintln!("{name}: nvcc -arch=sm_90 -cubin passed");
}

#[test]
fn blocks_decode() {
    let mut draws = Noise(91);
    for k in [48, 64, 288] {
        let g = linear(k, 7);
        let weights = noise(vec![7, k as usize], &mut draws);
        for form in common::forms(k as usize)
            .into_iter()
            .filter(|f| f.block() > 1)
        {
            let rounded = form.roundtrip(weights.floats().unwrap());
            for settings in runs() {
                for first in (0..k as usize).step_by(32) {
                    let m = (k as usize - first).min(32);
                    let device = Device::guess(&settings)
                        .pinned(Pins::all(form))
                        .favoring(form.suffixed("lanes"));
                    let plan = lower(&g, 0, &[(Rows, m as u64)], &device).unwrap();
                    let mut runner = open(plan, &settings).unwrap();
                    runner.load(0, weights.view()).unwrap();
                    let mut basis = vec![0.0; m * k as usize];
                    for row in 0..m {
                        basis[row * k as usize + first + row] = 1.0;
                    }
                    let x = Tensor::f32(vec![m, k as usize], basis);
                    let got = runner.run(&[x.view()]).unwrap().remove(0);
                    for (row, output) in got.floats().unwrap().as_chunks::<7>().0.iter().enumerate()
                    {
                        for (col, value) in output.iter().enumerate() {
                            assert_eq!(
                                *value,
                                rounded[col * k as usize + first + row],
                                "{} k={k} at={} col={col}",
                                form.name(),
                                first + row
                            );
                        }
                    }
                }
            }
        }
    }
}
