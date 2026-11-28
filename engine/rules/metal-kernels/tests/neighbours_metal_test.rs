//! Parity tests for Metal 3x3 neighbours MMA kernel.

use tessera_kernels as _;
use tessera_metal_kernels as _;

use tessera_graph::Dim;
use tessera_graph::DimName::Seqs;
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM, Load as Stored};
use tessera_lower::{Backend, Device, Pins, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, graphs, max_abs_diff};

#[test]
fn neighbours_metal_matches_cpu() {
    let Some(settings) = runs().into_iter().find(|s| s.gpu() == Some(Backend::Metal)) else {
        return;
    };
    let mut noise = Noise(131);
    let cases: [(usize, usize, usize, &'static dyn Stored, f32); 6] = [
        (37, 64, 2, F32_FORM, 1e-3),
        (19, 12, 2, F32_FORM, 1e-3),
        (40, 32, 3, BF16_FORM, 2e-1),
        (25, 20, 2, BF16_FORM, 2e-1),
        (32, 64, 2, FP8_FORM, 1.5),
        (11, 3, 2, F32_FORM, 1e-3),
    ];
    for (pixels, width, seqs, w_form, tol) in cases {
        let g = graphs::neighbours(&[Dim::bounded(Seqs, 4)], [pixels as u64, 9, width as u64]);
        let ids = Tensor::f32(
            vec![pixels, 9],
            (0..pixels * 9)
                .map(|i| ((i / 9 + i % 9) % (pixels + 2)) as f32)
                .collect(),
        );
        let w = Tensor::f32(
            vec![width, 9 * width],
            (0..width * 9 * width).map(|_| noise.draw()).collect(),
        );
        let x = Tensor::f32(
            vec![seqs, pixels, width],
            (0..seqs * pixels * width).map(|_| noise.draw()).collect(),
        );
        let weights = [ids.clone(), w.clone()];
        let want = run(&g, &weights, 0, &[x.clone(), x.clone()]).unwrap();
        let pins = if w_form.name() == "f32" {
            Pins::default()
        } else {
            Pins::default().pin("w", w_form)
        };
        let dev = Device::guess(&settings).pinned(pins);
        let plan = lower(&g, 0, &[(Seqs, seqs as u64)], &dev).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        assert!(
            names.contains(&"neighbours_metal"),
            "p={pixels} w={width} {}: {names:?}",
            w_form.name()
        );
        let mut runner = open(plan, &settings).unwrap();
        runner.load(0, ids.view()).unwrap();
        runner.load(1, w.view()).unwrap();
        let got = runner.run(&[x.view(), x.view()]).unwrap();
        let diff = max_abs_diff(got[1].floats().unwrap(), want[1].floats().unwrap());
        assert!(
            diff < tol,
            "p={pixels} w={width} {}: diff={diff}",
            w_form.name()
        );
    }
}
