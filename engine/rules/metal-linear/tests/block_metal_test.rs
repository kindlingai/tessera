//! Verifies native Metal block-quantized decode against CPU reference.

use tessera_kernels as _;
use tessera_linear as _;
use tessera_metal_linear as _;

use tessera_graph::Dim;
use tessera_graph::DimName::Rows;
use tessera_interp::{Tensor, run};
use tessera_lower::form::named;
use tessera_lower::{Backend, Device, Pick, Pins, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, cosine_f32, graphs, max_abs_diff};

#[test]
fn block_metal_matches() {
    let Some(settings) = runs()
        .into_iter()
        .find(|s| matches!(s.backend, Pick::Gpu(Backend::Metal)))
    else {
        return;
    };
    let mut draws = Noise(77);
    let fill = |shape: Vec<usize>, d: &mut Noise| {
        let len: usize = shape.iter().product();
        Tensor::f32(shape, (0..len).map(|_| d.draw()).collect())
    };
    for (k, n) in [(32u64, 7u64), (96, 17), (512, 96)] {
        let g = graphs::linear(&[Dim::bounded(Rows, 16)], k, n);
        let w = fill(vec![n as usize, k as usize], &mut draws);
        for name in ["mxfp8", "q8_0", "int4_g32", "q4_0"] {
            let form = named(name).unwrap();
            let rounded = form.roundtrip(w.floats().unwrap());
            let weights = [Tensor::f32(w.shape.clone(), rounded)];
            for m in [1usize, 2, 4, 5, 7, 8, 9] {
                let x = fill(vec![m, k as usize], &mut draws);
                let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
                let device = Device::guess(&settings)
                    .favoring("block_metal")
                    .pinned(Pins::all(form));
                let plan = lower(&g, 0, &[(Rows, m as u64)], &device).unwrap();
                let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                assert_eq!(names, [form.suffixed("block_metal")], "m={m} {name}");
                let mut runner = open(plan, &settings).unwrap();
                runner.load(0, weights[0].view()).unwrap();
                let got = runner.run(&[x.view()]).unwrap().remove(0);
                let (got, want) = (got.floats().unwrap(), want[0].floats().unwrap());
                let off = max_abs_diff(got, want);
                let cos = cosine_f32(got, want);
                assert!(
                    off < 2e-3 && cos > 0.99999,
                    "{name} {m}x{k}->{n}: off={off:e} cos={cos}"
                );
            }
        }
    }
}
