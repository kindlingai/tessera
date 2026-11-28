//! Verifies native Metal simdgroup_matrix prefill matmul across supported forms.

use tessera_linear as _;
use tessera_metal_linear as _;

use tessera_graph::DimName::{Hidden, Rows, Vocab};
use tessera_graph::{DType, Dim, PathSeg, Port, Role, TensorType};
use tessera_interp::{Tensor, run};
use tessera_lower::form::named;
use tessera_lower::{Backend, Device, Pick, Pins, View, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, max_abs_diff};
use tessera_trace::{EntryKind, Linear, Model, Weights};

#[test]
fn mm_metal_matches() {
    let Some(settings) = runs()
        .into_iter()
        .find(|s| matches!(s.backend, Pick::Gpu(Backend::Metal)))
    else {
        return;
    };
    let mut draws = Noise(97);
    let forms = [
        ("f16", false, 64usize, 5e-2f32),
        ("fp8", false, 32, 0.4),
        ("mxfp8", true, 9, 0.4),
        ("q8_0", true, 9, 0.1),
        ("int4_g32", true, 9, 0.5),
        ("q4_0", true, 9, 0.5),
    ];
    let shapes = [
        (11usize, 19usize, 32usize),
        (32, 64, 64),
        (37, 131, 96),
        (71, 73, 37),
    ];
    for &(m, n, k) in &shapes {
        let model = Model::new(vec![
            Dim::bounded(Rows, m as u64),
            Dim::fixed(Hidden, k as u64),
            Dim::fixed(Vocab, n as u64),
        ]);
        let [x] = model.entry(
            EntryKind::Step,
            [Port {
                role: Role::Tokens,
                ty: TensorType::new([Rows, Hidden], DType::F32),
            }],
        );
        let mut path = vec![PathSeg::Field("w".into())];
        let w_sym = Linear::declare(&model, &mut path, &[Hidden.into(), Vocab.into()]);
        let out = x.matmul(&w_sym);
        let g = model.finish([(Role::Logits, &out)]);
        let x_t = Tensor::f32(vec![m, k], (0..m * k).map(|_| draws.draw() * 0.5).collect());
        let w_t = Tensor::f32(vec![n, k], (0..n * k).map(|_| draws.draw() * 0.5).collect());
        let want = run(
            &g,
            std::slice::from_ref(&w_t),
            0,
            std::slice::from_ref(&x_t),
        )
        .unwrap();
        let views: Vec<View> = vec![x_t.view()];
        for &(fname, block, min_m, tol) in &forms {
            if m < min_m || (block && !k.is_multiple_of(32)) {
                continue;
            }
            let form = named(fname).unwrap();
            let pins = Pins {
                weights: form,
                ..Default::default()
            };
            let dev_new = Device::guess(&settings)
                .favoring("mm_metal")
                .pinned(pins.clone());
            let plan_new = lower(&g, 0, &[(Rows, m as u64)], &dev_new).unwrap();
            assert!(
                plan_new.steps.iter().any(|s| s.kernel.name() == "mm_metal"),
                "expected mm_metal for {fname} {m}x{n}x{k}"
            );
            let mut r_new = open(plan_new, &settings).unwrap();
            r_new.load(0, w_t.view()).unwrap();
            let got = r_new.run(&views).unwrap().remove(0);

            // Check exact F32 parity against baseline Metal kernel.
            let dev_base = Device::guess(&settings).without("mm_metal").pinned(pins);
            let plan_base = lower(&g, 0, &[(Rows, m as u64)], &dev_base).unwrap();
            let mut r_base = open(plan_base, &settings).unwrap();
            r_base.load(0, w_t.view()).unwrap();
            let base = r_base.run(&views).unwrap().remove(0);

            let off_base = max_abs_diff(got.floats().unwrap(), base.floats().unwrap());
            assert!(
                off_base < 5e-4,
                "{fname} {m}x{n}x{k}: off vs baseline {off_base:e}"
            );
            let off_ref = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
            assert!(
                off_ref < tol,
                "{fname} {m}x{n}x{k}: off vs interp {off_ref:e}"
            );
        }
    }
}
