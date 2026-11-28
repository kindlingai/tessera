//! Parity and narrow-form lowering checks for Walsh-Hadamard transform.

use tessera_kernels as _;

use tessera_graph::DimName::{Hidden, Rows};
use tessera_graph::{DType, Dim, Graph, TensorType};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM, Load as Stored};
use tessera_lower::{Device, Pins, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, max_abs_diff};
use tessera_trace::{EntryKind, Model, Port, Role};

fn hadamard_graph(width: u64) -> Graph {
    let m = Model::new(vec![Dim::bounded(Rows, 512), Dim::fixed(Hidden, width)]);
    let [x] = m.entry(
        EntryKind::Forward,
        [Port {
            role: Role::Tokens,
            ty: TensorType::new([Rows, Hidden], DType::F32),
        }],
    );
    // Double transform creates an internal activation edge for narrow pins.
    let mid = x.hadamard(128);
    let out = mid.hadamard(128);
    m.finish([(Role::Logits, &out)])
}

#[test]
fn hadamard_narrow_matches_cpu() {
    let g = hadamard_graph(512);
    let mut noise = Noise(42);
    let x = Tensor::f32(vec![4, 512], (0..4 * 512).map(|_| noise.draw()).collect());
    let want = run(&g, &[], 0, std::slice::from_ref(&x)).unwrap();
    let want = want[0].floats().unwrap();
    let forms: [(&'static dyn Stored, f32); 3] =
        [(F32_FORM, 1e-4), (BF16_FORM, 5e-2), (FP8_FORM, 5e-1)];
    for settings in runs() {
        for (form, tol) in forms {
            let pins = Pins::default().activations(form);
            let dev = Device::guess(&settings).pinned(pins);
            let plan = lower(&g, 0, &[(Rows, 4)], &dev).unwrap();
            let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
            if settings.gpu().is_some() {
                assert_eq!(names, ["hadamard", "hadamard"], "{form:?}: {names:?}");
            }
            let mut runner = open(plan, &settings).unwrap();
            let got = runner.run(&[x.view()]).unwrap();
            let diff = max_abs_diff(got[0].floats().unwrap(), want);
            assert!(diff < tol, "{form:?} {:?}: diff={diff}", settings.backend);
        }
    }
}

#[test]
fn hadamard_fp8_input_lowers() {
    let m = Model::new(vec![Dim::bounded(Rows, 8), Dim::fixed(Hidden, 256)]);
    let [x] = m.entry(
        EntryKind::Forward,
        [Port {
            role: Role::Tokens,
            ty: TensorType::new([Rows, Hidden], DType::F32),
        }],
    );
    // Pin internal tensor to fp8 so hadamard reads fp8.
    let mid = x.scale(1.0);
    let out = mid.hadamard(128);
    let g = m.finish([(Role::Logits, &out)]);
    let pins = Pins::default().activations(FP8_FORM);
    let dev = Device::default().pinned(pins);
    let plan = lower(&g, 0, &[(Rows, 2)], &dev).unwrap();
    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
    assert!(names.contains(&"hadamard"), "{names:?}");
}
