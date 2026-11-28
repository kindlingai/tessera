//! Parity tests for native Metal stacked block matmul kernels.

use tessera_kernels as _;
use tessera_linear as _;
use tessera_metal_linear as _;

use tessera_graph::DimName::{Heads, Hidden, Rows};
use tessera_graph::{DType, Dim, DimExpr, Graph, PathSeg, TensorType};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM, Load as Stored};
use tessera_lower::{Backend, Device, Pins, Settings, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, cosine_f32, max_abs_diff};
use tessera_trace::{EntryKind, Model, Port, Role};

fn metal_settings() -> Option<Settings> {
    runs().into_iter().find(|s| s.gpu() == Some(Backend::Metal))
}

fn draw(shape: Vec<usize>, noise: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
    Tensor::f32(shape, data)
}

fn stacked_graph(
    batches: u64,
    block_rows: u64,
    width: u64,
    slice_start: u64,
    slice_len: u64,
    swapped: bool,
) -> Graph {
    let k = if swapped { slice_len } else { width };
    let m = Model::new(vec![
        Dim::fixed(Heads, batches),
        Dim::bounded(Rows, 128),
        Dim::fixed(Hidden, k),
    ]);
    let [x] = m.entry(
        EntryKind::Forward,
        [Port {
            role: Role::Tokens,
            ty: TensorType::new([Heads, Rows, Hidden], DType::F32),
        }],
    );
    let w = m.weight(
        &[PathSeg::Field("blocks".into())],
        TensorType::new(
            vec![DimExpr::from(batches * block_rows), DimExpr::from(width)],
            DType::F32,
        ),
    );
    let w3 = w.reshape(vec![
        DimExpr::from(batches),
        DimExpr::from(block_rows),
        DimExpr::from(width),
    ]);
    let sliced = if slice_start > 0 || slice_len < block_rows {
        w3.slice(1, slice_start, slice_len)
    } else {
        w3
    };
    let ready = if swapped {
        sliced.permute(&[0, 2, 1])
    } else {
        sliced
    };
    let out = x.matmul_rows(&ready);
    m.finish([(Role::Logits, &out)])
}

#[test]
fn stacked_metal_matches_cpu() {
    let Some(settings) = metal_settings() else {
        return;
    };
    let mut draws = Noise(71);
    type Case = (
        u64,
        u64,
        u64,
        u64,
        u64,
        bool,
        usize,
        &'static dyn Stored,
        &'static dyn Stored,
        f32,
    );
    let cases: &[Case] = &[
        // Non-transposed decode and prefill across formats and sliced offsets.
        (4, 32, 64, 0, 32, false, 1, F32_FORM, F32_FORM, 1e-3),
        (4, 48, 128, 16, 32, false, 3, F32_FORM, BF16_FORM, 5e-3),
        (4, 32, 128, 0, 32, false, 1, BF16_FORM, BF16_FORM, 5e-2),
        (4, 32, 128, 0, 32, false, 3, F32_FORM, FP8_FORM, 5e-3),
        (3, 37, 73, 5, 29, false, 7, F32_FORM, F32_FORM, 1e-3),
        (4, 64, 128, 0, 64, false, 16, F32_FORM, BF16_FORM, 5e-3),
        (3, 70, 96, 2, 66, false, 33, F32_FORM, FP8_FORM, 5e-3),
        // Transposed decode and prefill across formats and edge shapes.
        (4, 64, 128, 0, 64, true, 1, F32_FORM, F32_FORM, 1e-3),
        (4, 96, 128, 16, 64, true, 3, F32_FORM, BF16_FORM, 5e-3),
        (4, 64, 128, 0, 64, true, 1, BF16_FORM, BF16_FORM, 5e-2),
        (4, 64, 128, 0, 64, true, 3, F32_FORM, FP8_FORM, 5e-3),
        (3, 45, 73, 4, 37, true, 7, F32_FORM, F32_FORM, 1e-3),
        (4, 64, 128, 0, 64, true, 16, F32_FORM, BF16_FORM, 5e-3),
        (3, 50, 130, 3, 41, true, 33, F32_FORM, FP8_FORM, 5e-3),
    ];

    for &(batches, block_rows, width, start, len, swapped, rows, act_form, w_form, tol) in cases {
        let g = stacked_graph(batches, block_rows, width, start, len, swapped);
        let k = if swapped { len } else { width };
        let w_raw = draw(
            vec![(batches * block_rows) as usize, width as usize],
            &mut draws,
        );
        let w_ref = Tensor::f32(
            w_raw.shape.clone(),
            w_form.roundtrip(w_raw.floats().unwrap()),
        );
        let x_raw = draw(vec![batches as usize, rows, k as usize], &mut draws);
        let x_ref = Tensor::f32(
            x_raw.shape.clone(),
            act_form.roundtrip(x_raw.floats().unwrap()),
        );
        let want = run(
            &g,
            std::slice::from_ref(&w_ref),
            0,
            std::slice::from_ref(&x_ref),
        )
        .unwrap();

        let mut pins = Pins::all(w_form);
        if act_form.name() != "f32" {
            pins = pins.activations(act_form);
        }
        let device = Device::guess(&settings).pinned(pins);
        let plan = lower(&g, 0, &[(Rows, rows as u64)], &device).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        let expected = w_form.suffixed(if swapped {
            "stacked_metal_t"
        } else {
            "stacked_metal"
        });
        assert!(
            names.contains(&expected),
            "expected {expected} in {names:?}"
        );

        let mut runner = open(plan, &settings).unwrap();
        runner.load(0, w_ref.view()).unwrap();
        let got = runner.run(&[x_ref.view()]).unwrap().remove(0);
        let (got_f, want_f) = (got.floats().unwrap(), want[0].floats().unwrap());
        let scale = want_f.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        let rel = max_abs_diff(got_f, want_f) / scale;
        let cos = cosine_f32(got_f, want_f);
        assert!(
            rel < tol && cos > 0.999,
            "b={batches} r={rows} sw={swapped} {}/{}: rel={rel:e} cos={cos}",
            act_form.name(),
            w_form.name()
        );
    }
}
