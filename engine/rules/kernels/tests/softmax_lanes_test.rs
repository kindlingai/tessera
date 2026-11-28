//! Parity and form tests for warp- and block-parallel softmax.

use tessera_kernels as _;

use tessera_graph::DimName::{Hidden, Rows};
use tessera_graph::{DType, Dim, Graph, TensorType};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM, Load as Stored};
use tessera_lower::{Device, Pins, lower};
use tessera_run::open;
use tessera_run::testing::{host, runs};
use tessera_testkit::{Noise, fill, max_abs_diff};
use tessera_trace::{Act, EntryKind, Model, Port, Role};

fn softmax_graph(width: usize) -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Rows, 64),
        Dim::fixed(Hidden, width as u64),
    ]);
    let port = Port {
        role: Role::Logits,
        ty: TensorType::new([Rows, Hidden], DType::F32),
    };
    let [x] = m.entry(EntryKind::Forward, [port]);
    let y = x.act(Act::Silu).softmax();
    m.finish([(Role::Logits, &y)])
}

fn plain_softmax_graph(width: usize) -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Rows, 64),
        Dim::fixed(Hidden, width as u64),
    ]);
    let port = Port {
        role: Role::Logits,
        ty: TensorType::new([Rows, Hidden], DType::F32),
    };
    let [x] = m.entry(EntryKind::Forward, [port]);
    let y = x.softmax();
    m.finish([(Role::Logits, &y)])
}

#[test]
fn lane_softmax_parity() {
    let cases = [
        (3usize, 2usize, "lane_softmax"),
        (3, 17, "lane_softmax"),
        (4, 32, "lane_softmax"),
        (4, 128, "lane_softmax"),
        (4, 600, "block_softmax"),
        (2, 1024, "block_softmax"),
        (1, 10000, "block_softmax"),
    ];
    for (rows, width, want_kernel) in cases {
        let g = plain_softmax_graph(width);
        let sizes = [(Rows, rows as u64)];
        let mut noise = Noise(19);
        let inp = fill(&g, &g.entries[0].inputs[0].ty, &sizes, &mut noise).unwrap();
        let want = run(&g, &[], 0, std::slice::from_ref(&inp)).unwrap();
        let dev = Device::guess(&host()).favoring("lane_softmax");
        let plan = lower(&g, 0, &sizes, &dev).unwrap();
        assert!(
            plan.steps.iter().any(|s| s.kernel.name() == want_kernel),
            "rows={rows} width={width}: expected {want_kernel}, got {:?}",
            plan.steps
                .iter()
                .map(|s| s.kernel.name())
                .collect::<Vec<_>>()
        );
        for settings in runs() {
            let mut runner = open(plan.clone(), &settings).unwrap();
            let got = runner.run(&[inp.view()]).unwrap();
            let diff = max_abs_diff(got[0].floats().unwrap(), want[0].floats().unwrap());
            assert!(diff < 1e-5, "rows={rows} width={width} diff={diff:e}");
        }
    }

    // Narrow forms run directly without widen or narrow steps.
    let narrow_cases: [(&'static dyn Stored, &'static dyn Stored, f32); 3] = [
        (BF16_FORM, BF16_FORM, 2e-2),
        (BF16_FORM, F32_FORM, 2e-2),
        (FP8_FORM, F32_FORM, 1e-1),
    ];
    for (act, out_form, tol) in narrow_cases {
        for width in [64usize, 600, 9216] {
            let g = softmax_graph(width);
            let sizes = [(Rows, 2u64)];
            let mut noise = Noise(23);
            let inp = fill(&g, &g.entries[0].inputs[0].ty, &sizes, &mut noise).unwrap();
            let want = run(&g, &[], 0, std::slice::from_ref(&inp)).unwrap();
            let dev = Device::guess(&host())
                .favoring("lane_softmax")
                .pinned(Pins::all(act).activations(act).outputs(out_form));
            let plan = lower(&g, 0, &sizes, &dev).unwrap();
            assert!(
                plan.steps.iter().any(|s| s.rule == "lane_softmax"),
                "expected lane_softmax for {}->{} width={width}",
                act.name(),
                out_form.name()
            );
            assert!(
                !plan.steps.iter().any(|s| s.kernel.name() == "widen"),
                "unexpected widen for {}->{} width={width}",
                act.name(),
                out_form.name()
            );
            for settings in runs() {
                let mut runner = open(plan.clone(), &settings).unwrap();
                let got = runner.run(&[inp.view()]).unwrap();
                let diff = max_abs_diff(got[0].floats().unwrap(), want[0].floats().unwrap());
                assert!(
                    diff < tol,
                    "form {}->{} width={width} diff={diff:e}",
                    act.name(),
                    out_form.name()
                );
            }
        }
    }
}

#[test]
fn lane_softmax_neg_inf() {
    for width in [8usize, 64, 600, 9216] {
        let g = plain_softmax_graph(width);
        let sizes = [(Rows, 2u64)];
        let mut data = vec![f32::NEG_INFINITY; 2 * width];
        data[1] = 2.0;
        data[width - 1] = 3.0;
        data[width + 3] = -5.0;
        data[width + width / 2] = -2.0;
        let inp = Tensor::f32(vec![2, width], data);
        let want = run(&g, &[], 0, std::slice::from_ref(&inp)).unwrap();
        let dev = Device::guess(&host()).favoring("lane_softmax");
        let plan = lower(&g, 0, &sizes, &dev).unwrap();
        for settings in runs() {
            let mut runner = open(plan.clone(), &settings).unwrap();
            let got = runner.run(&[inp.view()]).unwrap();
            let diff = max_abs_diff(got[0].floats().unwrap(), want[0].floats().unwrap());
            assert!(diff < 1e-6, "neg_inf width={width} diff={diff:e}");
        }
    }
}
