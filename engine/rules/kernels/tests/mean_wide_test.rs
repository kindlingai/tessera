//! Parity tests for vectorized mean and mix across float forms.

use tessera_kernels as _;

use tessera_graph::DimName::{Heads, Hidden, Rows};
use tessera_graph::{DType, Dim, TensorType};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM, Load as Stored};
use tessera_lower::{Device, Pins, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, max_abs_diff};
use tessera_trace::{Act, EntryKind, Model, Port, Role};

fn draw(shape: Vec<usize>, noise: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
    Tensor::f32(shape, data)
}

#[test]
fn mean_mix_wide_match() {
    let mut noise = Noise(113);
    let forms: [&'static dyn Stored; 2] = [F32_FORM, BF16_FORM];
    for &(rows, count, inner) in &[(7usize, 8usize, 64usize), (5, 4, 7)] {
        for &weighted in &[false, true] {
            for &form in &forms {
                let model = Model::new(vec![
                    Dim::bounded(Rows, 32),
                    Dim::fixed(Heads, count as u64),
                    Dim::fixed(Hidden, inner as u64),
                ]);
                let x_port = Port {
                    role: Role::Tokens,
                    ty: TensorType::new([Rows, Heads, Hidden], DType::F32),
                };
                let w_port = Port {
                    role: Role::Tokens,
                    ty: TensorType::new([Rows, Heads], DType::F32),
                };
                let (g, inputs) = if weighted {
                    let [x, w] = model.entry(EntryKind::Forward, [x_port, w_port]);
                    let xs = if form.name() == "f32" {
                        x
                    } else {
                        x.act(Act::Silu)
                    };
                    let out = xs.mix(1, &w);
                    let fin = if form.name() == "f32" {
                        out
                    } else {
                        out.act(Act::Silu)
                    };
                    (
                        model.finish([(Role::Logits, &fin)]),
                        vec![
                            draw(vec![rows, count, inner], &mut noise),
                            draw(vec![rows, count], &mut noise),
                        ],
                    )
                } else {
                    let [x] = model.entry(EntryKind::Forward, [x_port]);
                    let xs = if form.name() == "f32" {
                        x
                    } else {
                        x.act(Act::Silu)
                    };
                    let out = xs.mean(1);
                    let fin = if form.name() == "f32" {
                        out
                    } else {
                        out.act(Act::Silu)
                    };
                    (
                        model.finish([(Role::Logits, &fin)]),
                        vec![draw(vec![rows, count, inner], &mut noise)],
                    )
                };
                let want = run(&g, &[], 0, &inputs).unwrap();
                let mut pins = Pins::default();
                if form.name() != "f32" {
                    pins = pins.activations(form);
                }
                let tol = if form.name() == "f32" { 1e-4 } else { 5e-2 };
                for settings in runs() {
                    let device = Device::guess(&settings).pinned(pins.clone());
                    let plan = lower(&g, 0, &[(Rows, rows as u64)], &device).unwrap();
                    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                    let expect = if weighted { "mix" } else { "mean" };
                    assert!(names.contains(&expect), "{names:?}");
                    if form.name() == "bf16" {
                        assert!(!names.contains(&"widen"), "unexpected widen: {names:?}");
                    }
                    let mut runner = open(plan, &settings).unwrap();
                    let views: Vec<_> = inputs.iter().map(Tensor::view).collect();
                    let got = runner.run(&views).unwrap().remove(0);
                    let diff = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                    assert!(
                        diff < tol,
                        "{expect} r={rows} c={count} i={inner} {}: diff={diff:e}",
                        form.name()
                    );
                }
            }
        }
    }
}
