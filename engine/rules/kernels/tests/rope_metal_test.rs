//! CPU parity tests for single-launch Metal RoPE.

use tessera_kernels as _;

use tessera_graph::DimName::{HeadDim, Heads, Rows};
use tessera_graph::{DType, Dim, Pairs, TensorType};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM, Load as Stored};
use tessera_lower::{Backend, Device, Pins, Settings, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, max_abs_diff};
use tessera_trace::{Act, EntryKind, Model, Port, Role};

fn metal_settings() -> Option<Settings> {
    runs().into_iter().find(|s| s.gpu() == Some(Backend::Metal))
}

#[test]
fn rope_metal_matches_cpu() {
    let Some(settings) = metal_settings() else {
        return;
    };
    let mut noise = Noise(83);
    type Case = (
        usize,
        usize,
        usize,
        Option<u32>,
        Pairs,
        &'static dyn Stored,
        f32,
    );
    let cases: &[Case] = &[
        (1, 32, 128, None, Pairs::Halves, F32_FORM, 5e-4),
        (1, 32, 256, Some(32), Pairs::Halves, F32_FORM, 5e-4),
        (4, 16, 128, None, Pairs::Adjacent, F32_FORM, 5e-4),
        (33, 24, 64, Some(16), Pairs::Adjacent, F32_FORM, 1e-3),
        (16, 32, 256, Some(32), Pairs::Halves, BF16_FORM, 2e-2),
        (4, 8, 128, None, Pairs::Adjacent, BF16_FORM, 2e-2),
    ];

    for &(rows, heads, dim, turned, pairs, form, tol) in cases {
        let model = Model::new(vec![
            Dim::bounded(Rows, 128),
            Dim::fixed(Heads, heads as u64),
            Dim::fixed(HeadDim, dim as u64),
        ]);
        let [x, pos] = model.entry(
            EntryKind::Forward,
            [
                Port {
                    role: Role::Tokens,
                    ty: TensorType::new([Rows, Heads, HeadDim], DType::F32),
                },
                Port {
                    role: Role::Positions,
                    ty: TensorType::new([Rows], DType::U32),
                },
            ],
        );
        let src = if form.name() == "f32" {
            x
        } else {
            x.act(Act::Silu)
        };
        let rotated = match turned {
            None => src.rope(&pos, 10_000.0, pairs),
            Some(t) => src.rope_part(&pos, 10_000.0, pairs, t),
        };
        let out = if form.name() == "f32" {
            rotated
        } else {
            rotated.act(Act::Silu)
        };
        let g = model.finish([(Role::Logits, &out)]);

        let x_raw: Vec<f32> = (0..rows * heads * dim).map(|_| noise.draw()).collect();
        let x_in = Tensor::f32(vec![rows, heads, dim], form.roundtrip(&x_raw));
        let pos_vals: Vec<u32> = (0..rows).map(|r| (r as u32) * 13 + 5).collect();
        let pos_in = Tensor::u32(vec![rows], pos_vals);
        let want = run(&g, &[], 0, &[x_in.clone(), pos_in.clone()]).unwrap();

        let mut pins = Pins::default();
        if form.name() != "f32" {
            pins = pins.activations(form);
        }
        let device = Device::guess(&settings).pinned(pins);
        let plan = lower(&g, 0, &[(Rows, rows as u64)], &device).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        let want_name = form.suffixed(if pairs == Pairs::Adjacent {
            "rope_metal_adj"
        } else {
            "rope_metal"
        });
        assert!(
            names.contains(&want_name),
            "expected {want_name} in {names:?}"
        );

        let mut runner = open(plan, &settings).unwrap();
        let got = runner.run(&[x_in.view(), pos_in.view()]).unwrap().remove(0);
        let diff = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
        assert!(
            diff < tol,
            "r={rows} h={heads} d={dim} {pairs:?} {turned:?} {}: diff={diff:e}",
            form.name()
        );
    }
}
