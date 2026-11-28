//! Parity and narrow-form lowering checks for `mask` and `causal`.

use tessera_kernels as _;

use tessera_graph::DimName::{Context, Heads, Rows};
use tessera_graph::{DType, Dim, Graph, Mask, TensorType};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM, Load as Stored};
use tessera_lower::{Device, Pins, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::Noise;
use tessera_trace::{EntryKind, Model, Port, Role};

fn mask_graph(heads: u64, queries: u64, keys: u64, kind: Mask) -> Graph {
    let m = Model::new(vec![
        Dim::fixed(Heads, heads),
        Dim::bounded(Rows, queries),
        Dim::fixed(Context, keys),
    ]);
    let [x] = m.entry(
        EntryKind::Forward,
        [Port {
            role: Role::Tokens,
            ty: TensorType::new([Heads, Rows, Context], DType::F32),
        }],
    );
    let masked = x.scale(1.0).mask(kind);
    m.finish([(Role::Logits, &masked)])
}

fn causal_graph(heads: u64, rows: u64, width: u64) -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Rows, rows),
        Dim::fixed(Heads, heads),
        Dim::fixed(Context, width),
    ]);
    let [x, pos] = m.entry(
        EntryKind::Forward,
        [
            Port {
                role: Role::Tokens,
                ty: TensorType::new([Rows, Heads, Context], DType::F32),
            },
            Port {
                role: Role::Positions,
                ty: TensorType::new([Rows], DType::U32),
            },
        ],
    );
    let out = x.scale(1.0).causal(&pos);
    m.finish([(Role::Logits, &out)])
}

fn check_masked_slice(got: &[f32], want: &[f32], tol: f32, label: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if w.is_infinite() {
            assert_eq!(g, w, "{label} [{i}]: got={g} want={w}");
        } else {
            assert!((g - w).abs() < tol, "{label} [{i}]: got={g} want={w}");
        }
    }
}

#[test]
fn mask_wide_narrow_parity() {
    let mut noise = Noise(91);
    let kinds = [Mask::Causal, Mask::Back(5), Mask::Within(3)];
    let forms: [(&'static dyn Stored, f32); 2] = [(F32_FORM, 1e-5), (BF16_FORM, 2e-2)];
    for &keys in &[16u64, 19u64] {
        for kind in kinds {
            let g = mask_graph(2, 6, keys, kind);
            let x = Tensor::f32(
                vec![2, 6, keys as usize],
                (0..2 * 6 * keys as usize).map(|_| noise.draw()).collect(),
            );
            let want = run(&g, &[], 0, std::slice::from_ref(&x)).unwrap();
            let want = want[0].floats().unwrap();
            for settings in runs() {
                for (form, tol) in forms {
                    let pins = Pins::default().activations(form);
                    let dev = Device::guess(&settings).pinned(pins);
                    let plan = lower(&g, 0, &[(Rows, 6)], &dev).unwrap();
                    let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                    assert!(names.contains(&"mask"), "{names:?}");
                    if settings.gpu().is_some() {
                        assert!(!names.contains(&"widen"), "{form:?}: {names:?}");
                    }
                    let mut runner = open(plan, &settings).unwrap();
                    let got = runner.run(&[x.view()]).unwrap();
                    let label = format!(
                        "mask keys={keys} {kind:?} {} {:?}",
                        form.name(),
                        settings.backend
                    );
                    check_masked_slice(got[0].floats().unwrap(), want, tol, &label);
                }
            }
        }
    }
}

#[test]
fn causal_wide_narrow_parity() {
    let mut noise = Noise(103);
    let forms: [(&'static dyn Stored, f32); 2] = [(F32_FORM, 1e-5), (BF16_FORM, 2e-2)];
    for &width in &[24u64, 21u64] {
        let g = causal_graph(3, 5, width);
        let x = Tensor::f32(
            vec![5, 3, width as usize],
            (0..5 * 3 * width as usize).map(|_| noise.draw()).collect(),
        );
        let pos = Tensor::u32(vec![5], vec![0, 3, 7, 15, width as u32]);
        let want = run(&g, &[], 0, &[x.clone(), pos.clone()]).unwrap();
        let want = want[0].floats().unwrap();
        for settings in runs() {
            for (form, tol) in forms {
                let pins = Pins::default().activations(form);
                let dev = Device::guess(&settings).pinned(pins);
                let plan = lower(&g, 0, &[(Rows, 5)], &dev).unwrap();
                let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                assert!(names.contains(&"causal"), "{names:?}");
                if settings.gpu().is_some() {
                    assert!(!names.contains(&"widen"), "{form:?}: {names:?}");
                }
                let mut runner = open(plan, &settings).unwrap();
                let got = runner.run(&[x.view(), pos.view()]).unwrap();
                let label = format!(
                    "causal width={width} {} {:?}",
                    form.name(),
                    settings.backend
                );
                check_masked_slice(got[0].floats().unwrap(), want, tol, &label);
            }
        }
    }
}
