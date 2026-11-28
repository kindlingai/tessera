//! Parity and narrow-form lowering checks for `allow`.

use tessera_kernels as _;

use tessera_graph::DimName::{Rows, Vocab};
use tessera_graph::{DType, Dim, DimExpr, Graph, TensorType};
use tessera_interp::{Tensor, run};
use tessera_lower::form::{BF16_FORM, F32_FORM, Load as Stored};
use tessera_lower::{Device, Pins, lower};
use tessera_run::open;
use tessera_run::testing::runs;
use tessera_testkit::Noise;
use tessera_trace::{EntryKind, Model, Port, Role};

fn allow_graph(vocab: u64, words: u64, edit_words: u64) -> Graph {
    let m = Model::new(vec![Dim::bounded(Rows, 16), Dim::fixed(Vocab, vocab)]);
    let bits_dim = if edit_words == 0 {
        DimExpr::Const(words)
    } else {
        DimExpr::Const(words) + DimExpr::Const(edit_words)
    };
    let [x, bits] = m.entry(
        EntryKind::Forward,
        [
            Port {
                role: Role::Tokens,
                ty: TensorType::new([Rows, Vocab], DType::F32),
            },
            Port {
                role: Role::Positions,
                ty: TensorType::new(vec![Rows.into(), bits_dim], DType::U32),
            },
        ],
    );
    let out = x.scale(1.0).allow(&bits);
    m.finish([(Role::Logits, &out)])
}

fn check_logits(got: &[f32], want: &[f32], tol: f32, label: &str) {
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
fn allow_mask_parity() {
    let mut noise = Noise(149);
    let forms: [(&'static dyn Stored, f32); 2] = [(F32_FORM, 1e-5), (BF16_FORM, 2e-2)];
    for &(vocab, with_edits) in &[(64u64, false), (67u64, false), (64u64, true)] {
        let words = (vocab as usize).div_ceil(32);
        let edit_words = if with_edits { 6 } else { 0 };
        let stride = words + edit_words;
        let g = allow_graph(vocab, words as u64, edit_words as u64);
        let rows = 3usize;
        let x = Tensor::f32(
            vec![rows, vocab as usize],
            (0..rows * vocab as usize).map(|_| noise.draw()).collect(),
        );
        let mut mask = vec![0u32; rows * stride];
        let pats = [0x0000_0001u32, 0xFFFF_FFFFu32, 0xA5A5_5A5Au32];
        for r in 0..rows {
            for w in 0..words {
                mask[r * stride + w] = pats[(r + w) % pats.len()];
            }
            if with_edits {
                let base = r * stride + words;
                mask[base] = 0;
                mask[base + 1] = 17;
                mask[base + 2] = 0.5f32.to_bits();
                mask[base + 3] = (-0.25f32).to_bits();
                mask[base + 4] = 1.2f32.to_bits();
                mask[base + 5] = 0.8f32.to_bits();
            }
        }
        let bits = Tensor::u32(vec![rows, stride], mask);
        let want = run(&g, &[], 0, &[x.clone(), bits.clone()]).unwrap();
        let want = want[0].floats().unwrap();
        for settings in runs() {
            for (form, tol) in forms {
                let dev = Device::guess(&settings).pinned(Pins::default().activations(form));
                let plan = lower(&g, 0, &[(Rows, rows as u64)], &dev).unwrap();
                let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                assert!(names.contains(&"allow"), "{names:?}");
                if settings.gpu().is_some() {
                    assert!(!names.contains(&"widen"), "{form:?}: {names:?}");
                }
                let mut runner = open(plan, &settings).unwrap();
                let got = runner.run(&[x.view(), bits.view()]).unwrap();
                let label = format!("v={vocab} ed={with_edits} {}", form.name());
                check_logits(got[0].floats().unwrap(), want, tol, &label);
            }
        }
    }
}
