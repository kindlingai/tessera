use crate::cpu_test::{close, eval, model, port};
use crate::*;
use trace::DType::{self, F32, I32, U32};
use trace::DimName::*;
use trace::{EntryKind, Graph, Role, TensorType};

/// One kept layer: rows write `x`, then read their slot.
fn kept() -> Graph {
    let m = model();
    let ports = [
        port([Rows, Heads, HeadDim], F32),
        port([Rows, KvHeads, HeadDim], F32),
        port([Rows], U32),
        port([Rows], U32),
    ];
    let [q, x, slots, positions] = m.entry(EntryKind::Step, ports);
    let rows = m.kept(TensorType::new([Seqs, Context, KvHeads, HeadDim], F32));
    let by = [&slots, &positions];
    let kv = x.append(&rows, by);
    m.finish([(Role::Logits, &m.attend_rows(1.0, [&q, &kv, &kv], by))])
}

/// Zero queries average the rows their slot sees.
fn feeds(rows: &[(u32, u32, f32)]) -> [Tensor; 4] {
    let n = rows.len();
    let mut x = Vec::new();
    for &(_, _, v) in rows {
        x.extend([v, 0.0]);
    }
    [
        Tensor::f32(vec![n, 2, 2], vec![0.0; n * 4]),
        Tensor::f32(vec![n, 1, 2], x),
        Tensor::u32(vec![n], rows.iter().map(|r| r.0).collect()),
        Tensor::u32(vec![n], rows.iter().map(|r| r.1).collect()),
    ]
}

/// Each row's first element, as one row.
fn firsts(out: Result<Vec<Tensor>, Error>) -> Tensor {
    let all = out.unwrap().remove(0);
    let v: Vec<f32> = all.floats().unwrap().iter().step_by(4).copied().collect();
    Tensor::f32(vec![v.len()], v)
}

/// The same row of floats, cast to each of `to`.
fn casts<const N: usize>(x: Tensor, to: [DType; N]) -> Result<Vec<Tensor>, Error> {
    let ports = [port([Rows, Hidden], x.dtype())];
    eval(ports, &[x], None, |_, [x]| to.map(|d| x.cast(d)))
}

#[test]
fn top_k_ties() {
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    // NaN is skipped. 0.0 ranks above -0.0.
    let x = vec![2., 7., 7., -1., -inf, 3., nan, -inf, -0.0, 0.0, -0.0, inf];
    let x = Tensor::f32(vec![3, 4], x);
    let out = eval([port([Rows, Hidden], F32)], &[x], None, |_, [x]| {
        [x.top_k(TopK)]
    });
    // Ties keep the lower index. Spare slots hold the width.
    let want = [1, 2, 0, 1, 4, 4, 3, 1, 0];
    assert_eq!(out.unwrap()[0].indices().unwrap(), want);
}

#[test]
fn casts_truncate() {
    let x = Tensor::f32(vec![1, 4], vec![2.7, -2.7, -0.5, 2e9]);
    let out = casts(x, [I32]).unwrap();
    // Toward zero.
    assert_eq!(out[0].view().as_i32().unwrap(), [2, -2, 0, 2_000_000_000]);
    let x = Tensor::f32(vec![1, 4], vec![2.7, 0.0, -0.5, 3e9]);
    let out = casts(x, [U32]).unwrap();
    assert_eq!(out[0].indices().unwrap(), [2, 0, 0, 3_000_000_000]);
    let ids = Tensor::u32(vec![1, 4], vec![7, 0, 1, u32::MAX]);
    let out = casts(ids.clone(), [F32]).unwrap();
    close(&out[0], &[7., 0., 1., 4_294_967_296.]);
    // Values out of the target's range are refused.
    for (x, to) in [(3e9, I32), (-2.7, U32), (f32::NAN, I32)] {
        let x = Tensor::f32(vec![1, 4], vec![x; 4]);
        assert!(matches!(casts(x, [to]), Err(Error::Mismatch(_))));
    }
    assert!(casts(ids, [I32]).is_err());
}

#[test]
fn store_keeps_rows() {
    let (g, s) = (kept(), &mut Store::default());
    let step = |s: &mut Store, rows: &[_]| firsts(run_kept(&g, &[], s, 0, &feeds(rows)));
    close(
        &step(s, &[(0, 0, 1.), (0, 1, 3.), (1, 0, 10.)]),
        &[1., 2., 10.],
    );
    // Without a store, no row wrote the earlier positions.
    let once = run(&g, &[], 0, &feeds(&[(0, 2, 5.)]));
    assert!(matches!(once, Err(Error::Mismatch(m)) if m.contains("position 0")));
    close(&step(s, &[(0, 2, 5.)]), &[3.]);
    // A launch shorter than the store sees its slot's rows.
    close(&step(s, &[(1, 1, 20.)]), &[15.]);
    // A position past the context bound is refused.
    assert!(run_kept(&g, &[], s, 0, &feeds(&[(0, 4, 1.)])).is_err());
    assert_eq!(s.len(0), 3);
    // A slot past `Seqs` is refused. The store keeps rows.
    assert!(run_kept(&g, &[], s, 0, &feeds(&[(2, 0, 1.)])).is_err());
    // A row past its slot's last written position is refused.
    assert!(run_kept(&g, &[], s, 0, &feeds(&[(1, 3, 1.)])).is_err());
    close(&step(s, &[(0, 3, 7.)]), &[4.]);
}
