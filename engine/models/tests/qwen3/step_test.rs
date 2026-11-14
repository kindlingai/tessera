//! A step entry's ragged rows against whole forward passes.

use gpukit::testing::{Noise, max_abs_diff};
use interp::{Store, Tensor, run, run_kept};
use models::qwen3::{Config, program, stepper};
use trace::Graph;

const SEQS: usize = 2;
const LEN: usize = 7;

fn weights(g: &Graph, noise: &mut Noise) -> Vec<Tensor> {
    let size = |e: &graph::DimExpr| e.poly(&g.dims).unwrap().as_const().unwrap() as usize;
    let fill = |w: &graph::Weight| {
        let shape: Vec<usize> = w.ty.shape.0.iter().map(size).collect();
        let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
        Tensor::f32(shape, data)
    };
    g.weights.iter().map(fill).collect()
}

/// Every sequence's tokens and positions, `[Seqs, Len]` each.
fn whole(tokens: &[u32]) -> [Tensor; 2] {
    let shape = vec![SEQS, LEN];
    let positions = (0..SEQS).flat_map(|_| 0..LEN as u32).collect();
    [
        Tensor::u32(shape.clone(), tokens.to_vec()),
        Tensor::u32(shape, positions),
    ]
}

/// One launch's rows: a slot and a position each.
fn feeds(tokens: &[u32], rows: &[(usize, usize)]) -> [Tensor; 3] {
    let shape = vec![rows.len()];
    let column = |of: &dyn Fn(usize, usize) -> u32| {
        Tensor::u32(
            shape.clone(),
            rows.iter().map(|(s, p)| of(*s, *p)).collect(),
        )
    };
    [
        column(&|s, p| tokens[s * LEN + p]),
        column(&|_, p| p as u32),
        column(&|s, _| s as u32),
    ]
}

fn best(row: &[f32]) -> u32 {
    let top = |a: &(usize, &f32), b: &(usize, &f32)| a.1.total_cmp(b.1);
    row.iter().enumerate().max_by(top).unwrap().0 as u32
}

/// Rows mixing both sequences at different positions, in three launches.
fn launches() -> [Vec<(usize, usize)>; 3] {
    [
        vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
        vec![(1, 2), (0, 3), (1, 3)],
        vec![(1, 4), (1, 5), (1, 6), (0, 4), (0, 5), (0, 6)],
    ]
}

#[test]
fn steps_match_forward() {
    let c = Config::toy(4, 16);
    let (forward, step) = (program(&c), stepper(&c));
    assert_eq!(forward.weights, step.weights);
    assert_eq!(step.kept.len(), 2 * 2, "keys and values per layer");
    let mut noise = Noise(3);
    let weights = weights(&forward, &mut noise);
    let draw = |_| ((noise.draw() + 1.0) * 16.0) as u32;
    let tokens: Vec<u32> = (0..SEQS * LEN).map(draw).collect();
    let want = run(&forward, &weights, 0, &whole(&tokens)).unwrap();
    let want = want[0].floats().unwrap();
    let vocab = want.len() / (SEQS * LEN);
    let mut store = Store::default();
    let mut got = vec![0.0; want.len()];
    for rows in launches() {
        let out = run_kept(&step, &weights, &mut store, 0, &feeds(&tokens, &rows)).unwrap();
        let (ids, logits) = (out[0].indices().unwrap(), out[1].floats().unwrap());
        assert_eq!(out[0].shape, [rows.len()]);
        for (at, row) in logits.chunks(vocab).enumerate() {
            let (s, p) = rows[at];
            assert_eq!(ids[at], best(row) as usize);
            got[(s * LEN + p) * vocab..][..vocab].copy_from_slice(row);
        }
    }
    assert_eq!(store.len(0), LEN);
    let off = max_abs_diff(&got, want);
    assert!(off < 1e-5, "steps drift from the forward pass by {off:e}");
}
