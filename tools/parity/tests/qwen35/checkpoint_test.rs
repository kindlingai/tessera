//! A tiny bf16 checkpoint with Clef's weight names.

use super::{LEN, SEQS, matches, part, positions, read, tokens};
use tessera_bundle::Overrides;
use tessera_bundle::qwen35::open;
use tessera_parity::tensor;

#[test]
fn checkpoint_loads() {
    let Some(dir) = part("checkpoint") else {
        return eprintln!("qwen35-tiny: no reference here; skipped");
    };
    let text = open(&dir, [SEQS as u64, LEN as u64], &Overrides::default()).unwrap();
    let g = &text.graph;
    let axis = |e: &tessera_trace::DimExpr| e.value(&g.dims, &|_| None).unwrap() as usize;
    let mut weights = Vec::new();
    text.each(|at, w| {
        let shape = g.weights[at as usize].ty.shape.0.iter().map(axis);
        weights.push(tensor(shape.collect(), w));
        Ok(())
    })
    .unwrap();
    let inputs = [tokens(&dir), positions()];
    matches(g, &weights, &text.pins, &inputs, &read(&dir, "logits"));
}
