//! Products of two computed values, batch by batch.

use tessera_codegen::prelude::*;
use tessera_graph::Op;
use tessera_lower::{Draft, Kernel, T, Term, Tree};

/// One thread per output. Batches of `w` serve rows.
#[kernel]
fn batched(
    x: Buf<F32>,
    w: Buf<F32>,
    mut out: BufMut<F32>,
    len: u32,
    n: u32,
    k: u32,
    x_rows: u32,
    batches: u32,
) {
    if gid() >= len {
        return;
    }
    let row = gid() / n;
    let rows = len / n;
    let x_at = (row / (rows / x_rows)) as u64 * k as u64;
    let w_row = (row / (rows / batches)) * n + gid() % n;
    out[gid() as u64] = x.dot(x_at, w, w_row as u64 * k as u64, k);
}

/// `x[.., k]` times `w[batches, n, k]` transposed.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(batched, threads = self.len, work = self.len * self.k)]
pub struct Batched {
    pub len: usize,
    pub n: usize,
    pub k: usize,
    pub x_rows: usize,
    pub batches: usize,
}

/// A matmul by a batch of matrices, with its reads.
pub fn site(tree: &Tree, at: T) -> Option<(Batched, [T; 2])> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (x, w) = (node.args[0], node.args[1]);
    let floats = |t: T| tree.node(t).form.dtype().is_float();
    let shape = &tree.node(w).shape;
    // Gathers of kept rows are ours, not picked weights.
    let wn = tree.node(w);
    let picked = matches!(wn.term, Term::Op(Op::Select(_))) && !tree.appends(wn.args[0]);
    if picked || shape.len() < 3 || !floats(x) || !floats(w) {
        return None;
    }
    let [.., n, k] = shape[..] else {
        return None;
    };
    let kernel = Batched {
        len: tree.elements(at),
        n,
        k,
        x_rows: tree.elements(x) / k,
        batches: tree.elements(w) / (n * k),
    };
    Some((kernel, [x, w]))
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (kernel, reads) = site(tree, at)?;
    Some(vec![Draft::new(kernel, reads, kernel.len)])
}

tessera_lower::rule!("batched", propose);
