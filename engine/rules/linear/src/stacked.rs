//! Products with blocks of a stored matrix, read in place.

use sensai_codegen::prelude::*;
use sensai_graph::{Op, View};
use sensai_lower::form::Load as Stored;
use sensai_lower::{Draft, Kernel, T, Term, Tree};

/// One thread per output. Batch `b` reads block `b`.
#[kernel]
fn stacked(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    mut out: BufMut<F32>,
    len: u32,
    rows: u32,
    n: u32,
    k: u32,
    block: u32,
    first: u32,
    col_step: u32,
    k_step: u32,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= len {
        return;
    }
    let col = gid() % n;
    let row = gid() / n;
    let at = (row / rows) as u64 * block as u64 + first as u64 + col as u64 * col_step as u64;
    let from = row as u64 * k as u64;
    let mut sum = 0.0f32;
    for i in 0..k {
        sum += x[from + i as u64] * w[at + i as u64 * k_step as u64];
    }
    out[gid() as u64] = sum;
}

/// `x[b, rows, k]` times block `b` of a stored matrix.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    stacked(self.form),
    name = self.form.suffixed("stacked"),
    threads = self.len,
    work = self.len * self.k,
    accepts = stacked_accepts,
)]
pub(crate) struct Stacked {
    pub len: usize,
    pub rows: usize,
    pub n: usize,
    pub k: usize,
    /// Elements between blocks, and the first one read.
    pub block: usize,
    pub first: usize,
    /// Elements between columns, and along `k`.
    pub col_step: usize,
    pub k_step: usize,
    pub form: &'static dyn Stored,
}

fn stacked_accepts(k: &Stacked, input: usize, form: &dyn Stored) -> bool {
    k.form.taken(input, 1, form)
}

/// A stored `[b*r, c]` matrix as `[b, r, c]` blocks.
struct Blocks {
    leaf: T,
    /// Rows per block, the first row read, the row width.
    rows: usize,
    first: usize,
    width: usize,
    swapped: bool,
}

fn blocks(tree: &Tree, mut w: T) -> Option<Blocks> {
    let term = |t: T| &tree.node(t).term;
    let swapped = matches!(term(w), Term::Op(Op::View(View::Permute(p))) if p[..] == [0, 2, 1]);
    if swapped {
        w = tree.node(w).args[0];
    }
    let mut first = 0;
    if let Term::Op(Op::View(View::Slice { axis: 1, start, .. })) = term(w) {
        first = start.value(&tree.g.dims, &|n| tree.size(n))? as usize;
        w = tree.node(w).args[0];
    }
    let Term::Op(Op::View(View::Reshape(_))) = term(w) else {
        return None;
    };
    let leaf = tree.node(w).args[0];
    let (&[_, rows, width], &[_, stored]) = (&tree.node(w).shape[..], &tree.node(leaf).shape[..])
    else {
        return None;
    };
    (tree.leaf(leaf) && stored == width).then_some(Blocks {
        leaf,
        rows,
        first,
        width,
        swapped,
    })
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (x, w) = (node.args[0], node.args[1]);
    let b = blocks(tree, w)?;
    let (&[batches, n, k], xs) = (&tree.node(w).shape[..], &tree.node(x).shape[..]) else {
        return None;
    };
    if xs.len() != 3 || xs[0] != batches || xs[2] != k {
        return None;
    }
    let (col_step, k_step) = match b.swapped {
        true => (1, b.width),
        false => (b.width, 1),
    };
    let kernel = Stacked {
        len: tree.elements(at),
        rows: xs[1],
        n,
        k,
        block: b.rows * b.width,
        first: b.first * b.width,
        col_step,
        k_step,
        form: tree.node(b.leaf).form,
    };
    Some(vec![Draft::new(kernel, [x, b.leaf], kernel.len)])
}

sensai_lower::rule!("stacked", propose);
