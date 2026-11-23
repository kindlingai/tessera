//! Entries of each row at given indices.

use tessera_codegen::prelude::*;
use tessera_graph::{Op, Select};
use tessera_lower::{Draft, Kernel, T, Term, Tree};

/// One thread per output. Ids past the row read zero.
#[kernel]
fn take(
    x: Buf<F32>,
    ids: Buf<U32>,
    mut out: BufMut<F32>,
    len: u32,
    width: u32,
    picks: u32,
    first: u32,
) {
    if gid() < len {
        let idx = ids[gid() as u64];
        let row = (gid() / picks) as u64 * width as u64;
        out[gid() as u64] = if idx >= first && idx - first < width {
            x[row + (idx - first) as u64]
        } else {
            0.0f32
        };
    }
}

/// `picks` ids per row of `width`.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(take, threads = self.len)]
pub struct Take {
    pub len: usize,
    pub width: usize,
    pub picks: usize,
    pub first: usize,
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Select(Select::Take)) {
        return None;
    }
    let (x, ids) = (node.args[0], node.args[1]);
    let width = *tree.node(x).shape.last()?;
    let len = tree.elements(ids);
    let (first, _) = tree.row_share(x);
    let take = Take {
        len,
        width,
        picks: len / (tree.elements(x) / width).max(1),
        first,
    };
    Some(vec![Draft::new(take, [x, ids], len)])
}

tessera_lower::rule!("take", propose);
