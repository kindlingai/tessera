//! The causal filter per channel before linear attention.

use tessera_codegen::prelude::*;
use tessera_graph::Op;
use tessera_lower::{Draft, Kernel, T, Term, Tree};

/// One thread per element: its channel's last `taps` rows.
#[kernel]
fn causal_conv(
    x: Buf<F32>,
    w: Buf<F32>,
    mut out: BufMut<F32>,
    len: u32,
    rows: u32,
    width: u32,
    taps: u32,
) {
    if gid() >= len {
        return;
    }
    let c = gid() % width;
    let t = gid() / width % rows;
    let mut sum = 0.0f32;
    for j in 0..taps {
        // Rows before the sequence's first read as zero.
        if t + j + 1 >= taps {
            let back = taps - 1 - j;
            sum += w[(c * taps + j) as u64] * x[(gid() - back * width) as u64];
        }
    }
    out[gid() as u64] = sum;
}

/// Whole sequences of `rows`, each `width` channels.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(causal_conv, name = "conv", threads = self.len)]
pub(crate) struct Conv {
    len: usize,
    rows: usize,
    width: usize,
    taps: usize,
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Conv) {
        return None;
    }
    let [x, w] = node.args[..] else {
        return None;
    };
    if tree.node(x).extents[1].runtime() {
        return None;
    }
    let [_, rows, width] = node.shape[..] else {
        return None;
    };
    let len = tree.elements(at);
    let taps = tree.node(w).shape[1];
    let conv = Conv {
        len,
        rows,
        width,
        taps,
    };
    Some(vec![Draft::new(conv, [x, w], len)])
}

tessera_lower::rule!("conv", propose);
