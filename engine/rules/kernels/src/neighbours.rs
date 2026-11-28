//! A 3x3 convolution: gathered neighbours and a matmul, fused.

use tessera_codegen::prelude::*;
use tessera_graph::Op;
use tessera_lower::sites::Near;
use tessera_lower::{Draft, Kernel, Load, T, Term, Tree};

/// Pixels a 3x3 kernel reads.
const TAPS: u32 = 9;

/// One thread per output channel. Missing pixels read zero.
#[kernel]
fn neighbours(
    x: Buf<F32>,
    ids: Buf<U32>,
    w: Buf<F32>,
    mut out: BufMut<F32>,
    len: u32,
    pixels: u32,
    cin: u32,
    cout: u32,
) {
    if gid() >= len {
        return;
    }
    let co = gid() % cout;
    let row = gid() / cout;
    let p = row % pixels;
    let first = (row - p) * cin;
    let mut sum = 0.0f32;
    for t in 0..TAPS {
        let id = ids[(p * TAPS + t) as u64];
        if id < pixels {
            let at = first + id * cin;
            let wat = (co * TAPS + t) * cin;
            for c in 0..cin {
                sum += x[(at + c) as u64] * w[(wat + c) as u64];
            }
        }
    }
    out[gid() as u64] = sum;
}

/// `[Seqs, pixels, cin]` through `TAPS` neighbours to `cout` channels.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    neighbours,
    threads = self.len,
    work = self.len * TAPS as usize * self.cin,
    // Each output reads its neighbours and its weights afresh.
    load = Some(Load {
        threads: self.len,
        group: 256,
        trips: (TAPS as usize * self.cin) as f64,
        trip: body.trip(),
        reads: vec![4.0 * (self.len * TAPS as usize * self.cin) as f64; 2],
        written: 4.0 * len as f64,
        flight: 4.0,
        ..Load::default()
    }),
    load_key = tessera_lower::load_key((self.len, self.pixels, self.cin, self.cout)),
)]
pub(crate) struct Neighbours {
    pub len: usize,
    pub pixels: usize,
    pub cin: usize,
    pub cout: usize,
}

/// Matmuls of each pixel's 3x3 neighbours, gathered by id.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (flat, w) = (node.args[0], node.args[1]);
    let near: Near = tree.site(flat)?;
    let plain = tree.node(w).form.dtype().is_float();
    if !(near.taps == TAPS as usize && plain) {
        return None;
    }
    let len = tree.elements(at);
    let cout = len * near.cin / tree.elements(near.src);
    let k = Neighbours {
        len,
        pixels: near.pixels,
        cin: near.cin,
        cout,
    };
    Some(vec![Draft::new(k, [near.src, near.ids, w], len)])
}

tessera_lower::rule!("neighbours", propose);
