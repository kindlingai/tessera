//! The causal filter per channel before linear attention.

use crate::slots::slot_scan;
use tessera_codegen::prelude::*;
use tessera_graph::Op;
use tessera_lower::{
    Arg, Draft, Extent, GUESS, Keeping, Kernel, KernelRef, Launch, Read, Shape, Step, T, Term,
    Tree, source,
};

/// Largest history depth held in thread registers.
const MAX_HIST: usize = 16;

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
    dilation: u32,
) {
    if gid() >= len {
        return;
    }
    let c = gid() % width;
    let t = gid() / width % rows;
    let mut sum = 0.0f32;
    for j in 0..taps {
        let back = (taps - 1 - j) * dilation;
        if t >= back {
            sum += w[(c * taps + j) as u64] * x[(gid() - back * width) as u64];
        }
    }
    out[gid() as u64] = sum;
}

slot_scan!(
    /// One thread per `(slot, channel)`: updates kept history per row.
    fn causal_conv_rows(
        x: Buf<F32>,
        w: Buf<F32>,
        slots: Buf<U32>,
        positions: Buf<U32>,
        mut state: BufMut<F32>,
        mut out: BufMut<F32>,
        jobs: u32,
        rows: u32,
        #[build] width: u32,
        #[build] taps: u32,
        #[build] dilation: u32,
        #[build] peek: u32,
    ),
    [slots, positions, out, rows, peek],
    [s, r, pos, at, last],
    setup: {
        if gid() >= jobs {
            return;
        }
        let s = gid() / width;
        let c = gid() % width;
        let hist_len = (taps - 1) * dilation;
        let mut hist = [0.0f32; 16];
        let mut taps_w = [0.0f32; 16];
        let base = (s * hist_len) as u64 * width as u64 + c as u64;
    },
    open: {
        for j in 0..taps {
            taps_w[j as usize] = w[(c * taps + j) as u64];
        }
    },
    zero: {
        for k in 0..hist_len {
            hist[k as usize] = 0.0;
        }
    },
    load: {
        for k in 0..hist_len {
            hist[k as usize] = state[base + (k * width) as u64];
        }
    },
    store: {
        for k in 0..hist_len {
            state[base + (k * width) as u64] = hist[k as usize];
        }
    },
    row: {
        let at = (r * width + c) as u64;
    },
    step: {
        let cur = x[at];
        let mut sum = cur * taps_w[(taps - 1) as usize];
        for j in 0..(taps - 1) {
            sum += hist[(j * dilation) as usize] * taps_w[j as usize];
        }
        for k in 0..(hist_len - 1) {
            hist[k as usize] = hist[(k + 1) as usize];
        }
        hist[(hist_len - 1) as usize] = cur;
        last = sum;
    },
);

/// Whole sequences of `rows`, each `width` channels.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(causal_conv, name = "conv", threads = self.len)]
pub(crate) struct Conv {
    len: Extent,
    rows: Extent,
    width: usize,
    taps: usize,
    dilation: usize,
}

/// Ragged rows updating fixed per-slot history.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = tessera_lower,
    name = "conv_rows",
    sizes = vec![self.seqs, self.rows.most(), self.width, self.taps, self.dilation],
    work = self.rows.most() * self.width * self.taps * 2,
    guess = GUESS,
    source = source(
        "conv_rows",
        causal_conv_rows_ir(
            self.width as u32,
            self.taps as u32,
            self.dilation as u32,
            self.keeping.peeks(),
        )
    ),
    launches = rows_launches,
    commit = rows_commit,
)]
pub(crate) struct ConvRows {
    seqs: usize,
    rows: Extent,
    width: usize,
    taps: usize,
    dilation: usize,
    keeping: Keeping,
}

fn rows_launches(k: &ConvRows, step: &Step) -> Vec<Launch> {
    let jobs = k.seqs * k.width;
    let mut launch = Launch::plain(step, &[jobs], Shape::Line(jobs));
    launch.args.push(Arg::size(k.keeping.rows(k.rows)));
    vec![launch]
}

fn rows_commit(k: &ConvRows) -> Option<KernelRef> {
    k.keeping.twin(|keeping| ConvRows { keeping, ..*k })
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Op(Op::Conv(c)) = node.term else {
        return None;
    };
    let dilation = c.dilation as usize;
    if let Some(idx) = c.state {
        let [x, w, slots, positions] = node.args[..] else {
            return None;
        };
        let rows = tree.node(x).extents[0];
        let width = *node.shape.last()?;
        let taps = tree.node(w).shape[1];
        let hist = taps.checked_sub(1)? * dilation;
        if hist == 0 || hist > MAX_HIST || taps > MAX_HIST {
            return None;
        }
        let seqs = *tree.fixed_kept(idx)?.first()?;
        let conv = ConvRows {
            seqs,
            rows,
            width,
            taps,
            dilation,
            keeping: Keeping::of(tree),
        };
        let reads = [
            Read::Node(x),
            Read::Node(w),
            Read::Node(slots),
            Read::Node(positions),
            Read::Kept(idx),
        ];
        return Some(vec![Draft::new(conv, reads, rows.most() * width)]);
    }
    let [x, w] = node.args[..] else {
        return None;
    };
    // An extent holds one runtime dimension: the rows.
    let [seqs, rows, _] = tree.node(x).extents[..] else {
        return None;
    };
    if seqs.runtime() {
        return None;
    }
    let width = *node.shape.last()?;
    let len = rows.times(seqs.most() * width);
    let taps = tree.node(w).shape[1];
    let conv = Conv {
        len,
        rows,
        width,
        taps,
        dilation,
    };
    Some(vec![Draft::new(conv, [x, w], len.most())])
}

tessera_lower::rule!("conv", propose);
