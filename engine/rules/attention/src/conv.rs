//! The causal filter per channel before linear attention.

use crate::slots::slot_scan;
use tessera_codegen::prelude::*;
use tessera_graph::Op;
use tessera_lower::form::{Load as Stored, plain};
use tessera_lower::{
    Arg, Backend, Draft, Extent, GUESS, Keeping, Kernel, KernelRef, Launch, Needs, Read, Shape,
    Step, T, Term, Tree, source,
};

/// One thread per row suits Metal's many small cores.
const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];

/// Fewer rows scan as fast per slot.
const SPAN_ROWS: usize = 32;

/// Largest history depth held in thread registers.
const MAX_HIST: usize = 16;

/// One thread per element: its channel's last `taps` rows.
#[kernel]
fn causal_conv<const DYNAMIC: bool>(
    x: Buf<F32>,
    #[form(w_form)] w: Buf<Dyn>,
    mut out: BufMut<F32>,
    len: u32,
    rows: u32,
    width: u32,
    taps: u32,
    dilation: u32,
    #[build] w_form: &'static dyn Stored,
) {
    if gid() >= len {
        return;
    }
    let c = gid() % width;
    let t = gid() / width % rows;
    // Dynamic taps differ at every row.
    let wc = if DYNAMIC { gid() } else { c };
    let mut sum = 0.0f32;
    for j in 0..taps {
        let back = (taps - 1 - j) * dilation;
        if t >= back {
            sum += w[(wc * taps + j) as u64] * x[(gid() - back * width) as u64];
        }
    }
    out[gid() as u64] = sum;
}

slot_scan!(
    /// One thread per `(slot, channel)`: updates kept history per row.
    fn causal_conv_rows(
        x: Buf<F32>,
        #[form(w_form)] w: Buf<Dyn>,
        slots: Buf<U32>,
        positions: Buf<U32>,
        mut state: BufMut<F32>,
        mut out: BufMut<F32>,
        mut keep: BufMut<F32>,
        jobs: u32,
        rows: u32,
        #[build] width: u32,
        #[build] taps: u32,
        #[build] dilation: u32,
        #[build] dynamic: u32,
        #[build] w_form: &'static dyn Stored,
    ),
    [slots, positions, out, rows],
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
        if dynamic == 0 {
            for j in 0..taps {
                taps_w[j as usize] = w[(c * taps + j) as u64];
            }
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
            keep[base + (k * width) as u64] = hist[k as usize];
        }
    },
    row: {
        let at = (r * width + c) as u64;
    },
    step: {
        if dynamic != 0 {
            for j in 0..taps {
                taps_w[j as usize] = w[at * taps as u64 + j as u64];
            }
        }
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

/// A channel's inputs by lag. Lag 0 is its own.
#[derive(Group)]
struct Lags {
    val: [f32; 17],
    /// The lag the next stepped row fills.
    m: u32,
    /// The row whose taps weigh the lags.
    taps_row: u32,
}

/// Fills older lags from kept state, then writes.
#[kernel]
fn close_span(
    lg: &mut Lags,
    #[form(w_form)] w: Buf<Dyn>,
    state: BufMut<F32>,
    mut out: BufMut<F32>,
    mut keep: BufMut<F32>,
    phase: u32,
    fresh: u32,
    s: u32,
    #[build] width: u32,
    #[build] taps: u32,
    #[build] dilation: u32,
    #[build] dynamic: u32,
    #[build] w_form: &'static dyn Stored,
) {
    let c = gid() % width;
    let hist_len = (taps - 1) * dilation;
    let base = (s * hist_len) as u64 * width as u64 + c as u64;
    if fresh == 0 {
        for l in (lg.m + 1)..(hist_len + 1) {
            lg.val[l as usize] = state[base + ((hist_len + lg.m - l) * width) as u64];
        }
    }
    if phase == 0 {
        let mut wc = c;
        if dynamic != 0 {
            wc = lg.taps_row * width + c;
        }
        let mut sum = 0.0f32;
        for j in 0..taps {
            sum += w[wc as u64 * taps as u64 + j as u64]
                * lg.val[((taps - 1 - j) * dilation) as usize];
        }
        out[gid() as u64] = sum;
    } else {
        for k in 0..hist_len {
            keep[base + (k * width) as u64] = lg.val[(hist_len - k) as usize];
        }
    }
}

/// Phase 0: a thread per output. Phase 1: state.
#[kernel]
fn causal_conv_span(
    x: Buf<F32>,
    #[form(w_form)] w: Buf<Dyn>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    state: BufMut<F32>,
    out: BufMut<F32>,
    keep: BufMut<F32>,
    jobs: u32,
    rows: u32,
    phase: u32,
    #[build] width: u32,
    #[build] taps: u32,
    #[build] dilation: u32,
    #[build] dynamic: u32,
    #[build] w_form: &'static dyn Stored,
) {
    if gid() >= jobs || (phase == 0 && gid() / width >= rows) {
        return;
    }
    let c = gid() % width;
    let hist_len = (taps - 1) * dilation;
    // An output walks back from its row. State, the end.
    let mut s = gid() / width;
    let mut top = rows;
    if phase == 0 {
        s = slots[(gid() / width) as u64];
        top = gid() / width + 1;
    }
    let mut lg = Lags {
        val: [0.0f32; 17],
        m: phase,
        taps_row: 0,
    };
    for l in 0..(hist_len + 1) {
        lg.val[l as usize] = 0.0;
    }
    let mut have = 0u32;
    let mut cand = 0u32;
    let mut cpos = 0u32;
    for back in 0..top {
        let i = top - 1 - back;
        if slots[i as u64] == s {
            let p = positions[i as u64];
            if have == 0 {
                have = 1;
                cand = i;
                cpos = p;
            } else if p == cpos {
                // The later row was a filler of this one.
                cand = i;
            } else {
                lg.val[lg.m as usize] = x[(cand * width + c) as u64];
                if lg.m == 0 {
                    lg.taps_row = cand;
                }
                // Position 0 starts afresh. Nothing older counts.
                if cpos == 0 || lg.m == hist_len {
                    close_span(
                        &mut lg, w, state, out, keep, phase, 1, s, width, taps, dilation, dynamic,
                        w_form,
                    );
                    return;
                }
                lg.m += 1;
                cand = i;
                cpos = p;
            }
        }
    }
    if have == 1 {
        lg.val[lg.m as usize] = x[(cand * width + c) as u64];
        if lg.m == 0 {
            lg.taps_row = cand;
        }
        let fresh = if cpos == 0 { 1 } else { 0 };
        close_span(
            &mut lg, w, state, out, keep, phase, fresh, s, width, taps, dilation, dynamic, w_form,
        );
    }
}

/// Whole sequences of `rows`, each `width` channels.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    causal_conv(self.w_form),
    variant = self.dynamic,
    name = self.w_form.suffixed(match self.dynamic {
        true => "conv_dynamic",
        false => "conv",
    }),
    threads = self.len,
    accepts = conv_accepts,
)]
pub(crate) struct Conv {
    len: Extent,
    rows: Extent,
    width: usize,
    taps: usize,
    dilation: usize,
    /// Whether every row has its own taps.
    dynamic: bool,
    w_form: &'static dyn Stored,
}

fn conv_accepts(k: &Conv, input: usize, form: &dyn Stored) -> bool {
    k.w_form.taken(input, 1, form)
}

/// Ragged rows updating fixed per-slot history.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = tessera_lower,
    name = self.w_form.suffixed(self.entry()),
    sizes = vec![
        self.seqs,
        self.rows.most(),
        self.width,
        self.taps,
        self.dilation,
        usize::from(self.dynamic),
    ],
    work = self.rows.most() * self.width * self.taps * self.passes(),
    guess = GUESS,
    needs = rows_needs(self),
    accepts = rows_accepts,
    source = source(self.w_form.suffixed(self.entry()), self.body()),
    launches = rows_launches,
    commit = rows_commit,
)]
pub(crate) struct ConvRows {
    seqs: usize,
    rows: Extent,
    width: usize,
    taps: usize,
    dilation: usize,
    dynamic: bool,
    keeping: Keeping,
    /// A thread per row and channel, not per slot.
    span: bool,
    w_form: &'static dyn Stored,
}

impl ConvRows {
    fn entry(&self) -> &'static str {
        match self.span {
            true => "conv_span",
            false => "conv_rows",
        }
    }

    /// A scan reads and shifts. A span only reads.
    fn passes(&self) -> usize {
        2 - usize::from(self.span)
    }

    fn body(&self) -> tessera_codegen::ir::Body {
        let (width, taps) = (self.width as u32, self.taps as u32);
        let (dilation, dynamic) = (self.dilation as u32, u32::from(self.dynamic));
        match self.span {
            true => causal_conv_span_ir(width, taps, dilation, dynamic, self.w_form),
            false => causal_conv_rows_ir(width, taps, dilation, dynamic, self.w_form),
        }
    }
}

fn rows_needs(k: &ConvRows) -> Needs {
    Needs {
        lacks: if k.span { LACKS } else { &[] },
        ..Needs::default()
    }
}

fn rows_accepts(k: &ConvRows, input: usize, form: &dyn Stored) -> bool {
    k.w_form.taken(input, 1, form)
}

fn rows_launches(k: &ConvRows, step: &Step) -> Vec<Launch> {
    let jobs = k.seqs * k.width;
    let mut launch = Launch::plain(step, &[jobs], Shape::Line(jobs));
    let at = launch.args.len() - 1;
    launch.args.insert(at, k.keeping.keep(step));
    launch.args.push(Arg::size(k.keeping.rows(k.rows)));
    if !k.span {
        return vec![launch];
    }
    // Every output first: the state launch may overwrite its reads.
    let all = k.rows.most() * k.width;
    let mut outputs = launch.clone();
    outputs.args[at + 1] = Arg::count(all);
    outputs.shape = Shape::Line(all);
    outputs.args.push(Arg::count(0));
    launch.args.push(Arg::count(1));
    vec![outputs, launch]
}

fn rows_commit(k: &ConvRows) -> Option<KernelRef> {
    k.keeping.twin(|keeping| ConvRows { keeping, ..*k })
}

/// Long launches: every row and channel at once.
fn propose_span(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let Term::Op(Op::Conv(c)) = tree.node(at).term else {
        return None;
    };
    c.state?;
    let rows = tree.node(tree.node(at).args[0]).extents[0].most();
    if rows < SPAN_ROWS {
        return None;
    }
    rows_drafts(tree, at, true)
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    rows_drafts(tree, at, false)
}

/// The filter at `at`. A span reads rows in parallel.
fn rows_drafts(tree: &Tree, at: T, span: bool) -> Option<Vec<Draft>> {
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
        let taps = *tree.node(w).shape.last()?;
        let dynamic = tree.node(w).shape.len() > 2;
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
            dynamic,
            keeping: Keeping::of(tree),
            span,
            w_form: plain(tree.node(w).form),
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
    let taps = *tree.node(w).shape.last()?;
    let conv = Conv {
        len,
        rows,
        width,
        taps,
        dilation,
        dynamic: tree.node(w).shape.len() > 2,
        w_form: plain(tree.node(w).form),
    };
    Some(vec![Draft::new(conv, [x, w], len.most())])
}

tessera_lower::rule!("conv", propose);
tessera_lower::rule!("conv_span", propose_span);
