//! Attention on tiles, the keys streamed in blocks.

use sensai_codegen::ir::Body;
use sensai_codegen::kept;
use sensai_codegen::prelude::*;
use sensai_graph::{Mask, Order};
use sensai_lower::sites;
use sensai_lower::{
    Arg, Attn, Draft, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Tree,
};
use std::cell::RefCell;
use std::collections::HashMap;

/// Queries one group handles.
const BQ: u32 = 32;

/// Keys per block of the stream.
const BK: u32 = 16;

/// Four SIMD groups: two form scores, four apply them.
const THREADS: u32 = 128;

/// Floats of padding after each staged row, against bank conflicts.
const PAD: u32 = 4;

/// Columns one group stages and owns at a time.
const CW: u32 = 128;

/// The widest head: four column slices.
const WIDEST: usize = 512;

/// Columns the kernel for `w` stages at a time.
fn chunk(w: usize) -> usize {
    w.min(CW as usize)
}

thread_local! {
    /// The kernel per head width, whole and ragged, built once.
    static TREES: RefCell<HashMap<(u32, bool), Body>> = RefCell::new(HashMap::new());
}

/// Group memory and tile sums the key stream passes through.
#[derive(Group)]
struct Stream {
    q_s: Shared<f32>,
    /// A chunk of keys, then a slice of values transposed.
    kv_s: Shared<f32>,
    p_s: Shared<f32>,
    alpha_s: Shared<f32>,
    l_s: Shared<f32>,
    /// Row r sees keys `[from, to)`. Spare rows see none.
    from_s: Shared<u32>,
    to_s: Shared<u32>,
    s_t: Tiles,
    o00: Tiles,
    o01: Tiles,
    o10: Tiles,
    o11: Tiles,
    /// Each row's peak and sum.
    state: [f32; 2],
}

/// The query block and head one group serves.
#[derive(Group)]
struct Head {
    q0: u32,
    queries: u32,
    heads: u32,
    head: u32,
    kv_heads: u32,
    kvh: u32,
}

/// The head and query block of group `at`, slices aside.
#[kernel]
fn head_at(at: u32, queries: u32, heads: u32, kv_heads: u32) -> Head {
    Head {
        q0: (at / heads) * BQ,
        queries,
        heads,
        head: at % heads,
        kv_heads,
        kvh: (at % heads) / (heads / kv_heads),
    }
}

/// The stream's memory and sums, staging `cw` columns.
#[kernel]
fn stream(#[build] cw: u32) -> Stream {
    Stream {
        q_s: shared::<f32>(BQ * (cw + PAD)),
        kv_s: shared::<f32>(cw * (BK + PAD)),
        p_s: shared::<f32>(BQ * (BK + PAD)),
        alpha_s: shared::<f32>(BQ),
        l_s: shared::<f32>(BQ),
        from_s: shared::<u32>(BQ),
        to_s: shared::<u32>(BQ),
        s_t: tiles(16, 16),
        o00: tiles(16, 16),
        o01: tiles(16, 16),
        o10: tiles(16, 16),
        o11: tiles(16, 16),
        state: [0.0f32; 2],
    }
}

/// Stages `cw` columns from `d0` of the group's query rows.
#[kernel]
fn stage_q(st: &mut Stream, q: Buf<F32>, hd: &Head, d0: u32, #[build] cw: u32, #[build] w: u32) {
    for i in (lid()..BQ * cw).by(THREADS) {
        let r = i / cw;
        let d = i % cw;
        st.q_s[r * (cw + PAD) + d] = if hd.q0 + r < hd.queries {
            q[((hd.q0 + r) * hd.heads + hd.head) as u64 * w as u64 + (d0 + d) as u64]
        } else {
            0.0
        };
    }
}

/// Stages `cw` columns from `d0` of the rows at `at`.
#[kernel]
fn stage_kv<const TRANSPOSED: bool>(
    st: &mut Stream,
    src: Buf<F32>,
    hd: &Head,
    at: u32,
    k0: u32,
    count: u32,
    d0: u32,
    #[build] cw: u32,
    #[build] w: u32,
) {
    for i in (lid()..BK * cw).by(THREADS) {
        let t = i / cw;
        let d = i % cw;
        let v = if k0 + t < count {
            src[((at + t) * hd.kv_heads + hd.kvh) as u64 * w as u64 + (d0 + d) as u64]
        } else {
            0.0
        };
        if TRANSPOSED {
            st.kv_s[d * (BK + PAD) + t] = v;
        } else {
            st.kv_s[t * (cw + PAD) + d] = v;
        }
    }
}

/// Adds `cw` staged columns to the score tiles.
#[kernel]
fn score_chunk(st: &mut Stream, simd: u32, #[build] cw: u32) {
    if simd < 2 {
        st.s_t
            .block(&st.q_s, simd * 16 * (cw + PAD), &st.kv_s, 0, cw + PAD, cw);
    }
}

/// Writes the scores to `p_s`. Unseen keys get -inf.
#[kernel]
fn scores(st: &mut Stream, simd: u32, k0: u32) {
    if simd < 2 {
        st.s_t.each(|r, c, v| {
            let t = k0 + c;
            let row = simd * 16 + r;
            let seen = t >= st.from_s[row] && t < st.to_s[row];
            st.p_s[row * (BK + PAD) + c] = if seen { v } else { f32::NEG_INFINITY };
        });
    }
}

/// A row's softmax step. `state` holds its peak and sum.
#[kernel]
fn soften(st: &mut Stream) {
    let row = lid();
    if row < BQ {
        let mut top = f32::NEG_INFINITY;
        for c in 0..BK {
            top = max(top, st.p_s[row * (BK + PAD) + c]);
        }
        let m_new = max(st.state[0], top);
        // A row that has seen nothing keeps zero sums.
        let safe = if m_new == f32::NEG_INFINITY {
            0.0
        } else {
            m_new
        };
        let alpha = exp(st.state[0] - safe);
        let mut sum = 0.0f32;
        for c in 0..BK {
            let p = exp(st.p_s[row * (BK + PAD) + c] - safe);
            st.p_s[row * (BK + PAD) + c] = p;
            sum += p;
        }
        st.state[1] = st.state[1] * alpha + sum;
        st.state[0] = m_new;
        st.alpha_s[row] = alpha;
    }
}

/// Scales two tile sums by `alpha_s`, then adds column `col`.
#[kernel]
fn fold_pair(
    top: &mut Tiles,
    bottom: &mut Tiles,
    p_s: &Shared<f32>,
    kv_s: &Shared<f32>,
    alpha_s: &Shared<f32>,
    col: u32,
) {
    top.scale(|r| alpha_s[r]);
    bottom.scale(|r| alpha_s[16 + r]);
    top.block(p_s, 0, kv_s, col, BK + PAD, BK);
    bottom.block(p_s, 16 * (BK + PAD), kv_s, col, BK + PAD, BK);
}

/// Writes two tile sums over `l_s`, rows that saw keys.
#[kernel]
fn write_pair(
    top: &Tiles,
    bottom: &Tiles,
    mut out: BufMut<F32>,
    st: &Stream,
    hd: &Head,
    col: u32,
    #[build] w: u32,
) {
    top.each(|r, c, v| {
        if st.to_s[r] > st.from_s[r] {
            let at = ((hd.q0 + r) * hd.heads + hd.head) as u64 * w as u64;
            out[at + (col + c) as u64] = v / st.l_s[r];
        }
    });
    bottom.each(|r, c, v| {
        if st.to_s[16 + r] > st.from_s[16 + r] {
            let at = ((hd.q0 + 16 + r) * hd.heads + hd.head) as u64 * w as u64;
            out[at + (col + c) as u64] = v / st.l_s[16 + r];
        }
    });
}

/// Softens a scored block, then folds in its values.
#[kernel]
fn take_block(
    st: &mut Stream,
    values: Buf<F32>,
    hd: &Head,
    at: u32,
    k0: u32,
    count: u32,
    d0: u32,
    #[build] cw: u32,
    #[build] w: u32,
) {
    let simd = lid() / LANES as u32;
    scores(st, simd, k0);
    barrier();
    soften(st);
    barrier();
    stage_kv::<true>(st, values, hd, at, k0, count, d0, cw, w);
    barrier();
    // Each SIMD group owns up to two column pieces.
    if simd * 16 < cw {
        let col = simd * 16 * (BK + PAD);
        fold_pair(
            &mut st.o00,
            &mut st.o01,
            &st.p_s,
            &st.kv_s,
            &st.alpha_s,
            col,
        );
    }
    if (simd + 4) * 16 < cw {
        let col = (simd + 4) * 16 * (BK + PAD);
        fold_pair(
            &mut st.o10,
            &mut st.o11,
            &st.p_s,
            &st.kv_s,
            &st.alpha_s,
            col,
        );
    }
    barrier();
}

/// Writes the four tile sums over the sums in `state`.
#[kernel]
fn write_all(
    st: &mut Stream,
    mut out: BufMut<F32>,
    hd: &Head,
    d0: u32,
    #[build] cw: u32,
    #[build] w: u32,
) {
    let simd = lid() / LANES as u32;
    if lid() < BQ {
        st.l_s[lid()] = st.state[1];
    }
    barrier();
    // The first column, then each SIMD group's pieces.
    if simd * 16 < cw {
        write_pair(&st.o00, &st.o01, out, st, hd, d0 + simd * 16, w);
    }
    if (simd + 4) * 16 < cw {
        write_pair(&st.o10, &st.o11, out, st, hd, d0 + (simd + 4) * 16, w);
    }
}

/// Lanes keep their output shares across the key stream.
#[kernel]
fn flash(
    q: Buf<F32>,
    k: Buf<F32>,
    values: Buf<F32>,
    mut out: BufMut<F32>,
    queries: u32,
    keys: u32,
    past: u32,
    heads: u32,
    kv_heads: u32,
    back: u32,
    ahead: u32,
    #[build] cw: u32,
    #[build] w: u32,
) {
    let simd = lid() / LANES as u32;
    // A group owns `cw` columns and scores the whole head.
    let slices = w / cw;
    let group = gid() / THREADS;
    let slice = group % slices;
    let hd = head_at(group / slices, queries, heads, kv_heads);
    let row = lid();
    let mut st = stream(cw);
    if lid() < BQ {
        let r = hd.q0 + row;
        let position = past + r;
        let live = r < queries && position < keys;
        let from = if position > back { position - back } else { 0 };
        let to = if keys - position - 1 > ahead {
            position + ahead + 1
        } else {
            keys
        };
        st.from_s[row] = if live { from } else { 0 };
        st.to_s[row] = if live { to } else { 0 };
    }
    barrier();
    // The key blocks any row of the group sees.
    let mut lo = keys;
    let mut hi = 0u32;
    for r in 0..BQ {
        if st.to_s[r] > st.from_s[r] {
            if st.from_s[r] < lo {
                lo = st.from_s[r];
            }
            if st.to_s[r] > hi {
                hi = st.to_s[r];
            }
        }
    }
    st.state[0] = f32::NEG_INFINITY;
    // The slice's first column.
    let start = slice * cw;
    for b in lo / BK..(hi + BK - 1) / BK {
        let k0 = b * BK;
        st.s_t.zero();
        for c in 0..slices {
            let d0 = c * cw;
            stage_q(&mut st, q, &hd, d0, cw, w);
            stage_kv::<false>(&mut st, k, &hd, k0, k0, keys, d0, cw, w);
            barrier();
            score_chunk(&mut st, simd, cw);
            barrier();
        }
        take_block(&mut st, values, &hd, k0, k0, keys, start, cw, w);
    }
    write_all(&mut st, out, &hd, start, cw, w);
}

/// Ragged rows: a pass per slot in the block.
#[kernel]
fn flash_rows(
    q: Buf<F32>,
    k: Buf<F32>,
    values: Buf<F32>,
    mut out: BufMut<F32>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    queries: u32,
    keys: u32,
    heads: u32,
    kv_heads: u32,
    #[build] w: u32,
) {
    let simd = lid() / LANES as u32;
    let hd = head_at(gid() / THREADS, queries, heads, kv_heads);
    let row = lid();
    // A pass serves one slot's rows, up to their positions.
    let mut st = stream(w);
    if lid() < BQ {
        st.from_s[row] = 0;
    }
    stage_q(&mut st, q, &hd, 0, w, w);
    for pass in 0..BQ {
        let lead = hd.q0 + pass;
        let live = lead < queries;
        let slot = if live { slots[lead as u64] } else { 0 };
        // An earlier row's slot was served on its pass.
        let mut fresh = live;
        if live {
            for j in 0..pass {
                if slots[(hd.q0 + j) as u64] == slot {
                    fresh = false;
                }
            }
        }
        if fresh {
            if lid() < BQ {
                let r = hd.q0 + row;
                st.to_s[row] = if r < queries && slots[r as u64] == slot {
                    positions[r as u64] + 1
                } else {
                    0
                };
            }
            barrier();
            let mut last = 0u32;
            for r in 0..BQ {
                if st.to_s[r] > last {
                    last = st.to_s[r];
                }
            }
            if last > keys {
                last = keys;
            }
            st.state[0] = f32::NEG_INFINITY;
            st.state[1] = 0.0;
            st.o00.zero();
            st.o01.zero();
            st.o10.zero();
            st.o11.zero();
            for b in 0..(last + BK - 1) / BK {
                let k0 = b * BK;
                let at = kept_row(slot, k0);
                stage_kv::<false>(&mut st, k, &hd, at, k0, last, 0, w, w);
                barrier();
                st.s_t.zero();
                score_chunk(&mut st, simd, w);
                take_block(&mut st, values, &hd, at, k0, last, 0, w, w);
            }
            write_all(&mut st, out, &hd, 0, w, w);
            barrier();
        }
    }
}

/// Reads scaled queries, and keys and values apart.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "flash",
    order = Order::PerConfig,
    sizes = self.0.sizes(),
    work = self.0.seqs * self.0.heads * self.0.queries * self.0.seen() * self.0.width,
    needs = flash_needs(&self.0),
    source = flash_source,
    launches = flash_launches,
    load = flash_load,
    unproven,
)]
pub(crate) struct Flash(pub Attn);

impl Flash {
    fn tree(&self) -> Body {
        let w = self.0.width;
        TREES.with(|t| {
            t.borrow_mut()
                .entry((w as u32, false))
                .or_insert_with(|| flash_ir(chunk(w) as u32, w as u32))
                .clone()
        })
    }

    /// A group per head, block of queries and column slice.
    fn groups(&self) -> usize {
        let slices = self.0.width / chunk(self.0.width);
        self.0.heads * self.0.queries.div_ceil(BQ as usize) * slices
    }
}

fn flash_load(k: &Flash, len: usize) -> Option<Load> {
    let a = &k.0;
    let (keys, side) = (a.seen() as f64, (a.kv_heads * a.width) as f64);
    // Each query block and slice reads the keys and values.
    let slices = (a.width / chunk(a.width)) as f64;
    let blocks = a.queries.div_ceil(BQ as usize) as f64 * slices;
    let per_seq = 4.0 * keys * side * blocks;
    Some(Load {
        threads: a.seqs * k.groups() * THREADS as usize,
        group: THREADS as usize,
        trips: keys / BK as f64 * slices,
        trip: k.tree().trip(),
        reads: vec![
            4.0 * len as f64,
            per_seq * a.seqs as f64,
            per_seq * a.seqs as f64,
        ],
        written: 4.0 * len as f64,
        flight: 16.0,
        ..Load::default()
    })
}

fn flash_source(k: &Flash) -> Source {
    Source::named(k.tree(), format!("flash_{}", k.0.width))
}

/// One launch per sequence, each at its own offset.
fn flash_launches(k: &Flash, step: &Step) -> Vec<Launch> {
    let a = &k.0;
    let per_q = a.queries * a.heads * a.width;
    let per_kv = a.keys.most() * a.kv_heads * a.width;
    let buffers = |s: usize| {
        [
            Arg::at(step.reads[0], s * per_q),
            Arg::at(step.reads[1], s * per_kv),
            Arg::at(step.reads[2], s * per_kv),
            Arg::at(step.write, s * per_q),
        ]
    };
    let launch = |s| a.launch(buffers(s), [], k.groups(), THREADS as usize);
    (0..a.seqs).map(launch).collect()
}

/// Q and K chunks, P, sums, and the seen ranges.
fn flash_needs(a: &Attn) -> Needs {
    let (w, bq, bk, pad) = (chunk(a.width), BQ as usize, BK as usize, PAD as usize);
    let floats = bq * (w + pad) + w * (bk + pad) + bq * (bk + pad) + 4 * bq;
    Needs {
        lanes: LANES,
        group_bytes: 4 * floats,
        group_threads: THREADS as usize,
        tile: true,
        ..Needs::default()
    }
}

/// Scaled query rows over kept keys, by slot and position.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "flash_rows",
    order = Order::PerConfig,
    sizes = self.0.sizes(),
    work = self.0.heads * self.0.queries * self.0.keys.most() * self.0.width * self.passes(),
    needs = rows_needs(&self.0),
    source = rows_source,
    launches = rows_launches,
    load = rows_load,
)]
pub(crate) struct FlashRows(pub Attn);

impl FlashRows {
    fn tree(&self) -> Body {
        let w = self.0.width as u32;
        TREES.with(|t| {
            t.borrow_mut()
                .entry((w, true))
                .or_insert_with(|| flash_rows_ir(w))
                .clone()
        })
    }

    fn groups(&self) -> usize {
        self.0.heads * self.0.queries.div_ceil(BQ as usize)
    }

    /// Key passes: one a segment, up to a block.
    fn passes(&self) -> usize {
        self.0.segments.min(BQ as usize)
    }
}

/// One launch. A block streams keys once per slot.
fn rows_load(k: &FlashRows, len: usize) -> Option<Load> {
    let a = &k.0;
    let (keys, side) = (a.keys.most() as f64, (a.kv_heads * a.width) as f64);
    let blocks = a.queries.div_ceil(BQ as usize) as f64;
    let passes = k.passes() as f64;
    let streamed = 4.0 * keys * side * blocks * passes;
    Some(Load {
        threads: k.groups() * THREADS as usize,
        group: THREADS as usize,
        trips: passes * keys / BK as f64,
        trip: k.tree().trip(),
        reads: vec![4.0 * len as f64, streamed, streamed],
        written: 4.0 * len as f64,
        flight: 16.0,
        ..Load::default()
    })
}

fn rows_source(k: &FlashRows) -> Source {
    let body = kept::attach(k.tree(), k.0.kept);
    let entry = format!("flash_rows_{}_{}", k.0.width, k.0.kept.name());
    Source::named(body, entry)
}

fn rows_launches(k: &FlashRows, step: &Step) -> Vec<Launch> {
    let a = &k.0;
    let mut args = a.rows_args(step);
    args.extend(a.kept_args());
    vec![Launch {
        args,
        shape: Shape::Grid {
            across: k.groups(),
            down: 1,
            threads: THREADS as usize,
        },
    }]
}

/// Q, the K and V slot, P, sums, seen counts.
fn rows_needs(a: &Attn) -> Needs {
    let (w, bq, bk, pad) = (a.width, BQ as usize, BK as usize, PAD as usize);
    let floats = bq * (w + pad) + w * (bk + pad) + bq * (bk + pad) + 3 * bq;
    Needs {
        lanes: LANES,
        group_bytes: 4 * floats,
        group_threads: THREADS as usize,
        tile: true,
        ..Needs::default()
    }
}

/// Scale the queries, then attend by tiles.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Attend = tree.site(at)?;
    // Tiles of sixteen along the head, in whole slices.
    let w = site.sizes.width;
    if !w.is_multiple_of(16) || w > WIDEST || (w > CW as usize && !w.is_multiple_of(CW as usize)) {
        return None;
    }
    if site.rows.is_some() {
        if site.sizes.mask != Mask::Causal || w > CW as usize {
            return None;
        }
        // Keys stage a block at a time, in one run.
        let run = site.sizes.kept.run(site.sizes.context);
        if !run.is_multiple_of(BK as usize) {
            return None;
        }
        return site.ragged(FlashRows(site.sizes));
    }
    let (scaled, out) = crate::scaled(&site)?;
    let reads = [Read::Draft(0), Read::Node(site.k), Read::Node(site.v)];
    Some(vec![scaled, Draft::new(Flash(site.sizes), reads, out)])
}

sensai_lower::rule!("flash", propose);
