//! Attention on tiles, the keys streamed in blocks.

use codegen::ir::Body;
use codegen::kept;
use codegen::prelude::*;
use graph::{Mask, Order};
use lower::kernels::{Map, MapStage};
use lower::sites;
use lower::{
    Arg, Attn, Draft, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Tree, UNPROVEN,
};
use std::cell::RefCell;
use std::collections::HashMap;

/// Queries one group handles.
const BQ: u32 = 32;

/// Keys one block of the stream holds.
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

/// Stages `cw` columns from `d0` of the group's query rows.
#[kernel]
fn stage_q(
    q_s: &mut Shared<f32>,
    q: Buf<F32>,
    q0: u32,
    queries: u32,
    heads: u32,
    h: u32,
    d0: u32,
    #[build] cw: u32,
    #[build] w: u32,
) {
    for i in (lid()..BQ * cw).by(THREADS) {
        let r = i / cw;
        let d = i % cw;
        q_s[r * (cw + PAD) + d] = if q0 + r < queries {
            q[((q0 + r) * heads + h) as u64 * w as u64 + (d0 + d) as u64]
        } else {
            0.0
        };
    }
}

/// Stages `cw` columns from `d0` of the rows at `at`.
#[kernel]
fn stage_kv<const TRANSPOSED: bool>(
    kv_s: &mut Shared<f32>,
    src: Buf<F32>,
    at: u32,
    k0: u32,
    count: u32,
    kv_heads: u32,
    kvh: u32,
    d0: u32,
    #[build] cw: u32,
    #[build] w: u32,
) {
    for i in (lid()..BK * cw).by(THREADS) {
        let t = i / cw;
        let d = i % cw;
        let v = if k0 + t < count {
            src[((at + t) * kv_heads + kvh) as u64 * w as u64 + (d0 + d) as u64]
        } else {
            0.0
        };
        if TRANSPOSED {
            kv_s[d * (BK + PAD) + t] = v;
        } else {
            kv_s[t * (cw + PAD) + d] = v;
        }
    }
}

/// Adds `cw` staged columns to the score tiles.
#[kernel]
fn score_chunk(
    s_t: &mut Tiles,
    q_s: &Shared<f32>,
    kv_s: &Shared<f32>,
    simd: u32,
    #[build] cw: u32,
) {
    if simd < 2 {
        s_t.block(q_s, simd * 16 * (cw + PAD), kv_s, 0, cw + PAD, cw);
    }
}

/// Writes the scores to `p_s`. Unseen keys get -inf.
#[kernel]
fn scores(
    s_t: &Tiles,
    p_s: &mut Shared<f32>,
    from_s: &Shared<u32>,
    to_s: &Shared<u32>,
    simd: u32,
    k0: u32,
) {
    if simd < 2 {
        s_t.each(|r, c, v| {
            let t = k0 + c;
            let row = simd * 16 + r;
            let seen = t >= from_s[row] && t < to_s[row];
            p_s[row * (BK + PAD) + c] = if seen { v } else { f32::NEG_INFINITY };
        });
    }
}

/// A row's softmax step. `state` holds its peak and sum.
#[kernel]
fn soften(p_s: &mut Shared<f32>, alpha_s: &mut Shared<f32>, state: &mut [f32; 2], row: u32) {
    if lid() < BQ {
        let mut top = f32::NEG_INFINITY;
        for c in 0..BK {
            top = max(top, p_s[row * (BK + PAD) + c]);
        }
        let m_new = max(state[0], top);
        // A row that has seen nothing keeps zero sums.
        let safe = if m_new == f32::NEG_INFINITY {
            0.0
        } else {
            m_new
        };
        let alpha = exp(state[0] - safe);
        let mut sum = 0.0f32;
        for c in 0..BK {
            let p = exp(p_s[row * (BK + PAD) + c] - safe);
            p_s[row * (BK + PAD) + c] = p;
            sum += p;
        }
        state[1] = state[1] * alpha + sum;
        state[0] = m_new;
        alpha_s[row] = alpha;
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
    l_s: &Shared<f32>,
    from_s: &Shared<u32>,
    to_s: &Shared<u32>,
    q0: u32,
    heads: u32,
    h: u32,
    col: u32,
    #[build] w: u32,
) {
    top.each(|r, c, v| {
        if to_s[r] > from_s[r] {
            out[((q0 + r) * heads + h) as u64 * w as u64 + (col + c) as u64] = v / l_s[r];
        }
    });
    bottom.each(|r, c, v| {
        if to_s[16 + r] > from_s[16 + r] {
            out[((q0 + 16 + r) * heads + h) as u64 * w as u64 + (col + c) as u64] = v / l_s[16 + r];
        }
    });
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
    let rest = group / slices;
    let h = rest % heads;
    let q0 = (rest / heads) * BQ;
    let kvh = h / (heads / kv_heads);
    let row = lid();
    let mut q_s = shared::<f32>(BQ * (cw + PAD));
    // A chunk of keys, then the slice of values transposed.
    let mut kv_s = shared::<f32>(cw * (BK + PAD));
    let mut p_s = shared::<f32>(BQ * (BK + PAD));
    let mut alpha_s = shared::<f32>(BQ);
    let mut l_s = shared::<f32>(BQ);
    // Row r sees keys `[from, to)`. Spare rows see none.
    let mut from_s = shared::<u32>(BQ);
    let mut to_s = shared::<u32>(BQ);
    if lid() < BQ {
        let r = q0 + row;
        let position = past + r;
        let live = r < queries && position < keys;
        let from = if position > back { position - back } else { 0 };
        let to = if keys - position - 1 > ahead {
            position + ahead + 1
        } else {
            keys
        };
        from_s[row] = if live { from } else { 0 };
        to_s[row] = if live { to } else { 0 };
    }
    barrier();
    // The key blocks any row of the group sees.
    let mut lo = keys;
    let mut hi = 0u32;
    for r in 0..BQ {
        if to_s[r] > from_s[r] {
            if from_s[r] < lo {
                lo = from_s[r];
            }
            if to_s[r] > hi {
                hi = to_s[r];
            }
        }
    }
    let mut s_t = tiles(16, 16);
    let mut o00 = tiles(16, 16);
    let mut o01 = tiles(16, 16);
    let mut o10 = tiles(16, 16);
    let mut o11 = tiles(16, 16);
    let mut state = [0.0f32; 2];
    state[0] = f32::NEG_INFINITY;
    // Each SIMD group owns up to two column pieces.
    let first = simd * 16 < cw;
    let second = (simd + 4) * 16 < cw;
    for b in lo / BK..(hi + BK - 1) / BK {
        let k0 = b * BK;
        s_t.zero();
        for c in 0..slices {
            let d0 = c * cw;
            stage_q(&mut q_s, q, q0, queries, heads, h, d0, cw, w);
            stage_kv::<false>(&mut kv_s, k, k0, k0, keys, kv_heads, kvh, d0, cw, w);
            barrier();
            score_chunk(&mut s_t, &q_s, &kv_s, simd, cw);
            barrier();
        }
        scores(&s_t, &mut p_s, &from_s, &to_s, simd, k0);
        barrier();
        soften(&mut p_s, &mut alpha_s, &mut state, row);
        barrier();
        let d0 = slice * cw;
        stage_kv::<true>(&mut kv_s, values, k0, k0, keys, kv_heads, kvh, d0, cw, w);
        barrier();
        if first {
            fold_pair(
                &mut o00,
                &mut o01,
                &p_s,
                &kv_s,
                &alpha_s,
                simd * 16 * (BK + PAD),
            );
        }
        if second {
            fold_pair(
                &mut o10,
                &mut o11,
                &p_s,
                &kv_s,
                &alpha_s,
                (simd + 4) * 16 * (BK + PAD),
            );
        }
        barrier();
    }
    if lid() < BQ {
        l_s[row] = state[1];
    }
    barrier();
    // The slice's first column, then each SIMD group's pieces.
    let d0 = slice * cw;
    if first {
        let col = d0 + simd * 16;
        write_pair(&o00, &o01, out, &l_s, &from_s, &to_s, q0, heads, h, col, w);
    }
    if second {
        let col = d0 + (simd + 4) * 16;
        write_pair(&o10, &o11, out, &l_s, &from_s, &to_s, q0, heads, h, col, w);
    }
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
    let group = gid() / THREADS;
    let h = group % heads;
    let q0 = (group / heads) * BQ;
    let kvh = h / (heads / kv_heads);
    let row = lid();
    let mut q_s = shared::<f32>(BQ * (w + PAD));
    // Keys, then the values transposed, in one slot.
    let mut kv_s = shared::<f32>(w * (BK + PAD));
    let mut p_s = shared::<f32>(BQ * (BK + PAD));
    let mut alpha_s = shared::<f32>(BQ);
    let mut l_s = shared::<f32>(BQ);
    // Rows the pass serves: one past the position. Else zero.
    let mut seen_s = shared::<u32>(BQ);
    let mut from_s = shared::<u32>(BQ);
    if lid() < BQ {
        from_s[row] = 0;
    }
    stage_q(&mut q_s, q, q0, queries, heads, h, 0, w, w);
    let mut s_t = tiles(16, 16);
    let mut o00 = tiles(16, 16);
    let mut o01 = tiles(16, 16);
    let mut o10 = tiles(16, 16);
    let mut o11 = tiles(16, 16);
    let mut state = [0.0f32; 2];
    // Each SIMD group owns up to two column pieces.
    let first = simd * 16 < w;
    let second = (simd + 4) * 16 < w;
    for pass in 0..BQ {
        let lead = q0 + pass;
        let live = lead < queries;
        let slot = if live { slots[lead as u64] } else { 0 };
        // An earlier row's slot was served on its pass.
        let mut fresh = live;
        if live {
            for j in 0..pass {
                if slots[(q0 + j) as u64] == slot {
                    fresh = false;
                }
            }
        }
        if fresh {
            if lid() < BQ {
                let r = q0 + row;
                seen_s[row] = if r < queries && slots[r as u64] == slot {
                    positions[r as u64] + 1
                } else {
                    0
                };
            }
            barrier();
            let mut last = 0u32;
            for r in 0..BQ {
                if seen_s[r] > last {
                    last = seen_s[r];
                }
            }
            if last > keys {
                last = keys;
            }
            state[0] = f32::NEG_INFINITY;
            state[1] = 0.0;
            o00.zero();
            o01.zero();
            o10.zero();
            o11.zero();
            for b in 0..(last + BK - 1) / BK {
                let k0 = b * BK;
                let at = kept_row(slot, k0);
                stage_kv::<false>(&mut kv_s, k, at, k0, last, kv_heads, kvh, 0, w, w);
                barrier();
                s_t.zero();
                score_chunk(&mut s_t, &q_s, &kv_s, simd, w);
                scores(&s_t, &mut p_s, &from_s, &seen_s, simd, k0);
                barrier();
                soften(&mut p_s, &mut alpha_s, &mut state, row);
                barrier();
                stage_kv::<true>(&mut kv_s, values, at, k0, last, kv_heads, kvh, 0, w, w);
                barrier();
                if first {
                    fold_pair(
                        &mut o00,
                        &mut o01,
                        &p_s,
                        &kv_s,
                        &alpha_s,
                        simd * 16 * (BK + PAD),
                    );
                }
                if second {
                    let col = (simd + 4) * 16 * (BK + PAD);
                    fold_pair(&mut o10, &mut o11, &p_s, &kv_s, &alpha_s, col);
                }
                barrier();
            }
            if lid() < BQ {
                l_s[row] = state[1];
            }
            barrier();
            if first {
                let col = simd * 16;
                write_pair(
                    &o00, &o01, out, &l_s, &from_s, &seen_s, q0, heads, h, col, w,
                );
            }
            if second {
                let col = (simd + 4) * 16;
                write_pair(
                    &o10, &o11, out, &l_s, &from_s, &seen_s, q0, heads, h, col, w,
                );
            }
            barrier();
        }
    }
}

/// Reads scaled queries, and keys and values apart.
#[derive(Clone, Copy, Debug)]
pub struct Flash(pub Attn);

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

impl Kernel for Flash {
    over_attn!("flash");

    fn load(&self, len: usize) -> Option<Load> {
        let a = &self.0;
        let (keys, side) = (a.seen() as f64, (a.kv_heads * a.width) as f64);
        // Each query block and slice reads the keys and values.
        let slices = (a.width / chunk(a.width)) as f64;
        let blocks = a.queries.div_ceil(BQ as usize) as f64 * slices;
        let per_seq = 4.0 * keys * side * blocks;
        Some(Load {
            threads: a.seqs * self.groups() * THREADS as usize,
            group: THREADS as usize,
            trips: keys / BK as f64 * slices,
            trip: self.tree().trip(),
            reads: vec![
                4.0 * len as f64,
                per_seq * a.seqs as f64,
                per_seq * a.seqs as f64,
            ],
            written: 4.0 * len as f64,
            flight: 16.0,
        })
    }

    fn source(&self) -> Source {
        Source::named(self.tree(), format!("flash_{}", self.0.width))
    }

    /// One launch per sequence, each at its own offset.
    fn launches(&self, step: &Step) -> Vec<Launch> {
        let a = &self.0;
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
        let launch = |s| a.launch(buffers(s), [], self.groups(), THREADS as usize);
        (0..a.seqs).map(launch).collect()
    }

    fn work(&self, _: usize) -> f64 {
        let a = &self.0;
        (a.seqs * a.heads * a.queries * a.seen() * a.width) as f64
    }

    fn guess(&self) -> f64 {
        UNPROVEN
    }

    /// Q and K chunks, P, sums, and the seen ranges.
    fn needs(&self) -> Needs {
        let (w, bq, bk, pad) = (chunk(self.0.width), BQ as usize, BK as usize, PAD as usize);
        let floats = bq * (w + pad) + w * (bk + pad) + bq * (bk + pad) + 4 * bq;
        Needs {
            lanes: LANES,
            group_bytes: 4 * floats,
            group_threads: THREADS as usize,
            tile: true,
            ..Needs::default()
        }
    }
}

/// Scaled query rows over kept keys, by slot and position.
#[derive(Clone, Copy, Debug)]
pub struct FlashRows(pub Attn);

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
}

impl Kernel for FlashRows {
    over_attn!("flash_rows");

    /// One launch. A block streams keys once per slot.
    fn load(&self, len: usize) -> Option<Load> {
        let a = &self.0;
        let (keys, side) = (a.keys.most() as f64, (a.kv_heads * a.width) as f64);
        let blocks = a.queries.div_ceil(BQ as usize) as f64;
        let passes = a.segments.min(BQ as usize) as f64;
        let streamed = 4.0 * keys * side * blocks * passes;
        Some(Load {
            threads: self.groups() * THREADS as usize,
            group: THREADS as usize,
            trips: passes * keys / BK as f64,
            trip: self.tree().trip(),
            reads: vec![4.0 * len as f64, streamed, streamed],
            written: 4.0 * len as f64,
            flight: 16.0,
        })
    }

    fn source(&self) -> Source {
        let body = kept::attach(self.tree(), self.0.kept);
        let entry = format!("flash_rows_{}_{}", self.0.width, self.0.kept.name());
        Source::named(body, entry)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let a = &self.0;
        let mut args = a.rows_args(step);
        args.extend(a.kept_args());
        vec![Launch {
            args,
            shape: Shape::Grid {
                across: self.groups(),
                down: 1,
                threads: THREADS as usize,
            },
        }]
    }

    fn work(&self, _: usize) -> f64 {
        let a = &self.0;
        let passes = a.segments.min(BQ as usize);
        (a.heads * a.queries * a.keys.most() * a.width * passes) as f64
    }

    /// Q, the K and V slot, P, sums, seen counts.
    fn needs(&self) -> Needs {
        let (w, bq, bk, pad) = (self.0.width, BQ as usize, BK as usize, PAD as usize);
        let floats = bq * (w + pad) + w * (bk + pad) + bq * (bk + pad) + 3 * bq;
        Needs {
            lanes: LANES,
            group_bytes: 4 * floats,
            group_threads: THREADS as usize,
            tile: true,
            ..Needs::default()
        }
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
    let sites::Attend {
        scale,
        q,
        k,
        v,
        sizes,
        ..
    } = site;
    let out = site.out();
    // Sequences would sit at moving offsets as the keys grow.
    if sizes.keys.runtime() && sizes.seqs > 1 {
        return None;
    }
    let out = sizes.seqs * out;
    let reads = [Read::Draft(0), Read::Node(k), Read::Node(v)];
    Some(vec![
        Draft::new(Map::plain(vec![MapStage::Scale(scale)]), [q], out),
        Draft::new(Flash(sizes), reads, out),
    ])
}

lower::rule!("flash", propose);
