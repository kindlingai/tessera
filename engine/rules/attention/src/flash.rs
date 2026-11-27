//! Attention on tiles, the keys streamed in blocks.

use std::cell::RefCell;
use std::collections::HashMap;
use tessera_codegen::ir::Body;
use tessera_codegen::prelude::*;
use tessera_graph::{Mask, Order};
use tessera_lower::form::Load as Stored;
use tessera_lower::sites;
use tessera_lower::{
    Arg, Attn, Draft, Kernel, Launch, Load, Needs, Read, Source, Step, T, Tree, knob,
};

mod sunk;
mod sunk_metal;

knob!("flash", "bk", 16, 32);
knob!("flash", "held", 0, 1);

/// Queries one group handles.
pub(crate) const BQ: u32 = 32;

/// Keys per block, and held queries or not.
pub(crate) const WAYS: [(u32, bool); 4] = [(16, false), (32, false), (16, true), (32, true)];

/// Four SIMD groups. Scores take two per 16 keys.
pub(crate) const THREADS: u32 = 128;

/// Query rows each SIMD group softens.
const ROWS: u32 = BQ / (THREADS / LANES as u32);

/// Floats of padding after each staged row, against bank conflicts.
const PAD: u32 = 4;

/// Columns one group stages and owns at a time.
pub(crate) const CW: u32 = 128;

/// Floats one wide load brings.
pub(crate) const WIDE: u32 = 4;

/// Floats a thread holds while staging, at most.
const HELD: usize = (32 * CW / THREADS) as usize;

/// The widest head: four column slices.
pub(crate) const WIDEST: usize = 512;

/// Head width rounded up to a tile multiple of 16.
pub(crate) fn padded(w: usize) -> usize {
    w.next_multiple_of(16)
}

/// Columns the kernel for `w` stages at a time.
pub(crate) fn chunk(w: usize) -> usize {
    padded(w).min(CW as usize)
}

type Key = (
    u32,
    bool,
    u32,
    bool,
    &'static str,
    &'static str,
    &'static str,
);

thread_local! {
    /// Each kernel tree, by width, raggedness, block and forms.
    static TREES: RefCell<HashMap<Key, Body>> = RefCell::new(HashMap::new());
}

/// Group memory and tile sums the key stream passes through.
#[derive(Group)]
pub(crate) struct Stream {
    q_s: Shared<f32>,
    /// A chunk of keys, then a slice of values transposed.
    kv_s: Shared<f32>,
    /// With held queries: keys, staged beside the values.
    k_s: Shared<f32>,
    p_s: Shared<f32>,
    alpha_s: Shared<f32>,
    l_s: Shared<f32>,
    /// Row r sees keys `[from, to)`. Spare rows see none.
    pub(crate) from_s: Shared<u32>,
    pub(crate) to_s: Shared<u32>,
    s_t: Tiles,
    o00: Tiles,
    o01: Tiles,
    o10: Tiles,
    o11: Tiles,
    /// Peaks, then sums, of the rows this SIMD group softens.
    state: [f32; 16],
    /// The query rows this SIMD group scores, when held.
    qa: Held,
}

/// The query block and head one group serves.
#[derive(Group)]
pub(crate) struct Head {
    pub(crate) q0: u32,
    pub(crate) queries: u32,
    pub(crate) heads: u32,
    pub(crate) head: u32,
    pub(crate) kv_heads: u32,
    pub(crate) kvh: u32,
}

/// The head and query block of group `at`, slices aside.
#[kernel]
pub(crate) fn head_at(at: u32, queries: u32, heads: u32, kv_heads: u32) -> Head {
    Head {
        q0: (at / heads) * BQ,
        queries,
        heads,
        head: at % heads,
        kv_heads,
        kvh: (at % heads) / (heads / kv_heads),
    }
}

/// Floats of `q_s`, `kv_s` and `k_s`.
fn staged(cw: u32, bk: u32, held_q: bool) -> [u32; 3] {
    // Held queries pass through `k_s`.
    let (queries, keys, values) = (BQ * (cw + PAD), bk * (cw + PAD), cw * (bk + PAD));
    // An unused buffer keeps one aligned four.
    match held_q {
        true => [WIDE, values, keys.max(queries)],
        false => [queries, keys.max(values), WIDE],
    }
}

/// The stream's memory and sums: `cw` columns, `bk` keys.
#[kernel]
pub(crate) fn stream(#[build] cw: u32, #[build] bk: u32, #[build] held_q: bool) -> Stream {
    Stream {
        q_s: shared::<f32>(staged(cw, bk, held_q)[0]),
        kv_s: shared::<f32>(staged(cw, bk, held_q)[1]),
        k_s: shared::<f32>(staged(cw, bk, held_q)[2]),
        p_s: shared::<f32>(BQ * (bk + PAD)),
        alpha_s: shared::<f32>(BQ),
        l_s: shared::<f32>(BQ),
        from_s: shared::<u32>(BQ),
        to_s: shared::<u32>(BQ),
        s_t: tiles(16, 16),
        o00: tiles(16, 16),
        o01: tiles(16, 16),
        o10: tiles(16, 16),
        o11: tiles(16, 16),
        state: [0.0f32; 16],
        qa: held(16, cw),
    }
}

/// Stages `cw` columns from `d0` of the group's query rows.
#[kernel]
fn stage_q(
    dst: &mut Shared<f32>,
    q: Buf<F32>,
    hd: &Head,
    d0: u32,
    #[build] cw: u32,
    #[build] w: u32,
) {
    for i in (lid()..BQ * cw).by(THREADS) {
        let r = i / cw;
        let d = i % cw;
        dst[r * (cw + PAD) + d] = if hd.q0 + r < hd.queries && d0 + d < w {
            q[((hd.q0 + r) * hd.heads + hd.head) as u64 * w as u64 + (d0 + d) as u64]
        } else {
            0.0
        };
    }
}

/// Stages `cw` columns from `d0` of the rows at `at`.
#[kernel]
fn stage_kv<const TRANSPOSED: bool>(
    dst: &mut Shared<f32>,
    #[form(kv_form)] src: Buf<Dyn>,
    hd: &Head,
    at: u32,
    k0: u32,
    count: u32,
    d0: u32,
    #[build] cw: u32,
    #[build] w: u32,
    #[build] bk: u32,
    #[build] kv_form: &'static dyn Stored,
) {
    // All loads, then all stores: the loads fly together.
    let mut held = [0.0f32; HELD];
    for j in 0..(bk * (cw / WIDE) + THREADS - 1) / THREADS {
        let i = lid() + j * THREADS;
        // Transposed, keys run across lanes: no bank conflicts.
        let t = if TRANSPOSED { i % bk } else { i / (cw / WIDE) };
        let d = if TRANSPOSED {
            (i / bk) * WIDE
        } else {
            (i % (cw / WIDE)) * WIDE
        };
        // Past `count` or `w` it reads 0.
        let mut v = F32x4::new(0.0, 0.0, 0.0, 0.0);
        if i < bk * (cw / WIDE) && k0 + t < count && d0 + d < w {
            v = src.load4(((at + t) * hd.kv_heads + hd.kvh) as u64 * w as u64 + (d0 + d) as u64);
        }
        held[(4 * j) as usize] = v.x;
        held[(4 * j + 1) as usize] = v.y;
        held[(4 * j + 2) as usize] = v.z;
        held[(4 * j + 3) as usize] = v.w;
    }
    for j in 0..(bk * (cw / WIDE) + THREADS - 1) / THREADS {
        let i = lid() + j * THREADS;
        let t = if TRANSPOSED { i % bk } else { i / (cw / WIDE) };
        let d = if TRANSPOSED {
            (i / bk) * WIDE
        } else {
            (i % (cw / WIDE)) * WIDE
        };
        if i < bk * (cw / WIDE) {
            for e in 0..WIDE {
                let v = held[(4 * j + e) as usize];
                if TRANSPOSED {
                    dst[(d + e) * (bk + PAD) + t] = v;
                } else {
                    dst[t * (cw + PAD) + d + e] = v;
                }
            }
        }
    }
}

/// Adds staged columns to each SIMD group's score tile.
#[kernel]
fn score_chunk(
    st: &mut Stream,
    simd: u32,
    #[build] cw: u32,
    #[build] bk: u32,
    #[build] held_q: bool,
) {
    if simd < bk / 8 {
        let a = (simd % 2) * 16 * (cw + PAD);
        let b = (simd / 2) * 16 * (cw + PAD);
        if held_q {
            st.s_t.block(&st.qa, 0, &st.k_s, b, cw + PAD, cw);
        } else {
            st.s_t.block(&st.q_s, a, &st.kv_s, b, cw + PAD, cw);
        }
    }
}

/// Writes the scores to `p_s`. Unseen keys get -inf.
#[kernel]
fn scores(st: &mut Stream, simd: u32, k0: u32, #[build] bk: u32) {
    if simd < bk / 8 {
        st.s_t.each(|r, c, v| {
            let col = (simd / 2) * 16 + c;
            let t = k0 + col;
            let row = (simd % 2) * 16 + r;
            let seen = t >= st.from_s[row] && t < st.to_s[row];
            st.p_s[row * (bk + PAD) + col] = if seen { v } else { f32::NEG_INFINITY };
        });
    }
}

/// A softmax step, `ROWS` rows per SIMD group.
#[kernel]
fn soften(st: &mut Stream, #[build] bk: u32) {
    let simd = lid() / LANES as u32;
    for i in 0..ROWS {
        let row = simd * ROWS + i;
        let at = row * (bk + PAD) + lane();
        let s = if lane() < bk {
            st.p_s[at]
        } else {
            f32::NEG_INFINITY
        };
        let mut top = s;
        top = max(top, shfl_xor(top, 16));
        top = max(top, shfl_xor(top, 8));
        top = max(top, shfl_xor(top, 4));
        top = max(top, shfl_xor(top, 2));
        top = max(top, shfl_xor(top, 1));
        let m_new = max(st.state[i as usize], top);
        // A row that has seen nothing keeps zero sums.
        let safe = if m_new == f32::NEG_INFINITY {
            0.0
        } else {
            m_new
        };
        let alpha = exp(st.state[i as usize] - safe);
        let p = exp(s - safe);
        let mut sum = p;
        sum = sum32(sum);
        if lane() < bk {
            st.p_s[at] = p;
        }
        st.state[(ROWS + i) as usize] = st.state[(ROWS + i) as usize] * alpha + sum;
        st.state[i as usize] = m_new;
        if lane() == 0 {
            st.alpha_s[row] = alpha;
        }
    }
}

/// Starts each softened row with no peak and no sum.
#[kernel]
fn restart(st: &mut Stream) {
    for i in 0..ROWS {
        st.state[i as usize] = f32::NEG_INFINITY;
        st.state[(ROWS + i) as usize] = 0.0;
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
    #[build] bk: u32,
) {
    top.scale(|r| alpha_s[r]);
    bottom.scale(|r| alpha_s[16 + r]);
    top.block(p_s, 0, kv_s, col, bk + PAD, bk);
    bottom.block(p_s, 16 * (bk + PAD), kv_s, col, bk + PAD, bk);
}

/// Writes two tile sums over `l_s`, rows that saw keys.
#[kernel]
fn write_pair<const ZERO_EMPTY: bool>(
    top: &Tiles,
    bottom: &Tiles,
    #[form(out_form)] mut out: BufMut<Dyn>,
    st: &Stream,
    hd: &Head,
    col: u32,
    #[build] w: u32,
    #[build] out_form: &'static dyn Stored,
) {
    top.each(|r, c, v| {
        let live = st.to_s[r] > st.from_s[r];
        if col + c < w && (live || (ZERO_EMPTY && hd.q0 + r < hd.queries)) {
            let at = ((hd.q0 + r) * hd.heads + hd.head) as u64 * w as u64;
            out[at + (col + c) as u64] = if live { v / st.l_s[r] } else { 0.0 };
        }
    });
    bottom.each(|r, c, v| {
        let live = st.to_s[16 + r] > st.from_s[16 + r];
        if col + c < w && (live || (ZERO_EMPTY && hd.q0 + 16 + r < hd.queries)) {
            let at = ((hd.q0 + 16 + r) * hd.heads + hd.head) as u64 * w as u64;
            out[at + (col + c) as u64] = if live { v / st.l_s[16 + r] } else { 0.0 };
        }
    });
}

/// Softens a scored block, then folds in its values.
#[kernel]
fn take_block(
    st: &mut Stream,
    #[form(v_form)] values: Buf<Dyn>,
    hd: &Head,
    at: u32,
    k0: u32,
    count: u32,
    d0: u32,
    #[build] cw: u32,
    #[build] w: u32,
    #[build] bk: u32,
    #[build] held_q: bool,
    #[build] v_form: &'static dyn Stored,
) {
    let simd = lid() / LANES as u32;
    scores(st, simd, k0, bk);
    barrier();
    fold_block(st, values, hd, at, k0, count, d0, cw, w, bk, held_q, v_form);
}

/// Softens the scores in `p_s`, then folds values in.
#[kernel]
fn fold_block(
    st: &mut Stream,
    #[form(v_form)] values: Buf<Dyn>,
    hd: &Head,
    at: u32,
    k0: u32,
    count: u32,
    d0: u32,
    #[build] cw: u32,
    #[build] w: u32,
    #[build] bk: u32,
    #[build] held_q: bool,
    #[build] v_form: &'static dyn Stored,
) {
    let simd = lid() / LANES as u32;
    soften(st, bk);
    barrier();
    // Held queries made room: values staged with the keys.
    if !held_q {
        stage_kv::<true>(
            &mut st.kv_s,
            values,
            hd,
            at,
            k0,
            count,
            d0,
            cw,
            w,
            bk,
            v_form,
        );
        barrier();
    }
    // Each SIMD group owns up to two column pieces.
    if simd * 16 < cw {
        let col = simd * 16 * (bk + PAD);
        fold_pair(
            &mut st.o00,
            &mut st.o01,
            &st.p_s,
            &st.kv_s,
            &st.alpha_s,
            col,
            bk,
        );
    }
    if (simd + 4) * 16 < cw {
        let col = (simd + 4) * 16 * (bk + PAD);
        fold_pair(
            &mut st.o10,
            &mut st.o11,
            &st.p_s,
            &st.kv_s,
            &st.alpha_s,
            col,
            bk,
        );
    }
    barrier();
}

/// Writes the four tile sums over the sums in `state`.
#[kernel]
fn write_all<const ZERO_EMPTY: bool>(
    st: &mut Stream,
    #[form(out_form)] mut out: BufMut<Dyn>,
    hd: &Head,
    d0: u32,
    #[build] cw: u32,
    #[build] w: u32,
    #[build] out_form: &'static dyn Stored,
) {
    let simd = lid() / LANES as u32;
    for i in 0..ROWS {
        if lane() == i {
            st.l_s[simd * ROWS + i] = st.state[(ROWS + i) as usize];
        }
    }
    barrier();
    // The first column, then each SIMD group's pieces.
    if simd * 16 < cw {
        write_pair::<ZERO_EMPTY>(&st.o00, &st.o01, out, st, hd, d0 + simd * 16, w, out_form);
    }
    if (simd + 4) * 16 < cw {
        write_pair::<ZERO_EMPTY>(
            &st.o10,
            &st.o11,
            out,
            st,
            hd,
            d0 + (simd + 4) * 16,
            w,
            out_form,
        );
    }
}

/// Streams key blocks covering `st.from_s..st.to_s`, writes outputs.
#[kernel]
pub(crate) fn flash_body(
    st: &mut Stream,
    q: Buf<F32>,
    #[form(k_form)] k: Buf<Dyn>,
    #[form(v_form)] values: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    hd: &Head,
    keys: u32,
    slice: u32,
    slices: u32,
    #[build] cw: u32,
    #[build] w: u32,
    #[build] bk: u32,
    #[build] held_q: bool,
    #[build] k_form: &'static dyn Stored,
    #[build] v_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    let simd = lid() / LANES as u32;
    // The key blocks any row of the group sees.
    let mut lo = keys;
    let mut hi = 0u32;
    for qr in 0..BQ {
        if st.to_s[qr] > st.from_s[qr] {
            if st.from_s[qr] < lo {
                lo = st.from_s[qr];
            }
            if st.to_s[qr] > hi {
                hi = st.to_s[qr];
            }
        }
    }
    restart(st);
    // The slice's first column.
    let start = slice * cw;
    // One slice holds the head: queries stay staged or held.
    if held_q {
        stage_q(&mut st.k_s, q, hd, 0, cw, w);
        barrier();
        st.qa.hold(&st.k_s, (simd % 2) * 16 * (cw + PAD), cw + PAD);
        barrier();
    }
    if slices == 1 && !held_q {
        stage_q(&mut st.q_s, q, hd, 0, cw, w);
    }
    let end = if hi > lo { (hi + bk - 1) / bk } else { lo / bk };
    for b in lo / bk..end {
        let k0 = b * bk;
        st.s_t.zero();
        for sc in 0..slices {
            let d0 = sc * cw;
            if slices > 1 {
                stage_q(&mut st.q_s, q, hd, d0, cw, w);
            }
            if held_q {
                stage_kv::<false>(&mut st.k_s, k, hd, k0, k0, keys, d0, cw, w, bk, k_form);
                stage_kv::<true>(
                    &mut st.kv_s,
                    values,
                    hd,
                    k0,
                    k0,
                    keys,
                    start,
                    cw,
                    w,
                    bk,
                    v_form,
                );
            } else {
                stage_kv::<false>(&mut st.kv_s, k, hd, k0, k0, keys, d0, cw, w, bk, k_form);
            }
            barrier();
            score_chunk(st, simd, cw, bk, held_q);
            barrier();
        }
        take_block(
            st, values, hd, k0, k0, keys, start, cw, w, bk, held_q, v_form,
        );
    }
    write_all::<true>(st, out, hd, start, cw, w, out_form);
}

/// Lanes keep their output shares across the key stream.
#[kernel]
fn flash(
    q: Buf<F32>,
    #[form(k_form)] k: Buf<Dyn>,
    #[form(v_form)] values: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    queries: u32,
    keys: u32,
    past: u32,
    heads: u32,
    kv_heads: u32,
    back: u32,
    ahead: u32,
    #[build] cw: u32,
    #[build] w: u32,
    #[build] bk: u32,
    #[build] held_q: bool,
    #[build] k_form: &'static dyn Stored,
    #[build] v_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    // A group owns `cw` columns and scores the whole head.
    let slices = (((w + 15) / 16) * 16) / cw;
    let group = gid() / THREADS;
    let slice = group % slices;
    let hd = head_at(group / slices, queries, heads, kv_heads);
    let row = lid();
    let mut st = stream(cw, bk, held_q);
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
    flash_body(
        &mut st, q, k, values, out, &hd, keys, slice, slices, cw, w, bk, held_q, k_form, v_form,
        out_form,
    );
}

/// Ragged rows: a pass per slot in the block.
#[kernel]
pub(crate) fn flash_rows(
    q: Buf<F32>,
    #[form(k_form)] k: Buf<Dyn>,
    #[form(v_form)] values: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    queries: u32,
    keys: u32,
    heads: u32,
    kv_heads: u32,
    #[build] w: u32,
    #[build] bk: u32,
    #[build] held_q: bool,
    #[build] k_form: &'static dyn Stored,
    #[build] v_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    let simd = lid() / LANES as u32;
    let hd = head_at(gid() / THREADS, queries, heads, kv_heads);
    let row = lid();
    // A pass serves one slot's rows, up to their positions.
    let mut st = stream(w, bk, held_q);
    if lid() < BQ {
        st.from_s[row] = 0;
    }
    stage_q(&mut st.q_s, q, &hd, 0, w, w);
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
            restart(&mut st);
            st.o00.zero();
            st.o01.zero();
            st.o10.zero();
            st.o11.zero();
            for b in 0..(last + bk - 1) / bk {
                let k0 = b * bk;
                let at = kept_row(slot, k0);
                stage_kv::<false>(&mut st.kv_s, k, &hd, at, k0, last, 0, w, w, bk, k_form);
                barrier();
                st.s_t.zero();
                score_chunk(&mut st, simd, w, bk, held_q);
                take_block(
                    &mut st, values, &hd, at, k0, last, 0, w, w, bk, held_q, v_form,
                );
            }
            write_all::<false>(&mut st, out, &hd, 0, w, w, out_form);
            barrier();
        }
    }
}

/// Reads scaled queries, and keys and values apart.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = way_name(self.1, self.2),
    order = Order::PerConfig,
    sizes = self.0.sizes(),
    work = self.0.seqs * self.0.heads * self.0.queries * self.0.seen() * self.0.width,
    needs = flash_needs(&self.0, self.1, self.2),
    accepts = flash_accepts,
    emits = flash_emits,
    source = flash_source,
    launches = flash_launches,
    load = flash_load,
    load_key = tessera_lower::load_key((self.0.load_key(), self.1, self.2)),
    unproven,
)]
pub(crate) struct Flash(pub Attn, pub u32, pub bool);

pub(crate) fn kv_accepts(a: &Attn, input: usize, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == a.k_form.name(),
        2 => form.name() == a.v_form.name(),
        _ => !form.narrow(),
    }
}

fn flash_accepts(k: &Flash, input: usize, form: &dyn Stored) -> bool {
    kv_accepts(&k.0, input, form)
}

fn flash_emits(k: &Flash, form: &dyn Stored) -> bool {
    form.name() == k.0.out_form.name()
}

/// The kernel name of a way.
pub(crate) fn way_name(bk: u32, held: bool) -> &'static str {
    match (bk, held) {
        (16, false) => "flash",
        (16, true) => "flash_held",
        (_, false) => "flash_k32",
        (_, true) => "flash_k32_held",
    }
}

impl Flash {
    fn tree(&self) -> Body {
        let a = &self.0;
        let (w, bk, held) = (a.width, self.1, self.2);
        let key = (
            w as u32,
            false,
            bk,
            held,
            a.k_form.name(),
            a.v_form.name(),
            a.out_form.name(),
        );
        TREES.with(|t| {
            t.borrow_mut()
                .entry(key)
                .or_insert_with(|| {
                    flash_ir(
                        chunk(w) as u32,
                        w as u32,
                        bk,
                        held,
                        a.k_form,
                        a.v_form,
                        a.out_form,
                    )
                })
                .clone()
        })
    }

    /// A group per head, block of queries and column slice.
    fn groups(&self) -> usize {
        let slices = padded(self.0.width) / chunk(self.0.width);
        self.0.heads * self.0.queries.div_ceil(BQ as usize) * slices
    }
}

fn flash_load(k: &Flash, len: usize) -> Option<Load> {
    let a = &k.0;
    let (keys, side) = (a.seen() as f64, (a.kv_heads * a.width) as f64);
    // Each query block and slice reads the keys and values.
    let slices = (padded(a.width) / chunk(a.width)) as f64;
    let blocks = a.queries.div_ceil(BQ as usize) as f64 * slices;
    let k_seq = a.k_form.element_bytes() * keys * side * blocks;
    let v_seq = a.v_form.element_bytes() * keys * side * blocks;
    Some(Load {
        threads: a.seqs * k.groups() * THREADS as usize,
        group: THREADS as usize,
        trips: keys / k.1 as f64 * slices,
        trip: k.tree().trip(),
        reads: vec![
            4.0 * len as f64,
            k_seq * a.seqs as f64,
            v_seq * a.seqs as f64,
        ],
        written: a.out_form.bytes(len) as f64,
        flight: 16.0,
        ..Load::default()
    })
}

pub(crate) fn form_suffix(a: &Attn) -> String {
    match (a.k_form.name(), a.v_form.name(), a.out_form.name()) {
        ("f32", "f32", "f32") => String::new(),
        (kf, vf, of) => format!("_{kf}_{vf}_{of}"),
    }
}

fn flash_source(k: &Flash) -> Source {
    let way = &way_name(k.1, k.2)["flash".len()..];
    let suffix = form_suffix(&k.0);
    Source::named(k.tree(), format!("flash_{}{way}{suffix}", k.0.width))
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

/// Floats of `stream`: Q, K and V, P, row arrays.
pub(crate) fn stream_floats(w: usize, bk: u32, held: bool) -> usize {
    let staged: u32 = staged(w as u32, bk, held).iter().sum();
    (staged + BQ * (bk + PAD) + 3 * BQ) as usize
}

/// Q and K chunks, P, sums, and the seen ranges.
pub(crate) fn flash_needs(a: &Attn, bk: u32, held: bool) -> Needs {
    let floats = stream_floats(chunk(a.width), bk, held) + BQ as usize;
    Needs {
        lanes: LANES,
        group_bytes: 4 * floats,
        group_threads: THREADS as usize,
        tile: true,
        ..Needs::default()
    }
}

/// Checks head width against tiled flash limits.
pub(crate) fn check_width(w: usize) -> Result<(), String> {
    if !w.is_multiple_of(WIDE as usize) {
        return Err(format!("head width {w} is not a multiple of {WIDE}"));
    }
    if w > WIDEST {
        return Err(format!("head width {w} exceeds max {WIDEST}"));
    }
    if w > CW as usize && !w.is_multiple_of(CW as usize) {
        return Err(format!("head width {w} is not a multiple of {CW}"));
    }
    Ok(())
}

/// Checks whether `site` satisfies flash attention shape and mask rules.
fn check_flash(site: &sites::Attend) -> Result<(), String> {
    let w = site.sizes.width;
    check_width(w)?;
    if site.rows.is_some() {
        if !w.is_multiple_of(16) {
            return Err(format!("ragged head width {w} is not a multiple of 16"));
        }
        if site.sizes.mask != Mask::Causal {
            return Err("ragged flash requires causal mask".into());
        }
        if w > CW as usize {
            return Err(format!("head width {w} exceeds ragged limit {CW}"));
        }
        // Keys stage a block at a time, in one run.
        let run = site.sizes.kept.run(site.sizes.context);
        if !run.is_multiple_of(WAYS[0].0 as usize) {
            return Err(format!("kept run {run} is not a multiple of {}", WAYS[0].0));
        }
    }
    Ok(())
}

/// Scale the queries, then attend by tiles.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Attend = tree.site(at)?;
    if site.packed.is_some() {
        return None;
    }
    check_flash(&site).ok()?;
    let w = site.sizes.width;
    if site.rows.is_some() {
        return site.ragged(crate::flash_rows::FlashRows(site.sizes));
    }
    let want_bk = tree.knobs.get("flash", "bk").map(|v| v as u32);
    if want_bk.is_some_and(|bk| bk != 16 && bk != 32) {
        return None;
    }
    let want_held = tree.knobs.get("flash", "held").map(|v| v != 0);
    // Each way: scaling, then flash. Held queries need one slice.
    let mut drafts = Vec::new();
    let ways = WAYS.into_iter().filter(|&(bk, held)| {
        (!held || w <= CW as usize)
            && want_bk.is_none_or(|b| b == bk)
            && want_held.is_none_or(|h| h == held)
    });
    for (way, (bk, held)) in ways.enumerate() {
        let (scaled, out) = crate::scaled(&site)?;
        let reads = [Read::Draft(0), Read::Node(site.k), Read::Node(site.v)];
        drafts.push(scaled.choice(way));
        drafts.push(Draft::new(Flash(site.sizes, bk, held), reads, out).choice(way));
    }
    (!drafts.is_empty()).then_some(drafts)
}

fn why_not(tree: &Tree, at: T) -> Option<String> {
    let site: sites::Attend = tree.site(at)?;
    if site.packed.is_some() {
        return None;
    }
    check_flash(&site).err()
}

tessera_lower::rule!("flash", propose, why = why_not);
