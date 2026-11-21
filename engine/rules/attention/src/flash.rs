//! Attention on tiles, the keys streamed in blocks.

use std::cell::RefCell;
use std::collections::HashMap;
use tessera_codegen::ir::Body;
use tessera_codegen::kept;
use tessera_codegen::prelude::*;
use tessera_graph::{Mask, Order};
use tessera_lower::sites;
use tessera_lower::{
    Arg, Attn, Draft, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Tree,
};

/// Queries one group handles.
const BQ: u32 = 32;

/// Keys per block, one way each. Metal fits 16.
const BKS: [u32; 2] = [16, 32];

/// Four SIMD groups. Scores take two per 16 keys.
const THREADS: u32 = 128;

/// Query rows each SIMD group softens.
const ROWS: u32 = BQ / (THREADS / LANES as u32);

/// Floats of padding after each staged row, against bank conflicts.
const PAD: u32 = 4;

/// Columns one group stages and owns at a time.
const CW: u32 = 128;

/// Floats one wide load brings.
const WIDE: u32 = 4;

/// Floats a thread holds while staging, at most.
const HELD: usize = (BKS[1] * CW / THREADS) as usize;

/// The widest head: four column slices.
const WIDEST: usize = 512;

/// Columns the kernel for `w` stages at a time.
fn chunk(w: usize) -> usize {
    w.min(CW as usize)
}

thread_local! {
    /// Each kernel tree, by width, raggedness and key block.
    static TREES: RefCell<HashMap<(u32, bool, u32), Body>> = RefCell::new(HashMap::new());
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
    /// Peaks, then sums, of the rows this SIMD group softens.
    state: [f32; 16],
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

/// The stream's memory and sums: `cw` columns, `bk` keys.
#[kernel]
fn stream(#[build] cw: u32, #[build] bk: u32) -> Stream {
    Stream {
        q_s: shared::<f32>(BQ * (cw + PAD)),
        kv_s: shared::<f32>((cw * (bk + PAD)).max(bk * (cw + PAD))),
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
    #[build] bk: u32,
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
        // Past `count` it reads 0. `zero()` leaves it unset.
        let mut v = F32x4::new(0.0, 0.0, 0.0, 0.0);
        if i < bk * (cw / WIDE) && k0 + t < count {
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
                    st.kv_s[(d + e) * (bk + PAD) + t] = v;
                } else {
                    st.kv_s[t * (cw + PAD) + d + e] = v;
                }
            }
        }
    }
}

/// Adds staged columns to each SIMD group's score tile.
#[kernel]
fn score_chunk(st: &mut Stream, simd: u32, #[build] cw: u32, #[build] bk: u32) {
    if simd < bk / 8 {
        let a = (simd % 2) * 16 * (cw + PAD);
        let b = (simd / 2) * 16 * (cw + PAD);
        st.s_t.block(&st.q_s, a, &st.kv_s, b, cw + PAD, cw);
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
    #[build] bk: u32,
) {
    let simd = lid() / LANES as u32;
    scores(st, simd, k0, bk);
    barrier();
    soften(st, bk);
    barrier();
    stage_kv::<true>(st, values, hd, at, k0, count, d0, cw, w, bk);
    barrier();
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
fn write_all(
    st: &mut Stream,
    mut out: BufMut<F32>,
    hd: &Head,
    d0: u32,
    #[build] cw: u32,
    #[build] w: u32,
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
    #[build] bk: u32,
) {
    let simd = lid() / LANES as u32;
    // A group owns `cw` columns and scores the whole head.
    let slices = w / cw;
    let group = gid() / THREADS;
    let slice = group % slices;
    let hd = head_at(group / slices, queries, heads, kv_heads);
    let row = lid();
    let mut st = stream(cw, bk);
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
    restart(&mut st);
    // The slice's first column.
    let start = slice * cw;
    // One slice holds the whole head: its queries stay staged.
    if slices == 1 {
        stage_q(&mut st, q, &hd, 0, cw, w);
    }
    for b in lo / bk..(hi + bk - 1) / bk {
        let k0 = b * bk;
        st.s_t.zero();
        for c in 0..slices {
            let d0 = c * cw;
            if slices > 1 {
                stage_q(&mut st, q, &hd, d0, cw, w);
            }
            stage_kv::<false>(&mut st, k, &hd, k0, k0, keys, d0, cw, w, bk);
            barrier();
            score_chunk(&mut st, simd, cw, bk);
            barrier();
        }
        take_block(&mut st, values, &hd, k0, k0, keys, start, cw, w, bk);
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
    #[build] bk: u32,
) {
    let simd = lid() / LANES as u32;
    let hd = head_at(gid() / THREADS, queries, heads, kv_heads);
    let row = lid();
    // A pass serves one slot's rows, up to their positions.
    let mut st = stream(w, bk);
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
            restart(&mut st);
            st.o00.zero();
            st.o01.zero();
            st.o10.zero();
            st.o11.zero();
            for b in 0..(last + bk - 1) / bk {
                let k0 = b * bk;
                let at = kept_row(slot, k0);
                stage_kv::<false>(&mut st, k, &hd, at, k0, last, 0, w, w, bk);
                barrier();
                st.s_t.zero();
                score_chunk(&mut st, simd, w, bk);
                take_block(&mut st, values, &hd, at, k0, last, 0, w, w, bk);
            }
            write_all(&mut st, out, &hd, 0, w, w);
            barrier();
        }
    }
}

/// Reads scaled queries, and keys and values apart.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = match self.1 {
        16 => "flash",
        _ => "flash_k32",
    },
    order = Order::PerConfig,
    sizes = self.0.sizes(),
    work = self.0.seqs * self.0.heads * self.0.queries * self.0.seen() * self.0.width,
    needs = flash_needs(&self.0, self.1),
    source = flash_source,
    launches = flash_launches,
    load = flash_load,
    unproven,
)]
pub(crate) struct Flash(pub Attn, pub u32);

impl Flash {
    fn tree(&self) -> Body {
        let (w, bk) = (self.0.width, self.1);
        TREES.with(|t| {
            t.borrow_mut()
                .entry((w as u32, false, bk))
                .or_insert_with(|| flash_ir(chunk(w) as u32, w as u32, bk))
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
        trips: keys / k.1 as f64 * slices,
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
    let entry = match k.1 {
        16 => format!("flash_{}", k.0.width),
        bk => format!("flash_{}_k{bk}", k.0.width),
    };
    Source::named(k.tree(), entry)
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

/// Floats of `stream`: Q, K or V, P, row arrays.
fn stream_floats(w: usize, bk: usize) -> usize {
    let (bq, pad) = (BQ as usize, PAD as usize);
    bq * (w + pad) + (w * (bk + pad)).max(bk * (w + pad)) + bq * (bk + pad) + 3 * bq
}

/// Q and K chunks, P, sums, and the seen ranges.
fn flash_needs(a: &Attn, bk: u32) -> Needs {
    let floats = stream_floats(chunk(a.width), bk as usize) + BQ as usize;
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
                .entry((w, true, BKS[0]))
                .or_insert_with(|| flash_rows_ir(w, BKS[0]))
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
        trips: passes * keys / BKS[0] as f64,
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
    let floats = stream_floats(a.width, BKS[0] as usize);
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
        if !run.is_multiple_of(BKS[0] as usize) {
            return None;
        }
        return site.ragged(FlashRows(site.sizes));
    }
    // Each key block is a way: scaling, then flash.
    let mut drafts = Vec::new();
    for (way, bk) in BKS.into_iter().enumerate() {
        let (scaled, out) = crate::scaled(&site)?;
        let reads = [Read::Draft(0), Read::Node(site.k), Read::Node(site.v)];
        drafts.push(scaled.choice(way));
        drafts.push(Draft::new(Flash(site.sizes, bk), reads, out).choice(way));
    }
    Some(drafts)
}

tessera_lower::rule!("flash", propose);
