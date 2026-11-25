//! Each row's largest entries, best first.

use tessera_codegen::ir::Body;
use tessera_codegen::prelude::*;
use tessera_lower::sites;
use tessera_lower::{Draft, Kernel, Launch, Needs, Read, Shape, Source, Step, T, Tree};

/// The most threads that share one row's scan.
const PARTS: usize = 512;

/// Threads per row in the radix selector.
const RADIX_THREADS: u32 = 256;

/// Four-bit radix buckets per histogram pass.
const BINS: u32 = 16;

/// Warps per radix block.
const WARPS: u32 = RADIX_THREADS / LANES as u32;

/// Elements per thread in each collect tile.
const TILE_ELEMS: u32 = 4;

/// Maximum `k` handled by the radix selector.
const RADIX_MAX_K: usize = 8192;

/// Shared pool slots for `k <= 4096`.
const SLOTS_SMALL: u32 = 4096;

/// Shared pool slots for `4096 < k <= 8192`.
const SLOTS_LARGE: u32 = 8192;

/// Scan scratch entries: per-thread plus bin totals.
const SCAN_LEN: u32 = RADIX_THREADS + BINS;

/// The best index in each piece of each row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(peaks, threads = self.rows * self.parts, work = self.rows * self.width)]
pub(crate) struct Peaks {
    pub rows: usize,
    pub width: usize,
    pub parts: usize,
}

/// Threads per row in the final argmax reduction.
const ARGMAX_THREADS: u32 = 256;

/// Threads per winning chunk in parallel `chunk_top`.
const CHUNK_THREADS: u32 = 128;

/// Threads per row in shared-memory `merge_top`.
const MERGE_THREADS: u32 = 256;

/// Maximum candidate list cached in shared memory by `merge_top`.
const MERGE_MAX: u32 = 1024;

/// The best of each row's peaks.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    argmax,
    sizes = [self.rows, self.width, self.parts],
    work = self.rows * self.parts,
    needs = argmax_needs(),
    launches = argmax_launches,
)]
pub(crate) struct Argmax {
    pub rows: usize,
    pub width: usize,
    pub parts: usize,
    pub first: usize,
    pub whole: usize,
}

fn argmax_needs() -> Needs {
    Needs {
        group_bytes: 8 * ARGMAX_THREADS as usize,
        group_threads: ARGMAX_THREADS as usize,
        ..Needs::default()
    }
}

fn argmax_launches(k: &Argmax, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: k.rows,
        down: 1,
        threads: ARGMAX_THREADS as usize,
    };
    vec![Launch::plain(
        step,
        &[k.rows, k.width, k.parts, k.first, k.whole],
        shape,
    )]
}

/// One thread per entry or pick, whichever is more.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    top_k,
    threads = self.rows * self.width.max(self.k),
    work = self.rows.saturating_mul(self.width).saturating_mul(self.width.max(self.k)),
)]
pub(crate) struct TopK {
    pub rows: usize,
    pub width: usize,
    pub k: usize,
    pub first: usize,
    pub whole: usize,
}

/// Top `k` indices inside each winning chunk of each row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    chunk_top,
    sizes = [self.rows, self.width, self.parts, self.k],
    work = self.rows * self.width * self.k / self.parts.max(1),
    needs = chunk_needs(),
    launches = chunk_launches,
)]
pub(crate) struct ChunkTop {
    pub rows: usize,
    pub width: usize,
    pub parts: usize,
    pub k: usize,
}

fn chunk_needs() -> Needs {
    Needs {
        group_bytes: 8 * CHUNK_THREADS as usize,
        group_threads: CHUNK_THREADS as usize,
        ..Needs::default()
    }
}

fn chunk_launches(c: &ChunkTop, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: c.rows * c.k,
        down: 1,
        threads: CHUNK_THREADS as usize,
    };
    vec![Launch::plain(step, &[c.rows, c.width, c.parts, c.k], shape)]
}

/// Selects top `k` from `total` distinct row candidates.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    merge_top,
    sizes = [self.rows, self.width, self.total, self.k],
    work = self.rows * self.total * self.total,
    needs = merge_needs(),
    launches = merge_launches,
)]
pub(crate) struct MergeTop {
    pub rows: usize,
    pub width: usize,
    pub total: usize,
    pub k: usize,
    pub first: usize,
    pub whole: usize,
}

fn merge_needs() -> Needs {
    Needs {
        group_bytes: 8 * MERGE_MAX as usize,
        group_threads: MERGE_THREADS as usize,
        ..Needs::default()
    }
}

fn merge_launches(m: &MergeTop, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: m.rows,
        down: 1,
        threads: MERGE_THREADS as usize,
    };
    vec![Launch::plain(
        step,
        &[m.rows, m.width, m.total, m.k, m.first, m.whole],
        shape,
    )]
}

/// Radix-select and bitonic-sort top `k`, one block per row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = radix_name(self.k),
    source = radix_source,
    launches = radix_launches,
    work = radix_work(self),
    needs = radix_needs(self),
)]
pub(crate) struct RadixTopK {
    pub rows: usize,
    pub width: usize,
    pub k: usize,
    pub first: usize,
    pub whole: usize,
}

fn radix_name(k: usize) -> &'static str {
    match k {
        0..=2048 => "radix_top_k",
        2049..=4096 => "radix_top_k_4096",
        _ => "radix_top_k_8192",
    }
}

fn radix_slots(k: usize) -> usize {
    match k {
        0..=4096 => SLOTS_SMALL as usize,
        _ => SLOTS_LARGE as usize,
    }
}

fn radix_body(k: &RadixTopK) -> Body {
    match k.k {
        0..=2048 => radix_top_k_ir::<SLOTS_SMALL, 11, true>(),
        2049..=4096 => radix_top_k_ir::<SLOTS_SMALL, 12, false>(),
        _ => radix_top_k_ir::<SLOTS_LARGE, 13, false>(),
    }
}

fn radix_source(k: &RadixTopK) -> Source {
    Source::named(radix_body(k), format!("plan_{}", k.name()))
}

fn radix_launches(k: &RadixTopK, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: k.rows,
        down: 1,
        threads: RADIX_THREADS as usize,
    };
    vec![Launch::plain(
        step,
        &[k.rows, k.width, k.k, k.first, k.whole],
        shape,
    )]
}

fn radix_needs(k: &RadixTopK) -> Needs {
    Needs {
        group_bytes: 4 * (radix_slots(k.k) + SCAN_LEN as usize),
        group_threads: RADIX_THREADS as usize,
        ..Needs::default()
    }
}

fn radix_work(k: &RadixTopK) -> usize {
    let passes: usize = if k.width > k.k { 9 } else { 1 };
    let cap = k.k.min(k.width).max(1).next_power_of_two();
    let stages = cap.trailing_zeros() as usize;
    let sort = cap * stages * (stages + 1) / 4;
    let per_row = passes
        .saturating_mul(k.width)
        .saturating_add(sort)
        .saturating_add(RADIX_THREADS as usize * 4);
    k.rows.saturating_mul(per_row)
}

/// Picks go by value, then index. Spare picks hold `whole`.
#[kernel]
fn top_k(x: Buf<F32>, mut out: BufMut<U32>, rows: u32, width: u32, k: u32, first: u32, whole: u32) {
    let span = if width > k { width } else { k };
    if gid() >= rows * span {
        return;
    }
    let row = gid() / span;
    let t = gid() % span;
    let at = row as u64 * width as u64;
    let v = if t < width {
        x[at + t as u64]
    } else {
        f32::NEG_INFINITY
    };
    // An entry's place counts the entries ahead of it.
    let mut place = 0u32;
    let mut finite = 0u32;
    for i in 0..width {
        let w = x[at + i as u64];
        if w > f32::NEG_INFINITY {
            finite += 1;
        }
        if w > v || (w == v && i < t) {
            place += 1;
        }
    }
    if v > f32::NEG_INFINITY && place < k {
        out[(row * k + place) as u64] = first + t;
    }
    if t >= finite && t < k {
        out[(row * k + t) as u64] = whole;
    }
}

/// Strictly greater, so the lowest index wins ties.
#[kernel]
fn peaks(x: Buf<F32>, mut out: BufMut<U32>, rows: u32, width: u32, parts: u32) {
    if gid() >= rows * parts {
        return;
    }
    let span = (width + parts - 1) / parts;
    let from = (gid() % parts) * span;
    let to = if from + span < width {
        from + span
    } else {
        width
    };
    let at = (gid() / parts) as u64 * width as u64;
    let mut arg = width;
    let mut best = f32::NEG_INFINITY;
    for i in from..to {
        if x[at + i as u64] > best {
            best = x[at + i as u64];
            arg = i;
        }
    }
    out[gid() as u64] = arg;
}

/// A piece with nothing finite holds `whole`.
#[kernel]
fn argmax(
    x: Buf<F32>,
    peaks: Buf<U32>,
    mut out: BufMut<U32>,
    rows: u32,
    width: u32,
    parts: u32,
    first: u32,
    whole: u32,
) {
    let row = gid() / ARGMAX_THREADS;
    if row >= rows {
        return;
    }
    let l = lid();
    let at = row as u64 * width as u64;
    let mut arg = width;
    let mut best = f32::NEG_INFINITY;
    for p in (l..parts).by(ARGMAX_THREADS) {
        let i = peaks[(row * parts + p) as u64];
        if i < width {
            let v = x[at + i as u64];
            if v > best || (v == best && i < arg) {
                best = v;
                arg = i;
            }
        }
    }
    let mut val_s = shared::<f32>(ARGMAX_THREADS);
    let mut idx_s = shared::<u32>(ARGMAX_THREADS);
    val_s[l] = best;
    idx_s[l] = arg;
    barrier();
    for shift in 0..8 {
        let s = 128u32 >> shift;
        if l < s {
            let ov = val_s[l + s];
            let oi = idx_s[l + s];
            let cv = val_s[l];
            let ci = idx_s[l];
            if ov > cv || (ov == cv && oi < ci) {
                val_s[l] = ov;
                idx_s[l] = oi;
            }
        }
        barrier();
    }
    if l == 0 {
        let win = idx_s[0];
        out[row as u64] = if win < width { first + win } else { whole };
    }
}

/// Top `k` indices inside each winning chunk.
#[kernel]
fn chunk_top(
    x: Buf<F32>,
    winners: Buf<U32>,
    mut out: BufMut<U32>,
    rows: u32,
    width: u32,
    parts: u32,
    k: u32,
) {
    let item = gid() / CHUNK_THREADS;
    if item >= rows * k {
        return;
    }
    let l = lid();
    let row = item / k;
    let win = winners[item as u64];
    let dst = (item * k) as u64;
    if win >= width {
        for p in (l..k).by(CHUNK_THREADS) {
            out[dst + p as u64] = width;
        }
        return;
    }
    let span = (width + parts - 1) / parts;
    let from = (win / span) * span;
    let to = if from + span < width {
        from + span
    } else {
        width
    };
    let at = row as u64 * width as u64;
    let mut val_s = shared::<f32>(CHUNK_THREADS);
    let mut idx_s = shared::<u32>(CHUNK_THREADS);
    if l == 0 {
        out[dst] = win;
    }
    let mut cap = x[at + win as u64];
    let mut prev = win;
    for p in 1..k {
        let mut arg = width;
        let mut best = f32::NEG_INFINITY;
        for i in (from + l..to).by(CHUNK_THREADS) {
            let v = x[at + i as u64];
            if v > f32::NEG_INFINITY {
                let under = v < cap || (v == cap && i > prev);
                if under && (v > best || (v == best && i < arg)) {
                    best = v;
                    arg = i;
                }
            }
        }
        val_s[l] = best;
        idx_s[l] = arg;
        barrier();
        for shift in 0..7 {
            let s = 64u32 >> shift;
            if l < s {
                let ov = val_s[l + s];
                let oi = idx_s[l + s];
                if ov > val_s[l] || (ov == val_s[l] && oi < idx_s[l]) {
                    val_s[l] = ov;
                    idx_s[l] = oi;
                }
            }
            barrier();
        }
        cap = val_s[0];
        prev = idx_s[0];
        if l == 0 {
            out[dst + p as u64] = prev;
        }
        barrier();
    }
}

/// Selects top `k` from `total` distinct row candidates.
#[kernel]
fn merge_top(
    x: Buf<F32>,
    cands: Buf<U32>,
    mut out: BufMut<U32>,
    rows: u32,
    width: u32,
    total: u32,
    k: u32,
    first: u32,
    whole: u32,
) {
    let row = gid() / MERGE_THREADS;
    if row >= rows {
        return;
    }
    let l = lid();
    let at = row as u64 * width as u64;
    let base = (row * total) as u64;
    let mut vals = shared::<f32>(MERGE_MAX);
    let mut idxs = shared::<u32>(MERGE_MAX);
    for m in (l..total).by(MERGE_THREADS) {
        let idx = cands[base + m as u64];
        idxs[m] = idx;
        vals[m] = if idx < width {
            x[at + idx as u64]
        } else {
            f32::NEG_INFINITY
        };
    }
    barrier();
    for t in (l..total).by(MERGE_THREADS) {
        let idx = idxs[t];
        let val = vals[t];
        let mut rank = 0u32;
        let mut seen = 0u32;
        for m in 0..total {
            let other = vals[m];
            let j = idxs[m];
            if other > f32::NEG_INFINITY {
                seen += 1;
            }
            if other > val || (other == val && j < idx) {
                rank += 1;
            }
        }
        if val > f32::NEG_INFINITY && rank < k {
            out[(row * k + rank) as u64] = first + idx;
        }
        if t >= seen && t < k {
            out[(row * k + t) as u64] = whole;
        }
    }
}

/// Order-preserving `u32` key for `v > -inf`, else `0`.
#[kernel]
fn float_key(v: f32) -> u32 {
    let b = v.to_bits();
    let mag = b & 0x7fff_ffffu32;
    let valid = mag <= 0x7f80_0000u32 && b != 0xff80_0000u32;
    let ord = if mag == 0u32 {
        0x8000_0000u32
    } else if (b & 0x8000_0000u32) != 0u32 {
        !b
    } else {
        b ^ 0x8000_0000u32
    };
    if valid { ord } else { 0u32 }
}

/// Radix-selects and bitonic-sorts top `k` indices per row.
#[kernel]
fn radix_top_k<const SLOTS: u32, const MAX_STAGES: u32, const CACHE_KEYS: bool>(
    x: Buf<F32>,
    mut out: BufMut<U32>,
    rows: u32,
    width: u32,
    k: u32,
    first: u32,
    whole: u32,
) {
    let row = gid() / RADIX_THREADS;
    if row >= rows {
        return;
    }
    let l = lid();
    let at = row as u64 * width as u64;
    let out_at = row as u64 * k as u64;
    let mut pool = shared::<u32>(SLOTS);
    let mut scan = shared::<u32>(SCAN_LEN);

    let mut strict_min = 1u32;
    let mut thresh = 0u32;
    let mut need_eq = 0u32;
    if width > k {
        let mut prefix = 0u32;
        let mut mask = 0u32;
        let mut need = k;
        let mut done = false;
        let passes = 8u32;
        for pass in 0..passes {
            if !done {
                let shift = 28u32 - pass * 4u32;
                for b in 0..BINS {
                    pool[b * RADIX_THREADS + l] = 0u32;
                }
                for i in (l..width).by(RADIX_THREADS) {
                    let key = float_key(x[at + i as u64]);
                    if key != 0u32 && (key & mask) == prefix {
                        let d = (key >> shift) & 15u32;
                        pool[d * RADIX_THREADS + l] += 1u32;
                    }
                }
            }
            barrier();
            if !done {
                let bin = l / 16u32;
                let part = l % 16u32;
                let base = bin * RADIX_THREADS + part * 16u32;
                let mut sum = 0u32;
                for j in 0..16u32 {
                    sum += pool[base + j];
                }
                scan[l] = sum;
            }
            barrier();
            if !done && l < BINS {
                let mut tot = 0u32;
                for p in 0..16u32 {
                    tot += scan[l * 16u32 + p];
                }
                scan[RADIX_THREADS + l] = tot;
            }
            barrier();
            if !done {
                if pass == 0u32 {
                    let mut finite = 0u32;
                    for b in 0..BINS {
                        finite += scan[RADIX_THREADS + b];
                    }
                    if finite <= k {
                        done = true;
                    }
                }
                if !done {
                    let shift = 28u32 - pass * 4u32;
                    let mut found = false;
                    for rev in 0..BINS {
                        if !found {
                            let b = 15u32 - rev;
                            let c = scan[RADIX_THREADS + b];
                            if need > c {
                                need -= c;
                            } else {
                                prefix = prefix | (b << shift);
                                mask = mask | (15u32 << shift);
                                if c == need {
                                    strict_min = if prefix > 0u32 { prefix } else { 1u32 };
                                    done = true;
                                }
                                found = true;
                            }
                        }
                    }
                }
            }
        }
        if !done {
            strict_min = prefix + 1u32;
            thresh = prefix;
            need_eq = need;
        }
    }

    let mut running_gt = 0u32;
    let mut running_eq = 0u32;
    let eq_base = k - need_eq;
    let tile = RADIX_THREADS * TILE_ELEMS;
    for base in (0..width).by(tile) {
        let t_base = base + l * TILE_ELEMS;
        let mut keys = [0u32; TILE_ELEMS as usize];
        let mut c_gt = 0u32;
        let mut c_eq = 0u32;
        for s in 0..TILE_ELEMS {
            let idx = t_base + s;
            let mut key = 0u32;
            if idx < width {
                key = float_key(x[at + idx as u64]);
            }
            keys[s as usize] = key;
            if key >= strict_min {
                c_gt += 1u32;
            } else if need_eq > 0u32 && key == thresh {
                c_eq += 1u32;
            }
        }
        scan[l] = c_gt | (c_eq << 16u32);
        barrier();
        if l < WARPS {
            let mut sum = 0u32;
            let w_base = l * LANES as u32;
            for j in 0..LANES as u32 {
                let v = scan[w_base + j];
                scan[w_base + j] = sum;
                sum += v;
            }
            scan[RADIX_THREADS + l] = sum;
        }
        barrier();
        let warp = l / LANES as u32;
        let mut w_pre = 0u32;
        let mut tile_tot = 0u32;
        for w in 0..WARPS {
            let wt = scan[RADIX_THREADS + w];
            if w < warp {
                w_pre += wt;
            }
            tile_tot += wt;
        }
        let pre = w_pre + scan[l];
        let mut pos_gt = running_gt + (pre & 0xffffu32);
        let mut pos_eq = running_eq + (pre >> 16u32);
        for s in 0..TILE_ELEMS {
            let idx = t_base + s;
            let key = keys[s as usize];
            if key >= strict_min {
                if pos_gt < k {
                    pool[pos_gt] = idx;
                    if CACHE_KEYS {
                        pool[2048u32 + pos_gt] = key;
                    }
                }
                pos_gt += 1u32;
            } else if need_eq > 0u32 && key == thresh {
                if pos_eq < need_eq {
                    out[out_at + (eq_base + pos_eq) as u64] = first + idx;
                }
                pos_eq += 1u32;
            }
        }
        running_gt += tile_tot & 0xffffu32;
        running_eq += tile_tot >> 16u32;
        barrier();
    }

    let above = running_gt;
    let mut cap = 1u32;
    let max_stages = MAX_STAGES;
    for _s in 0..max_stages {
        if cap < above {
            cap = cap * 2u32;
        }
    }
    for p in (above + l..cap).by(RADIX_THREADS) {
        pool[p] = width;
        if CACHE_KEYS {
            pool[2048u32 + p] = 0u32;
        }
    }
    barrier();

    let pairs = cap / 2u32;
    for stage in 0..max_stages {
        let size = 2u32 << stage;
        for sub in 0..stage + 1u32 {
            if size <= cap {
                let stride = 1u32 << (stage - sub);
                for pair in (l..pairs).by(RADIX_THREADS) {
                    let block = pair / stride;
                    let off = pair % stride;
                    let p = block * stride * 2u32 + off;
                    let q = p + stride;
                    let ip = pool[p];
                    let iq = pool[q];
                    let mut kp = 0u32;
                    let mut kq = 0u32;
                    if CACHE_KEYS {
                        kp = pool[2048u32 + p];
                        kq = pool[2048u32 + q];
                    } else {
                        if ip < width {
                            kp = float_key(x[at + ip as u64]);
                        }
                        if iq < width {
                            kq = float_key(x[at + iq as u64]);
                        }
                    }
                    let q_better = kq > kp || (kq == kp && iq < ip);
                    let dir = (p & size) == 0u32;
                    let swap = if dir { q_better } else { !q_better };
                    if swap {
                        pool[p] = iq;
                        pool[q] = ip;
                        if CACHE_KEYS {
                            pool[2048u32 + p] = kq;
                            pool[2048u32 + q] = kp;
                        }
                    }
                }
            }
            barrier();
        }
    }

    for p in (l..above).by(RADIX_THREADS) {
        out[out_at + p as u64] = first + pool[p];
    }
    if need_eq == 0u32 {
        for p in (above + l..k).by(RADIX_THREADS) {
            out[out_at + p as u64] = whole;
        }
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Top {
        x,
        rows,
        width,
        k,
        first,
        whole,
    } = tree.site(at)?;
    if k > 1 {
        if width > 512 && k <= 32 {
            let parts = PARTS.min(width / 2).max(k);
            return Some(vec![
                Draft::new(Peaks { rows, width, parts }, [x], rows * parts),
                Draft::new(
                    MergeTop {
                        rows,
                        width,
                        total: parts,
                        k,
                        first: 0,
                        whole: width,
                    },
                    [Read::Node(x), Read::Draft(0)],
                    rows * k,
                ),
                Draft::new(
                    ChunkTop {
                        rows,
                        width,
                        parts,
                        k,
                    },
                    [Read::Node(x), Read::Draft(1)],
                    rows * k * k,
                ),
                Draft::new(
                    MergeTop {
                        rows,
                        width,
                        total: k * k,
                        k,
                        first,
                        whole,
                    },
                    [Read::Node(x), Read::Draft(2)],
                    rows * k,
                ),
            ]);
        }
        let mut drafts = vec![Draft::new(
            TopK {
                rows,
                width,
                k,
                first,
                whole,
            },
            [x],
            rows * k,
        )];
        if k > 32 && k <= RADIX_MAX_K {
            drafts.push(
                Draft::new(
                    RadixTopK {
                        rows,
                        width,
                        k,
                        first,
                        whole,
                    },
                    [x],
                    rows * k,
                )
                .choice(1),
            );
        }
        return Some(drafts);
    }
    let parts = PARTS.min(width).max(1);
    let both = [Read::Node(x), Read::Draft(0)];
    Some(vec![
        Draft::new(Peaks { rows, width, parts }, [x], rows * parts),
        Draft::new(
            Argmax {
                rows,
                width,
                parts,
                first,
                whole,
            },
            both,
            rows,
        ),
    ])
}

tessera_lower::rule!("argmax", propose);
