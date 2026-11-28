//! Each row's largest entries, best first.

use tessera_codegen::prelude::*;
use tessera_lower::{Draft, Kernel, Knobs, Launch, Needs, Read, Shape, Step, T, Tree, knob, sites};

use crate::radix::{HIST_BINS, RADIX_MAX_K, RADIX_THREADS, RadixTopK};

/// The most threads that share one row's scan.
const PARTS: usize = 512;

/// Threads per row in the final argmax reduction.
const ARGMAX_THREADS: u32 = 256;

/// Threads per winning chunk in parallel `chunk_top`.
const CHUNK_THREADS: u32 = 128;

/// Threads per row in shared-memory `merge_top`.
const MERGE_THREADS: u32 = 256;

/// Maximum candidate list cached in shared memory by `merge_top`.
const MERGE_MAX: u32 = 1024;

knob!("argmax", "parts", 1, 4096);
knob!("argmax", "chunk_min_width", 1, 1_048_576);
knob!("argmax", "chunk_max_k", 1, 64);
knob!("argmax", "radix_min_k", 2, 8192);
knob!("argmax", "argmax_threads", 32, 1024);
knob!("argmax", "chunk_threads", 32, 1024);
knob!("argmax", "merge_threads", 32, 1024);
knob!("argmax", "merge_max", 32, 4096);
knob!("argmax", "radix_threads", 64, 1024);
knob!("argmax", "cache_keys", 0, 1);

#[derive(Clone, Copy)]
struct ArgmaxCfg {
    parts: usize,
    chunk_min_width: usize,
    chunk_max_k: usize,
    radix_min_k: usize,
    argmax_threads: u32,
    chunk_threads: u32,
    merge_threads: u32,
    merge_max: u32,
    radix_threads: u32,
    cache_keys: Option<bool>,
}

impl ArgmaxCfg {
    fn of(knobs: &Knobs) -> Option<Self> {
        let get = |name, default| knobs.get("argmax", name).unwrap_or(default);
        let cfg = Self {
            parts: get("parts", PARTS),
            chunk_min_width: get("chunk_min_width", 512),
            chunk_max_k: get("chunk_max_k", 32),
            radix_min_k: get("radix_min_k", 33),
            argmax_threads: get("argmax_threads", ARGMAX_THREADS as usize) as u32,
            chunk_threads: get("chunk_threads", CHUNK_THREADS as usize) as u32,
            merge_threads: get("merge_threads", MERGE_THREADS as usize) as u32,
            merge_max: get("merge_max", MERGE_MAX as usize) as u32,
            radix_threads: get("radix_threads", RADIX_THREADS as usize) as u32,
            cache_keys: knobs.get("argmax", "cache_keys").map(|v| v != 0),
        };
        let pow2 = cfg.argmax_threads.is_power_of_two()
            && cfg.chunk_threads.is_power_of_two()
            && cfg.radix_threads.is_power_of_two()
            && HIST_BINS.is_multiple_of(cfg.radix_threads);
        pow2.then_some(cfg)
    }
}

/// The best index in each piece of each row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(peaks, threads = self.rows * self.parts, work = self.rows * self.width)]
pub(crate) struct Peaks {
    pub rows: usize,
    pub width: usize,
    pub parts: usize,
}

/// The best of each row's peaks.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    argmax(self.threads, self.threads.trailing_zeros(), self.threads / 2),
    sizes = [self.rows, self.width, self.parts],
    work = self.rows * self.parts,
    needs = group_needs(8 * self.threads as usize, self.threads),
    launches = argmax_launches,
)]
pub(crate) struct Argmax {
    pub rows: usize,
    pub width: usize,
    pub parts: usize,
    pub first: usize,
    pub whole: usize,
    pub threads: u32,
}

/// Group memory of `bytes` for `threads`.
pub(super) fn group_needs(bytes: usize, threads: u32) -> Needs {
    Needs {
        group_bytes: bytes,
        group_threads: threads as usize,
        ..Needs::default()
    }
}

/// One launch of `across` blocks of `threads`.
pub(super) fn grid_launch(
    step: &Step,
    sizes: &[usize],
    across: usize,
    threads: u32,
) -> Vec<Launch> {
    let shape = Shape::Grid {
        across,
        down: 1,
        threads: threads as usize,
    };
    vec![Launch::plain(step, sizes, shape)]
}

fn argmax_launches(k: &Argmax, step: &Step) -> Vec<Launch> {
    grid_launch(
        step,
        &[k.rows, k.width, k.parts, k.first, k.whole],
        k.rows,
        k.threads,
    )
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
    chunk_top(self.threads, self.threads.trailing_zeros(), self.threads / 2),
    sizes = [self.rows, self.width, self.parts, self.k],
    work = self.rows * self.width * self.k / self.parts.max(1),
    needs = group_needs(8 * self.threads as usize, self.threads),
    launches = chunk_launches,
)]
pub(crate) struct ChunkTop {
    pub rows: usize,
    pub width: usize,
    pub parts: usize,
    pub k: usize,
    pub threads: u32,
}

fn chunk_launches(c: &ChunkTop, step: &Step) -> Vec<Launch> {
    grid_launch(
        step,
        &[c.rows, c.width, c.parts, c.k],
        c.rows * c.k,
        c.threads,
    )
}

/// Selects top `k` from `total` distinct row candidates.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    merge_top(self.threads, self.merge_max),
    sizes = [self.rows, self.width, self.total, self.k],
    work = self.rows * self.total * self.total,
    needs = group_needs(8 * self.merge_max as usize, self.threads),
    launches = merge_launches,
)]
pub(crate) struct MergeTop {
    pub rows: usize,
    pub width: usize,
    pub total: usize,
    pub k: usize,
    pub first: usize,
    pub whole: usize,
    pub threads: u32,
    pub merge_max: u32,
}

fn merge_launches(m: &MergeTop, step: &Step) -> Vec<Launch> {
    grid_launch(
        step,
        &[m.rows, m.width, m.total, m.k, m.first, m.whole],
        m.rows,
        m.threads,
    )
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

/// Best entry ends in slot 0. Lowest index wins ties.
#[kernel]
fn best_of(val_s: &mut Shared<f32>, idx_s: &mut Shared<u32>, steps: u32, half: u32) {
    let l = lid();
    for shift in 0..steps {
        let s = half >> shift;
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
    #[build] threads: u32,
    #[build] steps: u32,
    #[build] half: u32,
) {
    let row = gid() / threads;
    if row >= rows {
        return;
    }
    let l = lid();
    let at = row as u64 * width as u64;
    let mut arg = width;
    let mut best = f32::NEG_INFINITY;
    for p in (l..parts).by(threads) {
        let i = peaks[(row * parts + p) as u64];
        if i < width {
            let v = x[at + i as u64];
            if v > best || (v == best && i < arg) {
                best = v;
                arg = i;
            }
        }
    }
    let mut val_s = shared::<f32>(threads);
    let mut idx_s = shared::<u32>(threads);
    val_s[l] = best;
    idx_s[l] = arg;
    barrier();
    best_of(&mut val_s, &mut idx_s, steps, half);
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
    #[build] threads: u32,
    #[build] steps: u32,
    #[build] half: u32,
) {
    let item = gid() / threads;
    if item >= rows * k {
        return;
    }
    let l = lid();
    let row = item / k;
    let win = winners[item as u64];
    let dst = (item * k) as u64;
    if win >= width {
        for p in (l..k).by(threads) {
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
    let mut val_s = shared::<f32>(threads);
    let mut idx_s = shared::<u32>(threads);
    if l == 0 {
        out[dst] = win;
    }
    let mut cap = x[at + win as u64];
    let mut prev = win;
    for p in 1..k {
        let mut arg = width;
        let mut best = f32::NEG_INFINITY;
        for i in (from + l..to).by(threads) {
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
        best_of(&mut val_s, &mut idx_s, steps, half);
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
    #[build] threads: u32,
    #[build] merge_max: u32,
) {
    let row = gid() / threads;
    if row >= rows {
        return;
    }
    let l = lid();
    let at = row as u64 * width as u64;
    let base = (row * total) as u64;
    let mut vals = shared::<f32>(merge_max);
    let mut idxs = shared::<u32>(merge_max);
    for m in (l..total).by(threads) {
        let idx = cands[base + m as u64];
        idxs[m] = idx;
        vals[m] = if idx < width {
            x[at + idx as u64]
        } else {
            f32::NEG_INFINITY
        };
    }
    barrier();
    for t in (l..total).by(threads) {
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

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Top {
        x,
        rows,
        width,
        k,
        first,
        whole,
    } = tree.site(at)?;
    let cfg = ArgmaxCfg::of(&tree.knobs)?;
    let parts = cfg.parts.min(width);
    if k > 1 {
        let can_chunk = width > cfg.chunk_min_width
            && k <= cfg.chunk_max_k
            && k * k <= cfg.merge_max as usize
            && parts >= k
            && parts <= cfg.merge_max as usize;
        if can_chunk {
            let merge = |total, first, whole, src| {
                Draft::new(
                    MergeTop {
                        rows,
                        width,
                        total,
                        k,
                        first,
                        whole,
                        threads: cfg.merge_threads,
                        merge_max: cfg.merge_max,
                    },
                    [Read::Node(x), Read::Draft(src)],
                    rows * k,
                )
            };
            return Some(vec![
                Draft::new(Peaks { rows, width, parts }, [x], rows * parts),
                merge(parts, 0, width, 0),
                Draft::new(
                    ChunkTop {
                        rows,
                        width,
                        parts,
                        k,
                        threads: cfg.chunk_threads,
                    },
                    [Read::Node(x), Read::Draft(1)],
                    rows * k * k,
                ),
                merge(k * k, first, whole, 2),
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
        if k >= cfg.radix_min_k && k <= RADIX_MAX_K {
            drafts.push(
                Draft::new(
                    RadixTopK {
                        rows,
                        width,
                        k,
                        first,
                        whole,
                        threads: cfg.radix_threads,
                        cache_keys: cfg.cache_keys.unwrap_or(k <= 2048),
                    },
                    [x],
                    rows * k,
                )
                .choice(1),
            );
        }
        return Some(drafts);
    }
    let parts = parts.max(1);
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
                threads: cfg.argmax_threads,
            },
            both,
            rows,
        ),
    ])
}

tessera_lower::rule!("argmax", propose);
