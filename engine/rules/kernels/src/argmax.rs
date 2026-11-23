//! Each row's largest entries, best first.

use tessera_codegen::prelude::*;
use tessera_lower::sites;
use tessera_lower::{Draft, Kernel, Read, T, Tree};

/// The most threads that share one row's scan.
const PARTS: usize = 256;

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
#[launch(argmax, threads = self.rows, work = self.rows * self.parts)]
pub(crate) struct Argmax {
    pub rows: usize,
    pub width: usize,
    pub parts: usize,
}

/// One thread per entry or pick, whichever is more.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(top_k, threads = self.rows * self.width.max(self.k), work = self.rows * self.width * self.width.max(self.k))]
pub(crate) struct TopK {
    pub rows: usize,
    pub width: usize,
    pub k: usize,
}

/// Top `k` indices inside each winning chunk of each row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(chunk_top, threads = self.rows * self.k, work = self.rows * self.width * self.k / self.parts.max(1))]
pub(crate) struct ChunkTop {
    pub rows: usize,
    pub width: usize,
    pub parts: usize,
    pub k: usize,
}

/// Selects top `k` from `total` distinct row candidates.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(merge_top, threads = self.rows * self.total, work = self.rows * self.total * self.total)]
pub(crate) struct MergeTop {
    pub rows: usize,
    pub width: usize,
    pub total: usize,
    pub k: usize,
}

/// Picks go by value, then index. Spare picks hold `width`.
#[kernel]
fn top_k(x: Buf<F32>, mut out: BufMut<U32>, rows: u32, width: u32, k: u32) {
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
        out[(row * k + place) as u64] = t;
    }
    if t >= finite && t < k {
        out[(row * k + t) as u64] = width;
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

/// A piece with nothing finite holds the width.
#[kernel]
fn argmax(x: Buf<F32>, peaks: Buf<U32>, mut out: BufMut<U32>, rows: u32, width: u32, parts: u32) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let mut arg = width;
    let mut best = f32::NEG_INFINITY;
    for p in 0..parts {
        let i = peaks[(gid() * parts + p) as u64];
        if i < width && x[at + i as u64] > best {
            best = x[at + i as u64];
            arg = i;
        }
    }
    out[gid() as u64] = arg;
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
    if gid() >= rows * k {
        return;
    }
    let row = gid() / k;
    let win = winners[gid() as u64];
    let dst = (gid() * k) as u64;
    if win >= width {
        for p in 0..k {
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
    let mut cap = f32::INFINITY;
    let mut prev = 0u32;
    for p in 0..k {
        let mut arg = width;
        let mut best = f32::NEG_INFINITY;
        for i in from..to {
            let v = x[at + i as u64];
            if v > f32::NEG_INFINITY {
                let under = v < cap || (v == cap && i > prev);
                if under && (v > best || (v == best && i < arg)) {
                    best = v;
                    arg = i;
                }
            }
        }
        out[dst + p as u64] = arg;
        cap = best;
        prev = arg;
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
) {
    if gid() >= rows * total {
        return;
    }
    let row = gid() / total;
    let t = gid() % total;
    let at = row as u64 * width as u64;
    let base = (row * total) as u64;
    let idx = cands[base + t as u64];
    let val = if idx < width {
        x[at + idx as u64]
    } else {
        f32::NEG_INFINITY
    };
    let mut rank = 0u32;
    let mut seen = 0u32;
    for m in 0..total {
        let j = cands[base + m as u64];
        let other = if j < width {
            x[at + j as u64]
        } else {
            f32::NEG_INFINITY
        };
        if other > f32::NEG_INFINITY {
            seen += 1;
        }
        if other > val || (other == val && j < idx) {
            rank += 1;
        }
    }
    if val > f32::NEG_INFINITY && rank < k {
        out[(row * k + rank) as u64] = idx;
    }
    if t >= seen && t < k {
        out[(row * k + t) as u64] = width;
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Top { x, rows, width, k } = tree.site(at)?;
    if k > 1 {
        if width > PARTS * 2 && k <= 32 {
            let parts = PARTS.min(width).max(k);
            return Some(vec![
                Draft::new(Peaks { rows, width, parts }, [x], rows * parts),
                Draft::new(
                    MergeTop {
                        rows,
                        width,
                        total: parts,
                        k,
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
                    },
                    [Read::Node(x), Read::Draft(2)],
                    rows * k,
                ),
            ]);
        }
        return Some(vec![Draft::new(TopK { rows, width, k }, [x], rows * k)]);
    }
    let parts = PARTS.min(width).max(1);
    let both = [Read::Node(x), Read::Draft(0)];
    Some(vec![
        Draft::new(Peaks { rows, width, parts }, [x], rows * parts),
        Draft::new(Argmax { rows, width, parts }, both, rows),
    ])
}

tessera_lower::rule!("argmax", propose);
