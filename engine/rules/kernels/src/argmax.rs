//! Each row's largest entries, best first.

use tessera_codegen::prelude::*;
use tessera_lower::sites;
use tessera_lower::{Draft, Kernel, Read, T, Tree};

/// The most threads that share one row's scan.
const PARTS: usize = 256;
/// Row slices for two-stage top-k to keep merge work small.
const TOP_PARTS: usize = 32;

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

/// Top `k` indices inside each slice of each row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(part_top, threads = self.rows * self.parts, work = self.rows * self.width * self.k)]
pub(crate) struct PartTop {
    pub rows: usize,
    pub width: usize,
    pub parts: usize,
    pub k: usize,
}

/// Merges per-slice top `k` candidates into row top `k`.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(merge_top, threads = self.rows, work = self.rows * self.parts * self.k * self.k)]
pub(crate) struct MergeTop {
    pub rows: usize,
    pub width: usize,
    pub parts: usize,
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

/// Top `k` indices per row slice, lowest index wins ties.
#[kernel]
fn part_top(x: Buf<F32>, mut out: BufMut<U32>, rows: u32, width: u32, parts: u32, k: u32) {
    if gid() >= rows * parts {
        return;
    }
    let row = gid() / parts;
    let part = gid() % parts;
    let at = row as u64 * width as u64;
    let dst = (gid() * k) as u64;
    let mut cap = f32::INFINITY;
    let mut prev = 0u32;
    for p in 0..k {
        let mut arg = width;
        let mut best = f32::NEG_INFINITY;
        for i in (part..width).by(parts) {
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

/// Merges slice candidates into each row's top `k` indices.
#[kernel]
fn merge_top(
    x: Buf<F32>,
    cands: Buf<U32>,
    mut out: BufMut<U32>,
    rows: u32,
    width: u32,
    parts: u32,
    k: u32,
) {
    let row = gid();
    if row >= rows {
        return;
    }
    let at = row as u64 * width as u64;
    let base = (row * parts * k) as u64;
    let dst = (row * k) as u64;
    let total = parts * k;
    let mut cap = f32::INFINITY;
    let mut prev = 0u32;
    for p in 0..k {
        let mut arg = width;
        let mut best = f32::NEG_INFINITY;
        for m in 0..total {
            let i = cands[base + m as u64];
            if i < width {
                let v = x[at + i as u64];
                let under = v < cap || (v == cap && i > prev);
                if v > f32::NEG_INFINITY && under && (v > best || (v == best && i < arg)) {
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

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Top { x, rows, width, k } = tree.site(at)?;
    if k > 1 {
        if width > PARTS * 2 && k <= 32 {
            let parts = TOP_PARTS.min(width).max(1);
            let both = [Read::Node(x), Read::Draft(0)];
            return Some(vec![
                Draft::new(
                    PartTop {
                        rows,
                        width,
                        parts,
                        k,
                    },
                    [x],
                    rows * parts * k,
                ),
                Draft::new(
                    MergeTop {
                        rows,
                        width,
                        parts,
                        k,
                    },
                    both,
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
