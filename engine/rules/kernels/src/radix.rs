//! Radix-select top `k`, one block per row.

use crate::argmax::{grid_launch, group_needs};
use tessera_codegen::ir::Body;
use tessera_codegen::prelude::*;
use tessera_lower::{Kernel, Launch, Needs, Source, Step};

/// Threads per row in the radix selector.
pub(super) const RADIX_THREADS: u32 = 256;

/// Eleven-bit radix buckets per histogram pass.
pub(super) const HIST_BINS: u32 = 2048;

/// Elements per thread in each collect tile.
const TILE_ELEMS: u32 = 4;

/// Maximum `k` handled by the radix selector.
pub(super) const RADIX_MAX_K: usize = 8192;

/// Shared pool slots for `k <= 4096`.
const SLOTS_SMALL: u32 = 4096;

/// Shared pool slots for `4096 < k <= 8192`.
const SLOTS_LARGE: u32 = 8192;

/// Radix-select and bitonic-sort top `k`, one block per row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = radix_band(self.k).0,
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
    pub threads: u32,
    pub cache_keys: bool,
}

/// Kernel name, pool slots and sort stages for `k`.
fn radix_band(k: usize) -> (&'static str, u32, u32) {
    match k {
        0..=2048 => ("radix_top_k", SLOTS_SMALL, 11),
        2049..=4096 => ("radix_top_k_4096", SLOTS_SMALL, 12),
        _ => ("radix_top_k_8192", SLOTS_LARGE, 13),
    }
}

fn radix_body(k: &RadixTopK) -> Body {
    let (_, slots, stages) = radix_band(k.k);
    let warps = k.threads / LANES as u32;
    let hist_step = HIST_BINS / k.threads;
    let scan_len = k.threads + warps;
    let cache_keys = k.cache_keys && k.k <= 2048;
    radix_top_k_ir(
        slots, stages, cache_keys, k.threads, warps, hist_step, scan_len,
    )
}

fn radix_source(k: &RadixTopK) -> Source {
    Source::named(radix_body(k), format!("plan_{}", k.name()))
}

fn radix_launches(k: &RadixTopK, step: &Step) -> Vec<Launch> {
    grid_launch(
        step,
        &[k.rows, k.width, k.k, k.first, k.whole],
        k.rows,
        k.threads,
    )
}

fn radix_needs(k: &RadixTopK) -> Needs {
    let warps = k.threads as usize / LANES;
    let scan = k.threads as usize + warps;
    let words = radix_band(k.k).1 as usize + HIST_BINS as usize + scan;
    group_needs(4 * words, k.threads)
}

fn radix_work(k: &RadixTopK) -> usize {
    let passes: usize = if k.width > k.k { 4 } else { 1 };
    let cap = k.k.min(k.width).max(1).next_power_of_two();
    let stages = cap.trailing_zeros() as usize;
    let sort = cap * stages * (stages + 1) / 4;
    let per_row = passes
        .saturating_mul(k.width)
        .saturating_add(sort)
        .saturating_add(k.threads as usize * 4);
    k.rows.saturating_mul(per_row)
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
#[allow(clippy::too_many_arguments)]
fn radix_top_k(
    x: Buf<F32>,
    mut out: BufMut<U32>,
    rows: u32,
    width: u32,
    k: u32,
    first: u32,
    whole: u32,
    #[build] slots: u32,
    #[build] max_stages: u32,
    #[build] cache_keys: bool,
    #[build] threads: u32,
    #[build] warps: u32,
    #[build] hist_step: u32,
    #[build] scan_len: u32,
) {
    let row = gid() / threads;
    if row >= rows {
        return;
    }
    let l = lid();
    let at = row as u64 * width as u64;
    let out_at = row as u64 * k as u64;
    let mut pool = shared::<u32>(slots);
    let mut hist = atomic_shared(HIST_BINS);
    let mut scan = shared::<u32>(scan_len);

    let mut strict_min = 1u32;
    let mut thresh = 0u32;
    let mut need_eq = 0u32;
    if width > k {
        let mut prefix = 0u32;
        let mut mask = 0u32;
        let mut need = k;
        let mut done = false;
        let passes = 3u32;
        for pass in 0..passes {
            let shift = if pass == 0u32 {
                21u32
            } else if pass == 1u32 {
                10u32
            } else {
                0u32
            };
            let bmask = if pass < 2u32 { 2047u32 } else { 1023u32 };
            if !done {
                for j in (l..HIST_BINS).by(threads) {
                    hist[j] = 0u32;
                }
            }
            barrier();
            if !done {
                let mut run_b = HIST_BINS;
                let mut run_c = 0u32;
                for i in (l..width).by(threads) {
                    let key = float_key(x[at + i as u64]);
                    if key != 0u32 && (key & mask) == prefix {
                        let d = (key >> shift) & bmask;
                        if d == run_b {
                            run_c += 1u32;
                        } else {
                            if run_c > 0u32 {
                                hist.add(run_b, run_c);
                            }
                            run_b = d;
                            run_c = 1u32;
                        }
                    }
                }
                if run_c > 0u32 {
                    hist.add(run_b, run_c);
                }
            }
            barrier();
            if !done {
                let base = l * hist_step;
                let mut sum = 0u32;
                for j in 0..hist_step {
                    sum += hist[base + j];
                }
                scan[l] = sum;
            }
            barrier();
            if !done && l < warps {
                let w_base = l * LANES as u32;
                let mut wt = 0u32;
                for j in 0..LANES as u32 {
                    wt += scan[w_base + j];
                }
                scan[threads + l] = wt;
            }
            barrier();
            if !done {
                if pass == 0u32 {
                    let mut finite = 0u32;
                    for w in 0..warps {
                        finite += scan[threads + w];
                    }
                    if finite <= k {
                        done = true;
                    }
                }
                if !done {
                    let mut win_w = 0u32;
                    let mut found_w = false;
                    for rev in 0..warps {
                        if !found_w {
                            let w = warps - 1u32 - rev;
                            let c = scan[threads + w];
                            if need > c {
                                need -= c;
                            } else {
                                win_w = w;
                                found_w = true;
                            }
                        }
                    }
                    let mut win_m = 0u32;
                    let mut found_m = false;
                    let w_base = win_w * LANES as u32;
                    for rev in 0..LANES as u32 {
                        if !found_m {
                            let m = LANES as u32 - 1u32 - rev;
                            let c = scan[w_base + m];
                            if need > c {
                                need -= c;
                            } else {
                                win_m = w_base + m;
                                found_m = true;
                            }
                        }
                    }
                    let mut found_b = false;
                    let b_base = win_m * hist_step;
                    for rev in 0..hist_step {
                        if !found_b {
                            let j = hist_step - 1u32 - rev;
                            let b = b_base + j;
                            let c = hist[b];
                            if need > c {
                                need -= c;
                            } else {
                                prefix = prefix | (b << shift);
                                mask = mask | (bmask << shift);
                                if c == need {
                                    strict_min = if prefix > 0u32 { prefix } else { 1u32 };
                                    done = true;
                                }
                                found_b = true;
                            }
                        }
                    }
                }
            }
            barrier();
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
    let tile = threads * TILE_ELEMS;
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
        if l < warps {
            let mut sum = 0u32;
            let w_base = l * LANES as u32;
            for j in 0..LANES as u32 {
                let v = scan[w_base + j];
                scan[w_base + j] = sum;
                sum += v;
            }
            scan[threads + l] = sum;
        }
        barrier();
        let warp = l / LANES as u32;
        let mut w_pre = 0u32;
        let mut tile_tot = 0u32;
        for w in 0..warps {
            let wt = scan[threads + w];
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
                    if cache_keys {
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
    for _s in 0..max_stages {
        if cap < above {
            cap = cap * 2u32;
        }
    }
    for p in (above + l..cap).by(threads) {
        pool[p] = width;
        if cache_keys {
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
                for pair in (l..pairs).by(threads) {
                    let block = pair / stride;
                    let off = pair % stride;
                    let p = block * stride * 2u32 + off;
                    let q = p + stride;
                    let ip = pool[p];
                    let iq = pool[q];
                    let mut kp = 0u32;
                    let mut kq = 0u32;
                    if cache_keys {
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
                        if cache_keys {
                            pool[2048u32 + p] = kq;
                            pool[2048u32 + q] = kp;
                        }
                    }
                }
            }
            barrier();
        }
    }

    for p in (l..above).by(threads) {
        out[out_at + p as u64] = first + pool[p];
    }
    if need_eq == 0u32 {
        for p in (above + l..k).by(threads) {
            out[out_at + p as u64] = whole;
        }
    }
}
