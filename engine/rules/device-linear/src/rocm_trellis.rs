//! RDNA 3/3.5 ROCm trellis WMMA rule dispatch.

use super::mm::pick_divisor;
use super::{BLOCK, THREADS, TrellisMma, TrellisPreMma, rocm_needs, rocm_rule};
use std::sync::OnceLock;
use tessera_device::{Backend, flatten_for};
use tessera_lower::{Draft, Needs, T, Tree};

const ROCM_TRELLIS: &str = include_str!("rocm_trellis.cu");
const SMEM_T1: usize = 24 * 1024;
const SMEM_T2: usize = 32 * 1024;
const SMEM_PRE: usize = 21_504;

pub(crate) fn dtr_ks_for(
    m: usize,
    n: usize,
    k: usize,
    narrow_m: usize,
    targets: [usize; 2],
) -> usize {
    let ibs = k / BLOCK;
    if m > 16 || ibs <= 1 {
        return 1;
    }
    let target = if m <= narrow_m {
        targets[0]
    } else {
        targets[1]
    };
    let want = target.div_ceil(n / BLOCK).clamp(1, 16).min(ibs);
    for d in want..=ibs.min(16) {
        if ibs.is_multiple_of(d) {
            return d;
        }
    }
    for d in (1..want).rev() {
        if ibs.is_multiple_of(d) {
            return d;
        }
    }
    1
}

pub(crate) fn rocm_trellis_source() -> &'static str {
    static ROCM: OnceLock<&'static str> = OnceLock::new();
    ROCM.get_or_init(|| flatten_for(Backend::Rocm, "rocm_trellis.kernel", ROCM_TRELLIS))
}

pub(crate) fn small_tile(m: usize) -> usize {
    if m <= 16 { 16 } else { 32 }
}

pub(crate) fn small_needs(m: usize) -> Needs {
    rocm_needs(
        if small_tile(m) == 16 {
            SMEM_T1
        } else {
            SMEM_T2
        },
        THREADS,
        true,
    )
}

pub(crate) fn pre_tile(m: usize, n: usize) -> usize {
    let tail_ok = m >= 512 || m.is_multiple_of(256) || (m % 256) >= 192;
    if m >= 256 && tail_ok && m.div_ceil(256) * (n / BLOCK) >= 40 {
        256
    } else if m > 64 {
        128
    } else {
        64
    }
}

pub(crate) fn pre_needs(_m: usize, _n: usize) -> Needs {
    rocm_needs(SMEM_PRE, THREADS, true)
}

pub(super) fn rocm_pre_ks(m: usize, n: usize, k: usize) -> usize {
    let ibs = k / BLOCK;
    let tile = pre_tile(m, n);
    let base = (n / BLOCK) * m.div_ceil(tile);
    if tile == 256 || ibs < 8 || base >= 64 {
        return 1;
    }
    let cap = if base <= 8 { 8 } else { 4 }.min(ibs / 2);
    let want = 64usize.div_ceil(base).clamp(1, cap);
    pick_divisor(ibs, want, cap)
}

pub(crate) fn small_entry(t: &TrellisMma) -> &'static str {
    match (small_tile(t.m), t.bits, t.book, t.turned, t.ks > 1) {
        (16, 4, 0, true, true) => "rocm_dtr_k4_t1s",
        (16, 4, 0, true, false) => "rocm_dtr_k4_t1",
        (16, 4, 0, false, true) => "rocm_dtr_k4_p1s",
        (16, 4, 0, false, false) => "rocm_dtr_k4_p1",
        (32, 4, 0, true, true) => "rocm_dtr_k4_t2s",
        (32, 4, 0, true, false) => "rocm_dtr_k4_t2",
        (16, _, _, _, _) => "rocm_dtr_gen1",
        (_, _, _, _, _) => "rocm_dtr_gen2",
    }
}

pub(crate) fn pre_entry(t: &TrellisPreMma) -> &'static str {
    match (
        pre_tile(t.m, t.n),
        t.bits,
        t.book,
        t.turned,
        t.ks > 1,
        t.rot_out,
    ) {
        (256, 4, 0, true, false, true) => "rocm_dtr_pre_k4_t16r",
        (128, 4, 0, true, false, true) => "rocm_dtr_pre_k4_t8r",
        (64, 4, 0, true, false, true) => "rocm_dtr_pre_k4_t4r",
        (256, 4, 0, true, false, false) => "rocm_dtr_pre_k4_t16",
        (128, 4, 0, true, true, _) => "rocm_dtr_pre_k4_t8s",
        (128, 4, 0, true, false, false) => "rocm_dtr_pre_k4_t8",
        (64, 4, 0, true, true, _) => "rocm_dtr_pre_k4_t4s",
        (64, 4, 0, true, false, false) => "rocm_dtr_pre_k4_t4",
        (256, _, _, _, _, _) => "rocm_dtr_pre_gen16",
        (128, _, _, _, _, _) => "rocm_dtr_pre_gen8",
        (_, _, _, _, _, _) => "rocm_dtr_pre_gen4",
    }
}

fn eval_rocm_trellis(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    super::gemv::eval_trellis_for(tree, at, true)
}

rocm_rule!("rocm_trellis_mma", check = eval_rocm_trellis);
