//! ARM64 NEON packed panel GEMM microkernel.

use crate::simd::WIDE;
use std::arch::aarch64::*;

/// Rows of `w` as `[depth][WIDE]`, four at a time.
pub(crate) fn pack(w: &[&[f32]; WIDE], packed: &mut [f32]) {
    let len = w[0].len();
    assert!(w.iter().all(|r| r.len() == len) && packed.len() >= len * WIDE);
    let whole = len / 4 * 4;
    for group in 0..WIDE / 4 {
        let rows = &w[group * 4..group * 4 + 4];
        let at = |kk: usize| kk * WIDE + group * 4;
        for kk in (0..whole).step_by(4) {
            // SAFETY: assert covers every row and `packed` index.
            unsafe {
                let r0 = vld1q_f32(rows[0].as_ptr().add(kk));
                let r1 = vld1q_f32(rows[1].as_ptr().add(kk));
                let r2 = vld1q_f32(rows[2].as_ptr().add(kk));
                let r3 = vld1q_f32(rows[3].as_ptr().add(kk));
                let t0 = vtrn1q_f32(r0, r1);
                let t1 = vtrn2q_f32(r0, r1);
                let t2 = vtrn1q_f32(r2, r3);
                let t3 = vtrn2q_f32(r2, r3);
                let c0 = vreinterpretq_f32_f64(vzip1q_f64(
                    vreinterpretq_f64_f32(t0),
                    vreinterpretq_f64_f32(t2),
                ));
                let c1 = vreinterpretq_f32_f64(vzip1q_f64(
                    vreinterpretq_f64_f32(t1),
                    vreinterpretq_f64_f32(t3),
                ));
                let c2 = vreinterpretq_f32_f64(vzip2q_f64(
                    vreinterpretq_f64_f32(t0),
                    vreinterpretq_f64_f32(t2),
                ));
                let c3 = vreinterpretq_f32_f64(vzip2q_f64(
                    vreinterpretq_f64_f32(t1),
                    vreinterpretq_f64_f32(t3),
                ));
                vst1q_f32(packed.as_mut_ptr().add(at(kk)), c0);
                vst1q_f32(packed.as_mut_ptr().add(at(kk + 1)), c1);
                vst1q_f32(packed.as_mut_ptr().add(at(kk + 2)), c2);
                vst1q_f32(packed.as_mut_ptr().add(at(kk + 3)), c3);
            }
        }
        for kk in whole..len {
            for (i, row) in rows.iter().enumerate() {
                packed[at(kk) + i] = row[kk];
            }
        }
    }
}

/// Vectors per row in one pass (16 columns).
const V: usize = 4;
const PASS: usize = V * 4;
/// Rows held simultaneously in 24 NEON accumulator registers.
const GROUP: usize = 6;

/// `R` rows of `x` times packed `w` from `col`.
fn rows<const R: usize>(x: [&[f32]; R], packed: &[f32], (col, acc): (usize, &mut [[f32; WIDE]])) {
    let k = packed.len() / WIDE;
    // SAFETY: `panel` checks `x`, `packed`, and `acc` lengths.
    unsafe {
        let z = vdupq_n_f32(0.0);
        let mut held = [[z; V]; R];
        for (r, h) in held.iter_mut().enumerate() {
            let p = acc[r].as_ptr().add(col);
            for (v, slot) in h.iter_mut().enumerate() {
                *slot = vld1q_f32(p.add(v * 4));
            }
        }
        let mut wv = [z; V];
        for kk in 0..k {
            let p = packed.as_ptr().add(kk * WIDE + col);
            for (v, slot) in wv.iter_mut().enumerate() {
                *slot = vld1q_f32(p.add(v * 4));
            }
            for (r, h) in held.iter_mut().enumerate() {
                let a = *x[r].get_unchecked(kk);
                for (slot, &w) in h.iter_mut().zip(&wv) {
                    *slot = vfmaq_n_f32(*slot, w, a);
                }
            }
        }
        for (r, h) in held.iter().enumerate() {
            let p = acc[r].as_mut_ptr().add(col);
            for (v, &slot) in h.iter().enumerate() {
                vst1q_f32(p.add(v * 4), slot);
            }
        }
    }
}

/// Every row of `x` times packed `w`, six at once.
pub(crate) fn panel(x: &[&[f32]], packed: &[f32], acc: &mut [[f32; WIDE]]) {
    let k = packed.len() / WIDE;
    assert!(x.iter().all(|r| r.len() == k), "rows of one length");
    assert_eq!(acc.len(), x.len(), "a sum row per row");
    for (col, first) in (0..WIDE)
        .step_by(PASS)
        .flat_map(|c| (0..x.len()).step_by(GROUP).map(move |f| (c, f)))
    {
        let row = |r: usize| x[first + r];
        let acc = (col, &mut acc[first..]);
        match (x.len() - first).min(GROUP) {
            1 => rows::<1>(std::array::from_fn(row), packed, acc),
            2 => rows::<2>(std::array::from_fn(row), packed, acc),
            3 => rows::<3>(std::array::from_fn(row), packed, acc),
            4 => rows::<4>(std::array::from_fn(row), packed, acc),
            5 => rows::<5>(std::array::from_fn(row), packed, acc),
            _ => rows::<6>(std::array::from_fn(row), packed, acc),
        }
    }
}
