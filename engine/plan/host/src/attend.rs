//! One query's softmax attention, in one call from compiled bodies.

use crate::lockstep::{f32_row, st_f32};
use crate::simd::{Level, dot_at, mix_at};
use std::cell::RefCell;

/// Where a query, its keys and its output lie.
pub struct Query {
    pub out_at: usize,
    pub q_at: usize,
    /// Key 0 in `k` and `v`, then `stride` per key.
    pub k_at: usize,
    pub v_at: usize,
    pub stride: usize,
    /// Keys the query may see, `from..to`.
    pub from: usize,
    pub to: usize,
    pub width: usize,
    pub scale: f32,
    /// The key list's start and length. Zero lists none.
    pub list_at: usize,
    pub picks: usize,
}

/// The buffers one query reads and writes, as bytes.
pub struct Rows<'a> {
    pub out: &'a mut [u8],
    pub q: &'a [u8],
    pub k: &'a [u8],
    pub v: &'a [u8],
    pub ids: &'a [u8],
}

/// Reusable per-thread scratch buffers to avoid per-query allocations.
#[derive(Default)]
struct Scratch {
    keys: Vec<usize>,
    scores: Vec<f32>,
    q: Vec<f32>,
    row: Vec<f32>,
    sum: Vec<f32>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Views `bytes` as `&[f32]` when 4-byte aligned.
#[inline(always)]
fn as_f32_slice(bytes: &[u8]) -> Option<&[f32]> {
    // SAFETY: f32 has no invalid bit patterns.
    let (head, body, tail) = unsafe { bytes.align_to::<f32>() };
    (head.is_empty() && tail.is_empty()).then_some(body)
}

/// Dot products of `q` against four key rows simultaneously.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot4(q: &[f32], k: [&[f32]; 4]) -> [f32; 4] {
    use std::arch::aarch64::*;
    let n = q.len();
    let (k0, k1, k2, k3) = (&k[0][..n], &k[1][..n], &k[2][..n], &k[3][..n]);
    let whole = n & !7;
    // SAFETY: aarch64 has NEON. All slices have length n.
    let mut s = unsafe {
        let (mut a0, mut b0) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let (mut a1, mut b1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let (mut a2, mut b2) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let (mut a3, mut b3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        let qp = q.as_ptr();
        let (p0, p1, p2, p3) = (k0.as_ptr(), k1.as_ptr(), k2.as_ptr(), k3.as_ptr());
        let mut i = 0;
        while i < whole {
            let qa = vld1q_f32(qp.add(i));
            let qb = vld1q_f32(qp.add(i + 4));
            a0 = vfmaq_f32(a0, qa, vld1q_f32(p0.add(i)));
            a1 = vfmaq_f32(a1, qa, vld1q_f32(p1.add(i)));
            a2 = vfmaq_f32(a2, qa, vld1q_f32(p2.add(i)));
            a3 = vfmaq_f32(a3, qa, vld1q_f32(p3.add(i)));
            b0 = vfmaq_f32(b0, qb, vld1q_f32(p0.add(i + 4)));
            b1 = vfmaq_f32(b1, qb, vld1q_f32(p1.add(i + 4)));
            b2 = vfmaq_f32(b2, qb, vld1q_f32(p2.add(i + 4)));
            b3 = vfmaq_f32(b3, qb, vld1q_f32(p3.add(i + 4)));
            i += 8;
        }
        [
            vaddvq_f32(vaddq_f32(a0, b0)),
            vaddvq_f32(vaddq_f32(a1, b1)),
            vaddvq_f32(vaddq_f32(a2, b2)),
            vaddvq_f32(vaddq_f32(a3, b3)),
        ]
    };
    for i in whole..n {
        s[0] = q[i].mul_add(k0[i], s[0]);
        s[1] = q[i].mul_add(k1[i], s[1]);
        s[2] = q[i].mul_add(k2[i], s[2]);
        s[3] = q[i].mul_add(k3[i], s[3]);
    }
    s
}

/// Dot products of `q` against four key rows simultaneously.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn dot4(q: &[f32], k: [&[f32]; 4]) -> [f32; 4] {
    let n = q.len();
    let (k0, k1, k2, k3) = (&k[0][..n], &k[1][..n], &k[2][..n], &k[3][..n]);
    let (mut a0, mut a1, mut a2, mut a3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
    let (qc, qr) = q.as_chunks::<4>();
    let (k0c, k0r) = k0.as_chunks::<4>();
    let (k1c, k1r) = k1.as_chunks::<4>();
    let (k2c, k2r) = k2.as_chunks::<4>();
    let (k3c, k3r) = k3.as_chunks::<4>();
    let chunks = qc.len();
    assert!(
        k0c.len() == chunks && k1c.len() == chunks && k2c.len() == chunks && k3c.len() == chunks
    );
    for i in 0..chunks {
        let qv = qc[i];
        let (v0, v1, v2, v3) = (k0c[i], k1c[i], k2c[i], k3c[i]);
        for l in 0..4 {
            a0[l] += qv[l] * v0[l];
            a1[l] += qv[l] * v1[l];
            a2[l] += qv[l] * v2[l];
            a3[l] += qv[l] * v3[l];
        }
    }
    let fold4 = |a: [f32; 4]| (a[0] + a[1]) + (a[2] + a[3]);
    let mut s = [fold4(a0), fold4(a1), fold4(a2), fold4(a3)];
    for i in 0..qr.len() {
        s[0] += qr[i] * k0r[i];
        s[1] += qr[i] * k1r[i];
        s[2] += qr[i] * k2r[i];
        s[3] += qr[i] * k3r[i];
    }
    s
}

/// Accumulates four weighted value rows into `out`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn accum4(out: &mut [f32], p: [f32; 4], v: [&[f32]; 4]) {
    use std::arch::aarch64::*;
    let n = out.len();
    let (v0, v1, v2, v3) = (&v[0][..n], &v[1][..n], &v[2][..n], &v[3][..n]);
    let whole = n & !7;
    // SAFETY: aarch64 has NEON. All slices have length n.
    unsafe {
        let (s0, s1) = (vdupq_n_f32(p[0]), vdupq_n_f32(p[1]));
        let (s2, s3) = (vdupq_n_f32(p[2]), vdupq_n_f32(p[3]));
        let op = out.as_mut_ptr();
        let (p0, p1, p2, p3) = (v0.as_ptr(), v1.as_ptr(), v2.as_ptr(), v3.as_ptr());
        let mut i = 0;
        while i < whole {
            let mut oa = vld1q_f32(op.add(i));
            let mut ob = vld1q_f32(op.add(i + 4));
            oa = vfmaq_f32(oa, s0, vld1q_f32(p0.add(i)));
            ob = vfmaq_f32(ob, s0, vld1q_f32(p0.add(i + 4)));
            oa = vfmaq_f32(oa, s1, vld1q_f32(p1.add(i)));
            ob = vfmaq_f32(ob, s1, vld1q_f32(p1.add(i + 4)));
            oa = vfmaq_f32(oa, s2, vld1q_f32(p2.add(i)));
            ob = vfmaq_f32(ob, s2, vld1q_f32(p2.add(i + 4)));
            oa = vfmaq_f32(oa, s3, vld1q_f32(p3.add(i)));
            ob = vfmaq_f32(ob, s3, vld1q_f32(p3.add(i + 4)));
            vst1q_f32(op.add(i), oa);
            vst1q_f32(op.add(i + 4), ob);
            i += 8;
        }
    }
    for i in whole..n {
        let ab = v0[i].mul_add(p[0], v1[i] * p[1]);
        let cd = v2[i].mul_add(p[2], v3[i] * p[3]);
        out[i] += ab + cd;
    }
}

/// Accumulates four weighted value rows into `out`.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn accum4(out: &mut [f32], p: [f32; 4], v: [&[f32]; 4]) {
    let n = out.len();
    let (v0, v1, v2, v3) = (&v[0][..n], &v[1][..n], &v[2][..n], &v[3][..n]);
    let (oc, or) = out.as_chunks_mut::<4>();
    let (v0c, v0r) = v0.as_chunks::<4>();
    let (v1c, v1r) = v1.as_chunks::<4>();
    let (v2c, v2r) = v2.as_chunks::<4>();
    let (v3c, v3r) = v3.as_chunks::<4>();
    let chunks = oc.len();
    assert!(
        v0c.len() == chunks && v1c.len() == chunks && v2c.len() == chunks && v3c.len() == chunks
    );
    for i in 0..chunks {
        let mut o = oc[i];
        let (r0, r1, r2, r3) = (v0c[i], v1c[i], v2c[i], v3c[i]);
        for l in 0..4 {
            o[l] += (r0[l] * p[0] + r1[l] * p[1]) + (r2[l] * p[2] + r3[l] * p[3]);
        }
        oc[i] = o;
    }
    for i in 0..or.len() {
        or[i] += (v0r[i] * p[0] + v1r[i] * p[1]) + (v2r[i] * p[2] + v3r[i] * p[3]);
    }
}

/// Fills `keys` with listed ids in `from..to`.
fn fill_listed(ids: &[u8], a: &Query, out: &mut Vec<usize>) {
    out.clear();
    for i in 0..a.picks {
        let at = (a.list_at + i) * 4;
        let j = u32::from_le_bytes([ids[at], ids[at + 1], ids[at + 2], ids[at + 3]]) as usize;
        if (a.from..a.to).contains(&j) {
            out.push(j);
        }
    }
}

/// Softmax over `scores` in place.
fn normalize_scores(scores: &mut [f32]) {
    let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - top).exp();
        total += *s;
    }
    let inv = 1.0 / total;
    for s in scores.iter_mut() {
        *s *= inv;
    }
}

/// Fast aligned attention over contiguous `from..to` keys.
fn attend_contiguous(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    a: &Query,
    scores: &mut Vec<f32>,
) {
    let (width, stride, scale) = (a.width, a.stride, a.scale);
    let n = a.to.saturating_sub(a.from);
    out.fill(0.0);
    if n == 0 {
        return;
    }
    scores.clear();
    scores.reserve(n);
    let level = Level::best();
    let mut j = a.from;
    while j + 4 <= a.to {
        let b0 = a.k_at + j * stride;
        let d = dot4(
            q,
            [
                &k[b0..b0 + width],
                &k[b0 + stride..b0 + stride + width],
                &k[b0 + 2 * stride..b0 + 2 * stride + width],
                &k[b0 + 3 * stride..b0 + 3 * stride + width],
            ],
        );
        scores.extend_from_slice(&[d[0] * scale, d[1] * scale, d[2] * scale, d[3] * scale]);
        j += 4;
    }
    while j < a.to {
        let b0 = a.k_at + j * stride;
        scores.push(dot_at(level, q, &k[b0..b0 + width]) * scale);
        j += 1;
    }
    normalize_scores(scores);
    let mut i = 0;
    while i + 4 <= n {
        let b0 = a.v_at + (a.from + i) * stride;
        accum4(
            out,
            [scores[i], scores[i + 1], scores[i + 2], scores[i + 3]],
            [
                &v[b0..b0 + width],
                &v[b0 + stride..b0 + stride + width],
                &v[b0 + 2 * stride..b0 + 2 * stride + width],
                &v[b0 + 3 * stride..b0 + 3 * stride + width],
            ],
        );
        i += 4;
    }
    while i < n {
        let b0 = a.v_at + (a.from + i) * stride;
        mix_at(level, out, &v[b0..b0 + width], 1.0, scores[i]);
        i += 1;
    }
}

/// Fast aligned attention over listed key indices.
fn attend_listed(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    a: &Query,
    keys: &[usize],
    scores: &mut Vec<f32>,
) {
    let (width, stride, scale) = (a.width, a.stride, a.scale);
    out.fill(0.0);
    if keys.is_empty() {
        return;
    }
    scores.clear();
    scores.reserve(keys.len());
    let level = Level::best();
    let (kc, kr) = keys.as_chunks::<4>();
    for &[j0, j1, j2, j3] in kc {
        let d = dot4(
            q,
            [
                &k[a.k_at + j0 * stride..a.k_at + j0 * stride + width],
                &k[a.k_at + j1 * stride..a.k_at + j1 * stride + width],
                &k[a.k_at + j2 * stride..a.k_at + j2 * stride + width],
                &k[a.k_at + j3 * stride..a.k_at + j3 * stride + width],
            ],
        );
        scores.extend_from_slice(&[d[0] * scale, d[1] * scale, d[2] * scale, d[3] * scale]);
    }
    for &j in kr {
        let b0 = a.k_at + j * stride;
        scores.push(dot_at(level, q, &k[b0..b0 + width]) * scale);
    }
    normalize_scores(scores);
    for (i, &[j0, j1, j2, j3]) in kc.iter().enumerate() {
        let s = i * 4;
        accum4(
            out,
            [scores[s], scores[s + 1], scores[s + 2], scores[s + 3]],
            [
                &v[a.v_at + j0 * stride..a.v_at + j0 * stride + width],
                &v[a.v_at + j1 * stride..a.v_at + j1 * stride + width],
                &v[a.v_at + j2 * stride..a.v_at + j2 * stride + width],
                &v[a.v_at + j3 * stride..a.v_at + j3 * stride + width],
            ],
        );
    }
    for (k_off, &j) in kr.iter().enumerate() {
        let b0 = a.v_at + j * stride;
        mix_at(
            level,
            out,
            &v[b0..b0 + width],
            1.0,
            scores[kc.len() * 4 + k_off],
        );
    }
}

/// Runs aligned contiguous or listed attention into `dst`.
#[allow(
    clippy::too_many_arguments,
    reason = "forwards aligned attention buffers"
)]
fn attend_aligned(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dst: &mut [f32],
    ids: &[u8],
    a: &Query,
    keys: &mut Vec<usize>,
    scores: &mut Vec<f32>,
) {
    if a.picks == 0 {
        attend_contiguous(q, k, v, dst, a, scores);
    } else {
        fill_listed(ids, a, keys);
        attend_listed(q, k, v, dst, a, keys, scores);
    }
}

/// Scores, their softmax, then the weighted value rows, into `out`.
pub fn attend_f32(b: Rows, a: &Query) {
    SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let Scratch {
            keys,
            scores,
            q: q_buf,
            row,
            sum,
        } = &mut *sc;
        let width = a.width;
        let q = f32_row(b.q, a.q_at, width, q_buf);
        if let (Some(k), Some(v)) = (as_f32_slice(b.k), as_f32_slice(b.v)) {
            let out_bytes = &mut b.out[a.out_at * 4..(a.out_at + width) * 4];
            // SAFETY: f32 has no invalid bit patterns.
            let (head, out_body, _) = unsafe { out_bytes.align_to_mut::<f32>() };
            if head.is_empty() && out_body.len() == width {
                attend_aligned(q, k, v, out_body, b.ids, a, keys, scores);
                return;
            }
            sum.resize(width, 0.0);
            attend_aligned(q, k, v, sum, b.ids, a, keys, scores);
            for (c, x) in sum.iter().enumerate() {
                st_f32(b.out, a.out_at + c, *x);
            }
            return;
        }
        // Unaligned fallback for arbitrary byte slices.
        let level = Level::best();
        keys.clear();
        if a.picks == 0 {
            keys.extend(a.from..a.to);
        } else {
            fill_listed(b.ids, a, keys);
        }
        scores.clear();
        scores.reserve(keys.len());
        for &j in keys.iter() {
            let k = f32_row(b.k, a.k_at + j * a.stride, width, row);
            scores.push(dot_at(level, q, k) * a.scale);
        }
        sum.clear();
        sum.resize(width, 0.0);
        if !scores.is_empty() {
            normalize_scores(scores);
            for (i, (&j, &p)) in keys.iter().zip(scores.iter()).enumerate() {
                let v = f32_row(b.v, a.v_at + j * a.stride, width, row);
                mix_at(level, sum, v, if i == 0 { 0.0 } else { 1.0 }, p);
            }
        }
        for (c, x) in sum.iter().enumerate() {
            st_f32(b.out, a.out_at + c, *x);
        }
    });
}
