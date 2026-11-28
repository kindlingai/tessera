//! Calls for compiled lockstep kernels: reads, shuffles and tiles.

use crate::simd::{Block, Level, WIDE, block_at, dot_at, mix_at, panel_at, reduce};
use tessera_codegen::form::{Load, Read};
use tessera_codegen::numerics::Precision;
use tessera_codegen::stage::{Row, apply};
use tessera_graph::{Bf16, E2M1, E4M3, F16};

/// The most threads in one compiled group.
pub const THREADS: usize = 256;

#[inline(always)]
pub fn rd_f32(bytes: &[u8], at: usize) -> f32 {
    let b = &bytes[at * 4..at * 4 + 4];
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline(always)]
pub fn rd_bf16(bytes: &[u8], at: usize) -> f32 {
    let b = &bytes[at * 2..at * 2 + 2];
    f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)
}

#[inline(always)]
pub fn rd_f16(bytes: &[u8], at: usize) -> f32 {
    let b = &bytes[at * 2..at * 2 + 2];
    f32::from(F16(u16::from_le_bytes([b[0], b[1]])))
}

#[inline(always)]
pub fn rd_fp8(bytes: &[u8], at: usize) -> f32 {
    f32::from(E4M3(bytes[at]))
}

#[inline(always)]
pub fn rd_u32(bytes: &[u8], at: usize) -> u32 {
    let b = &bytes[at * 4..at * 4 + 4];
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline(always)]
pub fn rd_u8(bytes: &[u8], at: usize) -> u32 {
    u32::from(bytes[at])
}

#[inline(always)]
pub fn st_f32(bytes: &mut [u8], at: usize, v: f32) {
    bytes[at * 4..at * 4 + 4].copy_from_slice(&v.to_le_bytes());
}

#[inline(always)]
pub fn st_bf16(bytes: &mut [u8], at: usize, v: f32) {
    bytes[at * 2..at * 2 + 2].copy_from_slice(&Bf16::from(v).0.to_le_bytes());
}

#[inline(always)]
pub fn st_fp8(bytes: &mut [u8], at: usize, v: f32) {
    bytes[at] = E4M3::from(v).0;
}

#[inline(always)]
pub fn st_u32(bytes: &mut [u8], at: usize, v: u32) {
    bytes[at * 4..at * 4 + 4].copy_from_slice(&v.to_le_bytes());
}

/// `n` floats from `at` through stage `code`.
pub fn stage_f32(
    out: &mut [u8],
    at: usize,
    n: usize,
    (code, s): (u32, f32),
    b: Option<(&[u8], usize)>,
) {
    let Some(row) = Row::from_code(code) else {
        return;
    };
    let mut other = Vec::new();
    let operand: &[f32] = match b {
        Some((bytes, b_at)) => f32_row(bytes, b_at, n, &mut other),
        None => &[],
    };
    let slice = &mut out[at * 4..(at + n) * 4];
    // SAFETY: f32 has no invalid bit patterns.
    let (head, body, _) = unsafe { slice.align_to_mut::<f32>() };
    if head.is_empty() && body.len() == n {
        return apply(row, body, operand, s);
    }
    let words = slice.as_chunks::<4>().0;
    let mut line: Vec<f32> = words.iter().map(|w| f32::from_le_bytes(*w)).collect();
    apply(row, &mut line, operand, s);
    for (i, v) in line.iter().enumerate() {
        slice[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
}

/// `n` floats from `at` become `out * keep + p * b`.
pub fn mix_f32(out: &mut [u8], at: usize, keep: f32, p: f32, b: &[u8], b_at: usize, n: usize) {
    let mut other = Vec::new();
    let b = f32_row(b, b_at, n, &mut other);
    let slice = &mut out[at * 4..(at + n) * 4];
    // SAFETY: f32 has no invalid bit patterns.
    let (head, body, _) = unsafe { slice.align_to_mut::<f32>() };
    if head.is_empty() && body.len() == n {
        // A zero `keep` drops old values. They may be junk.
        mix_at(Level::best(), body, b, keep, p);
        return;
    }
    for (i, x) in b.iter().enumerate() {
        let (o, _) = slice[i * 4..].split_first_chunk::<4>().expect("four bytes");
        let old = if keep == 0.0 {
            0.0
        } else {
            f32::from_le_bytes(*o) * keep
        };
        slice[i * 4..i * 4 + 4].copy_from_slice(&(old + p * x).to_le_bytes());
    }
}

/// The u32 at byte offset `at`.
#[inline(always)]
pub fn word(bytes: &[u8], at: usize) -> u32 {
    let b = &bytes[at..at + 4];
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// The u16 at byte offset `at`, widened.
#[inline(always)]
pub fn half(bytes: &[u8], at: usize) -> u32 {
    u32::from(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
}

/// The `n` bits of `w` from bit `low`.
#[inline(always)]
pub fn field(w: u32, low: u32, n: u32) -> u32 {
    (w >> low) & ((1u64 << n) - 1) as u32
}

#[inline(always)]
pub fn f16(w: u32, low: u32) -> f32 {
    f32::from(F16(field(w, low, 16) as u16))
}

#[inline(always)]
pub fn e4m3(w: u32, low: u32) -> f32 {
    f32::from(E4M3(field(w, low, 8) as u8))
}

#[inline(always)]
pub fn e2m1(w: u32, low: u32) -> f32 {
    f32::from(E2M1(field(w, low, 4) as u8))
}

/// `x` to `e` and `m` bits, as kernels round.
#[inline(always)]
fn round_to(x: f32, e: u32, m: u32, top: f32) -> f32 {
    let a = x.abs();
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let steps = f32::from_bits((125 + (1u32 << (e - 1)) + m) << 23);
    let small = (a * steps + 8388608.0) - 8388608.0;
    let bits = a.to_bits();
    let nudge = ((1u32 << (22 - m)) - 1) + ((bits >> (23 - m)) & 1);
    let r = f32::from_bits(bits.wrapping_add(nudge) & !((1u32 << (23 - m)) - 1));
    let big = if r > top { top } else { r };
    let least = f32::from_bits((129 - (1u32 << (e - 1))) << 23);
    sign * if a < least { small / steps } else { big }
}

#[inline(always)]
pub fn round_f16(x: f32) -> f32 {
    round_to(x, 5, 10, 65504.0)
}

#[inline(always)]
pub fn round_e4m3(x: f32) -> f32 {
    round_to(x, 4, 3, 448.0)
}

/// `hi` and `lo` to bf16, nearest even, `hi` high.
#[inline(always)]
pub fn bf16_pair(hi: f32, lo: f32) -> u32 {
    let round = |x: f32| {
        let b = x.to_bits();
        b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000
    };
    round(hi) | (round(lo) >> 16)
}

/// Each lane's value from lane `m` away in its group.
pub fn shfl_xor(v: &[f32; THREADS], m: &[u32; THREADS], n: usize, lanes: usize) -> [f32; THREADS] {
    let mut out = *v;
    for t in 0..n {
        let other = t ^ (m[t] as usize);
        if other / lanes == t / lanes && other < n {
            out[t] = v[other];
        }
    }
    out
}

/// Each live lane takes its group's sum of live lanes.
pub fn sum32(v: &mut [f32; THREADS], mask: &[bool; THREADS], n: usize, lanes: usize) {
    let mut group = 0;
    while group < n {
        let end = (group + lanes).min(n);
        let mut sum = 0.0;
        for t in group..end {
            if mask[t] {
                sum += v[t];
            }
        }
        for t in group..end {
            if mask[t] {
                v[t] = sum;
            }
        }
        group = end;
    }
}

/// `rows` by `cols` sums per SIMD group of `lanes` threads.
pub struct Tiles {
    rows: usize,
    cols: usize,
    lanes: usize,
    blocks: Vec<Vec<f32>>,
}

impl Tiles {
    pub fn new(rows: usize, cols: usize, n: usize, lanes: usize) -> Self {
        let simds = n.div_ceil(lanes);
        Self {
            rows,
            cols,
            lanes,
            blocks: vec![vec![0.0; rows * cols]; simds],
        }
    }

    /// Each lane's sums.
    pub fn per(&self) -> usize {
        (self.rows * self.cols).div_ceil(self.lanes)
    }

    pub fn zero(&mut self) {
        for block in &mut self.blocks {
            block.fill(0.0);
        }
    }

    /// Lane `t`'s sum `e`: live, row, column and value.
    pub fn each(&self, t: usize, e: usize, live: bool) -> (bool, u32, u32, f32) {
        let at = t % self.lanes + e * self.lanes;
        let on = live && at < self.rows * self.cols;
        let (r, c) = ((at / self.cols) as u32, (at % self.cols) as u32);
        let v = if on {
            self.blocks[t / self.lanes][at]
        } else {
            0.0
        };
        (on, r, c, v)
    }

    /// The row of lane `t`'s sum `e`.
    pub fn row(&self, t: usize, e: usize) -> u32 {
        ((t % self.lanes + e * self.lanes) / self.cols) as u32
    }

    /// Scales lane `t`'s sum `e` by `by`.
    pub fn scale(&mut self, t: usize, e: usize, by: f32) {
        let at = t % self.lanes + e * self.lanes;
        if at < self.rows * self.cols {
            self.blocks[t / self.lanes][at] *= by;
        }
    }

    /// Copies each SIMD group's slab from `a`.
    pub fn hold(
        &mut self,
        a: &[f32],
        a_at: &[u32; THREADS],
        stride: &[u32; THREADS],
        mask: &[bool; THREADS],
        n: usize,
    ) {
        let cols = self.cols;
        for (s, t) in leaders(self.lanes, n, mask) {
            let (at, stride) = (a_at[t] as usize, stride[t] as usize);
            for (i, v) in self.blocks[s].iter_mut().enumerate() {
                *v = a[at + i / cols * stride + i % cols];
            }
        }
    }

    /// Adds `depth` columns of the slab `held` times `b`.
    #[allow(clippy::too_many_arguments, reason = "mirrors the tile statement")]
    pub fn block_held(
        &mut self,
        held: &Tiles,
        b: &[f32],
        b_at: &[u32; THREADS],
        stride: &[u32; THREADS],
        depth: usize,
        mask: &[bool; THREADS],
        n: usize,
        p: Precision,
    ) {
        for (s, t) in leaders(self.lanes, n, mask) {
            let stride = stride[t] as usize;
            let at = Slabs {
                a: 0,
                b: b_at[t] as usize,
                stride,
                parts: 0,
                apart: 0,
            };
            let slab = &held.blocks[s];
            let a = |i: usize| slab[i / stride * held.cols + i % stride].to_bits();
            let b = |i: usize| b[i].to_bits();
            at.products(p, self.rows, self.cols, depth, a, b, &mut self.blocks[s]);
        }
    }

    /// Adds `depth` columns of `a` times `b`, unit rounded.
    #[allow(clippy::too_many_arguments, reason = "mirrors the tile statement")]
    pub fn block(
        &mut self,
        a: &[f32],
        a_at: &[u32; THREADS],
        b: &[f32],
        b_at: &[u32; THREADS],
        stride: &[u32; THREADS],
        depth: usize,
        (parts, apart): (usize, &[u32; THREADS]),
        mask: &[bool; THREADS],
        n: usize,
        p: Precision,
    ) {
        for (s, t) in leaders(self.lanes, n, mask) {
            let at = Slabs {
                a: a_at[t] as usize,
                b: b_at[t] as usize,
                stride: stride[t] as usize,
                parts,
                apart: apart[t] as usize,
            };
            let (a, b) = (|i: usize| a[i].to_bits(), |i: usize| b[i].to_bits());
            at.products(p, self.rows, self.cols, depth, a, b, &mut self.blocks[s]);
        }
    }
}

/// Each SIMD group with a live lane, and that lane.
pub(crate) fn leaders(
    lanes: usize,
    n: usize,
    mask: &[bool],
) -> impl Iterator<Item = (usize, usize)> {
    let first = move |s: usize| (s * lanes..((s + 1) * lanes).min(n)).find(|t| mask[*t]);
    (0..n.div_ceil(lanes)).filter_map(move |s| first(s).map(|t| (s, t)))
}

/// Where a SIMD group's tile operands sit in group memory.
pub(crate) struct Slabs {
    pub a: usize,
    pub b: usize,
    /// Elements a row, or words for bf16 pairs.
    pub stride: usize,
    /// Zero for f32. Else A sums this many bf16 slabs.
    pub parts: usize,
    pub apart: usize,
}

impl Slabs {
    /// `block += a * b` over `depth` columns, reading words by index.
    #[allow(clippy::too_many_arguments, reason = "every tile operand")]
    pub(crate) fn products(
        &self,
        p: Precision,
        rows: usize,
        cols: usize,
        depth: usize,
        a: impl Fn(usize) -> u32,
        b: impl Fn(usize) -> u32,
        block: &mut [f32],
    ) {
        let Slabs {
            a: a0,
            b: b0,
            stride,
            parts,
            apart,
        } = *self;
        if parts == 0 {
            let (fa, fb) = (
                |i: usize| f32::from_bits(a(a0 + i)),
                |i: usize| f32::from_bits(b(b0 + i)),
            );
            return products(p, rows, cols, stride, depth, fa, fb, block);
        }
        let ld = 2 * stride;
        // Element `i` sits at `(i / ld, i % ld)`.
        let value = |word: &dyn Fn(usize) -> u32, base: usize, parts: usize, i: usize| {
            let (r, k) = (i / ld, i % ld);
            let mut v = 0.0f32;
            for part in (0..parts).rev() {
                let w = word(base + part * apart + r * stride + k / 2);
                v += f32::from_bits(if k % 2 == 0 { w << 16 } else { w & 0xffff_0000 });
            }
            v
        };
        let (fa, fb) = (|i| value(&a, a0, parts, i), |i| value(&b, b0, 1, i));
        products(p.parts(), rows, cols, ld, depth, fa, fb, block);
    }
}

/// `block[i][j] += sum of a[i][k] * b[j][k]`, as `p` rounds.
#[allow(clippy::too_many_arguments, reason = "every tile operand")]
pub(crate) fn products(
    p: Precision,
    rows: usize,
    cols: usize,
    ld: usize,
    depth: usize,
    a: impl Fn(usize) -> f32,
    b: impl Fn(usize) -> f32,
    block: &mut [f32],
) {
    let chunk = p.depth.max(1);
    for i in 0..rows {
        for j in 0..cols {
            let sum = &mut block[i * cols + j];
            // The unit sums `p.depth` products, then adds them in.
            for start in (0..depth).step_by(chunk) {
                let mut part = 0.0f32;
                for k in start..(start + chunk).min(depth) {
                    let (ah, al) = p.split(a(i * ld + k));
                    let (bh, bl) = p.split(b(j * ld + k));
                    // The split's products: hi·hi, hi·lo, lo·hi.
                    part = ah.mul_add(bh, part);
                    part = ah.mul_add(bl, part);
                    part = al.mul_add(bh, part);
                }
                *sum = p.accumulate(*sum, part);
            }
        }
    }
}

/// The sum of `a` times `b`, on the widest vectors.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    dot_at(Level::best(), a, b)
}

/// `n` floats of `bytes` from element `at`, copied when unaligned.
pub(crate) fn f32_row<'a>(
    bytes: &'a [u8],
    at: usize,
    n: usize,
    scratch: &'a mut Vec<f32>,
) -> &'a [f32] {
    let slice = &bytes[at * 4..(at + n) * 4];
    // SAFETY: f32 has no invalid bit patterns.
    let (head, body, _) = unsafe { slice.align_to::<f32>() };
    if head.is_empty() && body.len() == n {
        return body;
    }
    scratch.clear();
    let (words, _) = slice.as_chunks::<4>();
    scratch.extend(words.iter().map(|c| f32::from_le_bytes(*c)));
    scratch
}

/// Depth the generic block sums in cache.
const SPAN: usize = 512;
/// Rows and columns of one block of sums.
pub const BLOCK: usize = WIDE;

/// Where a block reads and writes, in elements.
pub struct Dots {
    pub at: usize,
    pub stride: usize,
    pub x_at: usize,
    pub w_at: usize,
    pub k: usize,
    /// Columns that exist. The rest read as zero.
    pub cols: usize,
}

use std::cell::RefCell;

/// Precomputed OCP E4M3 to f32 decode table.
const E4M3_LUT: [f32; 256] = {
    let mut t = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = E4M3(i as u8).to_f32();
        i += 1;
    }
    t
};

/// Widens one little-endian bf16 or f16 pair to `f32`.
#[inline(always)]
fn widen_half<const BF16: bool>(pair: [u8; 2]) -> f32 {
    let raw = u16::from_le_bytes(pair);
    if BF16 {
        return f32::from_bits(u32::from(raw) << 16);
    }
    let sign = u32::from(raw & 0x8000) << 16;
    let mag = u32::from(raw & 0x7FFF);
    let inf = if mag >= 0x7C00 { 0x7F80_0000u32 } else { 0 };
    f32::from_bits(sign | (mag << 13) | inf) * f32::from_bits(0x7780_0000)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn widen_f16_x4(h: std::arch::aarch64::uint16x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    // SAFETY: aarch64 always supports NEON vector instructions.
    unsafe {
        let u = vmovl_u16(h);
        let sign = vandq_u32(vshlq_n_u32::<16>(u), vdupq_n_u32(0x8000_0000));
        let mag = vandq_u32(u, vdupq_n_u32(0x7FFF));
        let inf = vandq_u32(
            vcgeq_u32(mag, vdupq_n_u32(0x7C00)),
            vdupq_n_u32(0x7F80_0000),
        );
        let bits = vorrq_u32(vorrq_u32(sign, vshlq_n_u32::<13>(mag)), inf);
        vmulq_f32(
            vreinterpretq_f32_u32(bits),
            vdupq_n_f32(f32::from_bits(0x7780_0000)),
        )
    }
}

/// Widens little-endian bf16 or f16 bytes into `dst`.
#[inline(always)]
fn widen_halves_row<const BF16: bool>(raw: &[u8], dst: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let (ws, wr) = raw.as_chunks::<16>();
        let (ds, dr) = dst.as_chunks_mut::<8>();
        // SAFETY: aarch64 has NEON and chunk bounds match.
        unsafe {
            for (wc, dc) in ws.iter().zip(ds) {
                let halves = vld1q_u16(wc.as_ptr().cast::<u16>());
                let (lo, hi) = if BF16 {
                    (
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(halves))),
                        vreinterpretq_f32_u32(vshll_high_n_u16::<16>(halves)),
                    )
                } else {
                    (
                        widen_f16_x4(vget_low_u16(halves)),
                        widen_f16_x4(vget_high_u16(halves)),
                    )
                };
                vst1q_f32(dc.as_mut_ptr(), lo);
                vst1q_f32(dc.as_mut_ptr().add(4), hi);
            }
        }
        for (o, &pair) in dr.iter_mut().zip(wr.as_chunks::<2>().0) {
            *o = widen_half::<BF16>(pair);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (o, &pair) in dst.iter_mut().zip(raw.as_chunks::<2>().0) {
        *o = widen_half::<BF16>(pair);
    }
}

/// Widens OCP e4m3 bytes into `dst` via the lookup table.
fn widen_fp8_row(raw: &[u8], dst: &mut [f32]) {
    for (o, &b) in dst.iter_mut().zip(raw) {
        *o = E4M3_LUT[b as usize];
    }
}

/// Copies `n` little-endian f32 values from `bytes` into `dst`.
fn copy_f32_row(bytes: &[u8], at: usize, n: usize, dst: &mut [f32]) {
    let slice = &bytes[at * 4..(at + n) * 4];
    for (o, &word) in dst.iter_mut().zip(slice.as_chunks::<4>().0) {
        *o = f32::from_le_bytes(word);
    }
}

/// Aligned f32 view of `n` elements, if aligned.
#[inline(always)]
fn f32_aligned(bytes: &[u8], at: usize, n: usize) -> Option<&[f32]> {
    let slice = &bytes[at * 4..(at + n) * 4];
    // SAFETY: f32 has no invalid bit patterns.
    let (head, body, _) = unsafe { slice.align_to::<f32>() };
    (head.is_empty() && body.len() == n).then_some(body)
}

/// Reusable per-thread scratch buffers for `dots_form`.
#[derive(Default)]
struct DotsScratch {
    w_buf: Vec<f32>,
    x_buf: Vec<f32>,
    zeros: Vec<f32>,
    acc: Vec<[f32; BLOCK]>,
}

thread_local! {
    static DOTS_SCRATCH: RefCell<DotsScratch> = RefCell::new(DotsScratch::default());
}

/// Runs `panel_at` or `block_rows` into `acc` and writes `out`.
fn run_dots_tile(
    xr: &[&[f32]],
    wr: &[&[f32]; BLOCK],
    acc: &mut [[f32; BLOCK]],
    out: &mut [u8],
    d: &Dots,
) {
    if !panel_at(Level::best(), xr, wr, acc) {
        // The generic block holds 16 rows by 16 columns.
        for (chunk, lines) in acc.chunks_mut(16).enumerate() {
            let x = &xr[chunk * 16..chunk * 16 + lines.len()];
            for col in (0..BLOCK).step_by(16) {
                block_rows(x, &wr[col..col + 16], (col, &mut *lines));
            }
        }
    }
    for (r, line) in acc.iter().enumerate() {
        let row_start = (d.at + r * d.stride) * 4;
        let row_bytes = &mut out[row_start..row_start + d.cols * 4];
        // SAFETY: f32 has no invalid bit patterns.
        let (head, body, _) = unsafe { row_bytes.align_to_mut::<f32>() };
        if head.is_empty() && body.len() == d.cols {
            body.copy_from_slice(&line[..d.cols]);
        } else {
            for (c, v) in line.iter().enumerate().take(d.cols) {
                st_f32(out, d.at + r * d.stride + c, *v);
            }
        }
    }
}

/// `rows` of `x` against `BLOCK` rows of `w`, into `out`.
pub fn dots_form(form: &dyn Load, out: &mut [u8], x: &[u8], w: &[u8], d: &Dots, rows: usize) {
    let k = d.k;
    let read = form.read();
    DOTS_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        let sc = &mut *sc;
        if sc.zeros.len() < k {
            sc.zeros.resize(k, 0.0);
        }
        let x_aligned = f32_aligned(x, d.x_at, rows * k).is_some();
        if !x_aligned {
            let need = rows * k;
            if sc.x_buf.len() < need {
                sc.x_buf.resize(need, 0.0);
            }
            for r in 0..rows {
                copy_f32_row(x, d.x_at + r * k, k, &mut sc.x_buf[r * k..(r + 1) * k]);
            }
        }
        let w_need = match read {
            Read::F32 => f32_aligned(w, d.w_at, d.cols * k).is_none(),
            Read::Bf16 | Read::F16 | Read::Fp8 => true,
            other => panic!("no tile of dots over {other:?} rows"),
        };
        if w_need {
            let need = d.cols * k;
            if sc.w_buf.len() < need {
                sc.w_buf.resize(need, 0.0);
            }
            match read {
                Read::F32 => {
                    for c in 0..d.cols {
                        copy_f32_row(w, d.w_at + c * k, k, &mut sc.w_buf[c * k..(c + 1) * k]);
                    }
                }
                Read::Bf16 => {
                    for c in 0..d.cols {
                        let first = d.w_at + c * k;
                        let raw = &w[first * 2..(first + k) * 2];
                        widen_halves_row::<true>(raw, &mut sc.w_buf[c * k..(c + 1) * k]);
                    }
                }
                Read::F16 => {
                    for c in 0..d.cols {
                        let first = d.w_at + c * k;
                        let raw = &w[first * 2..(first + k) * 2];
                        widen_halves_row::<false>(raw, &mut sc.w_buf[c * k..(c + 1) * k]);
                    }
                }
                Read::Fp8 => {
                    for c in 0..d.cols {
                        let first = d.w_at + c * k;
                        widen_fp8_row(&w[first..first + k], &mut sc.w_buf[c * k..(c + 1) * k]);
                    }
                }
                other => panic!("no tile of dots over {other:?} rows"),
            }
        }
        let zero = &sc.zeros[..k];
        let mut wr: [&[f32]; BLOCK] = [zero; BLOCK];
        if w_need {
            for (c, slot) in wr.iter_mut().enumerate().take(d.cols) {
                *slot = &sc.w_buf[c * k..(c + 1) * k];
            }
        } else {
            let w_all = f32_aligned(w, d.w_at, d.cols * k).expect("aligned f32");
            for (c, slot) in wr.iter_mut().enumerate().take(d.cols) {
                *slot = &w_all[c * k..(c + 1) * k];
            }
        }
        sc.acc.clear();
        sc.acc.resize(rows, [0.0f32; BLOCK]);
        let x_all = if x_aligned {
            f32_aligned(x, d.x_at, rows * k).expect("aligned f32")
        } else {
            &sc.x_buf[..rows * k]
        };
        if rows <= BLOCK {
            let mut xr = [&[][..]; BLOCK];
            for (r, slot) in xr.iter_mut().enumerate().take(rows) {
                *slot = &x_all[r * k..(r + 1) * k];
            }
            run_dots_tile(&xr[..rows], &wr, &mut sc.acc, out, d);
        } else {
            let xr: Vec<&[f32]> = (0..rows).map(|r| &x_all[r * k..(r + 1) * k]).collect();
            run_dots_tile(&xr, &wr, &mut sc.acc, out, d);
        }
    });
}

/// A few rows of `x` by `w`, span by span.
fn block_rows(x: &[&[f32]], w: &[&[f32]], (col, lines): (usize, &mut [[f32; BLOCK]])) {
    let k = w[0].len();
    // Each quad's partial sums, added up once at the end.
    let mut sums: Block = [[[[[0.0; 16]; 4]; 4]; 4]; 4];
    for start in (0..k).step_by(SPAN) {
        let end = (start + SPAN).min(k);
        let mut xs = [&[][..]; 16];
        for (dst, src) in xs.iter_mut().zip(x) {
            *dst = &src[start..end];
        }
        let mut ws = [&[][..]; 16];
        for (dst, src) in ws.iter_mut().zip(w) {
            *dst = &src[start..end];
        }
        block_at(&xs[..x.len()], x.len(), &ws[..w.len()], &mut sums);
    }
    for (r, line) in lines.iter_mut().enumerate() {
        for (cq, quad) in sums[r / 4].iter().enumerate() {
            line[col + 4 * cq..col + 4 * cq + 4].copy_from_slice(&reduce(quad)[r % 4]);
        }
    }
}

/// Dot of `x` against bf16 or f16 bytes.
#[inline(always)]
fn dot_halves<const BF16: bool>(x: &[f32], w: &[u8]) -> f32 {
    let (xs, xr) = x.as_chunks::<8>();
    let (ws, wr) = w.as_chunks::<16>();
    #[cfg(target_arch = "aarch64")]
    let mut sum = {
        use std::arch::aarch64::*;
        // SAFETY: aarch64 has NEON. Loads stay within the chunks.
        unsafe {
            let (mut lo, mut hi) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
            for (xc, wc) in xs.iter().zip(ws) {
                let halves = vld1q_u16(wc.as_ptr().cast::<u16>());
                let (a, b) = if BF16 {
                    (
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(halves))),
                        vreinterpretq_f32_u32(vshll_high_n_u16::<16>(halves)),
                    )
                } else {
                    (
                        widen_f16_x4(vget_low_u16(halves)),
                        widen_f16_x4(vget_high_u16(halves)),
                    )
                };
                lo = vfmaq_f32(lo, vld1q_f32(xc.as_ptr()), a);
                hi = vfmaq_f32(hi, vld1q_f32(xc.as_ptr().add(4)), b);
            }
            vaddvq_f32(vaddq_f32(lo, hi))
        }
    };
    #[cfg(not(target_arch = "aarch64"))]
    let mut sum = {
        let mut acc = [0.0f32; 8];
        for (xc, wc) in xs.iter().zip(ws) {
            for i in 0..8 {
                acc[i] += xc[i] * widen_half::<BF16>([wc[2 * i], wc[2 * i + 1]]);
            }
        }
        acc.iter().sum::<f32>()
    };
    for (i, xi) in xr.iter().enumerate() {
        sum += xi * widen_half::<BF16>([wr[2 * i], wr[2 * i + 1]]);
    }
    sum
}

/// `x` against bf16 `w`, widened as it is multiplied.
pub fn dot_bf16(x: &[f32], w: &[u8]) -> f32 {
    dot_halves::<true>(x, w)
}

/// `x` against f16 `w`, widened as it is multiplied.
pub(crate) fn dot_f16(x: &[f32], w: &[u8]) -> f32 {
    dot_halves::<false>(x, w)
}

/// `x` against OCP e4m3 `w` via the lookup table.
pub(crate) fn dot_fp8(x: &[f32], w: &[u8]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (xs, xr) = x.as_chunks::<8>();
    let (ws, wr) = w.as_chunks::<8>();
    for (xc, wc) in xs.iter().zip(ws) {
        for i in 0..8 {
            acc[i] = xc[i].mul_add(E4M3_LUT[wc[i] as usize], acc[i]);
        }
    }
    let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
    for (xi, &wb) in xr.iter().zip(wr) {
        sum += xi * E4M3_LUT[wb as usize];
    }
    sum
}

/// `n` products of an f32 row and a `form` row.
pub fn dot_form(form: &dyn Load, a: &[u8], a_at: usize, b: &[u8], b_at: usize, n: usize) -> f32 {
    let mut scratch = Vec::new();
    let x = f32_row(a, a_at, n, &mut scratch);
    match form.read() {
        Read::F32 => {
            let mut other = Vec::new();
            dot(x, f32_row(b, b_at, n, &mut other))
        }
        Read::Bf16 => dot_bf16(x, &b[b_at * 2..(b_at + n) * 2]),
        Read::F16 => dot_f16(x, &b[b_at * 2..(b_at + n) * 2]),
        Read::Fp8 => dot_fp8(x, &b[b_at..b_at + n]),
        Read::U32 | Read::U8 | Read::Decode => form.dot(x, b, b_at),
    }
}
