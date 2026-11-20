//! Calls for compiled lockstep kernels: reads, shuffles and tiles.

use crate::exec::{ArgVal, Grid};
use crate::simd::{Block, Level, WIDE, block_at, dot_at, mix_at, panel_at, reduce};
use tessera_codegen::form::{Load, Read};
use tessera_codegen::numerics::Precision;
use tessera_codegen::stage::{Row, apply};
use tessera_graph::{E2M1, E4M3, F16};

/// The most threads in one compiled group.
pub const THREADS: usize = 256;

/// A compiled body. False means the grid was too wide.
pub type Kernel = fn(&[ArgVal], &mut [&mut [u8]], Grid, usize, Precision) -> bool;

/// A buffer argument: launch index and elements to skip.
pub fn buf(arg: &ArgVal) -> (usize, usize) {
    match arg {
        ArgVal::Buf { at, skip, .. } => (*at, *skip),
        other => panic!("{other:?} where a buffer was due"),
    }
}

pub fn u32(arg: &ArgVal) -> u32 {
    match arg {
        ArgVal::U32(v) => *v,
        other => panic!("{other:?} where a u32 was due"),
    }
}

pub fn f32(arg: &ArgVal) -> f32 {
    match arg {
        ArgVal::F32(v) => *v,
        other => panic!("{other:?} where an f32 was due"),
    }
}

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

/// An `n`-bit field as a signed value.
#[inline(always)]
pub fn signed(w: u32, low: u32, n: u32) -> f32 {
    let shift = 32 - n;
    (((field(w, low, n) << shift) as i32) >> shift) as f32
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

macro_rules! wide {
    ($($name:ident = $op:tt),*) => {$(
        #[inline(always)]
        pub fn $name(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
            [a[0] $op b[0], a[1] $op b[1], a[2] $op b[2], a[3] $op b[3]]
        }
    )*};
}

wide!(add4 = +, sub4 = -, mul4 = *, div4 = /);

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
        for s in 0..self.blocks.len() {
            // Any live lane of the SIMD group names its slabs.
            let range = s * self.lanes..((s + 1) * self.lanes).min(n);
            let Some(t) = range.into_iter().find(|t| mask[*t]) else {
                continue;
            };
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

/// `rows` of `x` against `BLOCK` rows of `w`, into `out`.
pub fn dots_form(form: &dyn Load, out: &mut [u8], x: &[u8], w: &[u8], d: &Dots, rows: usize) {
    let k = d.k;
    let mut held: Vec<Vec<f32>> = vec![Vec::new(); rows];
    let mut xr: Vec<&[f32]> = Vec::with_capacity(rows);
    for (r, h) in held.iter_mut().enumerate() {
        xr.push(f32_row(x, d.x_at + r * k, k, h));
    }
    let mut kept: [Vec<f32>; BLOCK] = std::array::from_fn(|_| Vec::new());
    let mut wr: [&[f32]; BLOCK] = [&[]; BLOCK];
    for (c, h) in kept.iter_mut().enumerate() {
        let first = d.w_at + c * k;
        if c >= d.cols {
            h.resize(k, 0.0);
            let h: &Vec<f32> = h;
            wr[c] = h.as_slice();
            continue;
        }
        wr[c] = match form.read() {
            Read::F32 => f32_row(w, first, k, h),
            Read::Bf16 => {
                let wide = |i: usize| u16::from_le_bytes([w[i * 2], w[i * 2 + 1]]);
                h.extend((first..first + k).map(|i| f32::from_bits(u32::from(wide(i)) << 16)));
                let h: &Vec<f32> = h;
                h.as_slice()
            }
            other => panic!("no tile of dots over {other:?} rows"),
        };
    }
    let mut acc = vec![[0.0f32; BLOCK]; rows];
    if !panel_at(Level::best(), &xr, &wr, &mut acc) {
        // The generic block holds 16 rows by 16 columns.
        for (chunk, lines) in acc.chunks_mut(16).enumerate() {
            let x = &xr[chunk * 16..chunk * 16 + lines.len()];
            for col in (0..BLOCK).step_by(16) {
                block_rows(x, &wr[col..col + 16], (col, &mut *lines));
            }
        }
    }
    for (r, line) in acc.iter().enumerate() {
        for (c, v) in line.iter().enumerate().take(d.cols) {
            st_f32(out, d.at + r * d.stride + c, *v);
        }
    }
}

/// A few rows of `x` by `w`, span by span.
fn block_rows(x: &[&[f32]], w: &[&[f32]], (col, lines): (usize, &mut [[f32; BLOCK]])) {
    let k = w[0].len();
    // Each quad's partial sums, added up once at the end.
    let mut sums: Block = [[[[[0.0; 16]; 4]; 4]; 4]; 4];
    for start in (0..k).step_by(SPAN) {
        let end = (start + SPAN).min(k);
        let xs: Vec<&[f32]> = x.iter().map(|r| &r[start..end]).collect();
        let ws: Vec<&[f32]> = w.iter().map(|r| &r[start..end]).collect();
        block_at(&xs, x.len(), &ws, &mut sums);
    }
    for (r, line) in lines.iter_mut().enumerate() {
        for (cq, quad) in sums[r / 4].iter().enumerate() {
            line[col + 4 * cq..col + 4 * cq + 4].copy_from_slice(&reduce(quad)[r % 4]);
        }
    }
}

/// `x` against bf16 `w`, widened as it is multiplied.
#[cfg(not(target_arch = "aarch64"))]
pub fn dot_bf16(x: &[f32], w: &[u8]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (xs, xr) = x.as_chunks::<8>();
    let (ws, wr) = w.as_chunks::<16>();
    for (xc, wc) in xs.iter().zip(ws) {
        for i in 0..8 {
            let bits = u16::from_le_bytes([wc[2 * i], wc[2 * i + 1]]);
            acc[i] += xc[i] * f32::from_bits(u32::from(bits) << 16);
        }
    }
    let mut sum = acc.iter().sum::<f32>();
    for (i, xi) in xr.iter().enumerate() {
        let bits = u16::from_le_bytes([wr[2 * i], wr[2 * i + 1]]);
        sum += xi * f32::from_bits(u32::from(bits) << 16);
    }
    sum
}

/// `x` against bf16 `w`: NEON widens eight halves a step.
#[cfg(target_arch = "aarch64")]
pub fn dot_bf16(x: &[f32], w: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    let (xs, xr) = x.as_chunks::<8>();
    let (ws, wr) = w.as_chunks::<16>();
    // SAFETY: aarch64 has NEON. Loads stay within the chunks.
    let mut sum = unsafe {
        let (mut lo, mut hi) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        for (xc, wc) in xs.iter().zip(ws) {
            let halves = vld1q_u16(wc.as_ptr().cast::<u16>());
            let a = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(halves)));
            let b = vreinterpretq_f32_u32(vshll_high_n_u16::<16>(halves));
            lo = vfmaq_f32(lo, vld1q_f32(xc.as_ptr()), a);
            hi = vfmaq_f32(hi, vld1q_f32(xc.as_ptr().add(4)), b);
        }
        vaddvq_f32(vaddq_f32(lo, hi))
    };
    for (i, xi) in xr.iter().enumerate() {
        let bits = u16::from_le_bytes([wr[2 * i], wr[2 * i + 1]]);
        sum += xi * f32::from_bits(u32::from(bits) << 16);
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
        Read::U32 | Read::U8 | Read::Decode => form.dot(x, b, b_at),
    }
}
