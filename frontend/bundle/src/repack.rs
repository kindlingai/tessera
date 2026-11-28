//! Generic Pins-driven load-time weight repacking for streamed bundles.

use crate::names::Rule;
use crate::{Failure, Stream, numbered};
use tessera_checkpoint::Offer;
use tessera_engine::form::{Load, named};
use tessera_engine::{DimExpr, Graph, PathSeg, Pins, Reads, Weight};

/// Target form and 2D matrix alignment for load-time repacking.
#[derive(Clone, Copy)]
pub(crate) struct Repack {
    pub form: &'static dyn Load,
    /// Required `[rows, cols]` divisibility for 2D weights.
    pub align: [usize; 2],
}

impl Repack {
    /// Repack into `form` using its own storage block alignment.
    #[cfg(test)]
    pub(crate) fn into(form: &'static dyn Load) -> Self {
        Self {
            form,
            align: [1, form.align()],
        }
    }

    /// Repack into MXFP8 with `fast_mm` tile alignment.
    pub(crate) fn mxfp8() -> Option<Self> {
        Some(Self {
            form: named("mxfp8")?,
            align: [16, 512],
        })
    }

    /// Whether a 2D `[rows, cols]` matrix satisfies this policy.
    pub(crate) fn fits(self, [rows, cols]: [usize; 2]) -> bool {
        self.form.packs()
            && rows.is_multiple_of(self.align[0].max(1))
            && cols.is_multiple_of(self.align[1].max(1))
    }
}

/// Fixed 2D `[rows, cols]` shape of a weight, if static.
pub(crate) fn fixed_2d(graph: &Graph, shape: &[DimExpr]) -> Option<[usize; 2]> {
    if shape.len() != 2 {
        return None;
    }
    let val = |d: &DimExpr| d.value(&graph.dims, &|_| None).map(|n| n as usize);
    Some([val(&shape[0])?, val(&shape[1])?])
}

/// Resolves checkpoint pins, upgrading selected float weights to `target`.
pub(crate) fn pins(
    graph: &Graph,
    offer: &dyn Offer,
    layers: u64,
    rules: &[&[Rule]],
    target: Option<Repack>,
    select: impl Fn(&[PathSeg]) -> bool,
) -> Result<Pins, Failure> {
    let slices = offer.slices();
    let claims = crate::names::resolve(graph, &slices, rules)?;
    let mut forms: Vec<&'static dyn Load> = Vec::with_capacity(claims.len());
    for (c, w) in claims.iter().zip(&graph.weights) {
        let slice = slices.iter().find(|s| s.name == c.name).expect("resolved");
        let name = slice.stored.name();
        let upgrade = target.filter(|t| {
            matches!(name, "bf16" | "f32" | "f16")
                && name != t.form.name()
                && c.parts.is_empty()
                && c.rows.is_none()
                && select(&w.path)
                && fixed_2d(graph, &w.ty.shape.0).is_some_and(|dims| t.fits(dims))
        });
        match upgrade {
            Some(t) => forms.push(t.form),
            None => forms.push(crate::form(slice.stored, !c.parts.is_empty())?),
        }
    }
    Ok(crate::gguf::pins(graph, &forms, layers))
}

/// Streamed weights from any `Offer`, repacking weights to match `pins`.
pub(crate) struct Streamed {
    pub(crate) graph: Graph,
    pub(crate) offer: Box<dyn Offer>,
    pub(crate) rules: Vec<Rule>,
    pub(crate) pins: Pins,
}

impl Stream for Streamed {
    fn stream(
        &self,
        shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
        take: &mut dyn FnMut(u32, Weight) -> Result<(), Failure>,
    ) -> Result<(), Failure> {
        stream(
            &self.graph,
            &*self.offer,
            &self.rules,
            &self.pins,
            shares,
            take,
        )
    }
}

/// Streams weights from `offer`, repacking any weight whose pin differs.
pub(crate) fn stream(
    graph: &Graph,
    offer: &dyn Offer,
    rules: &[Rule],
    pins: &Pins,
    shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
    take: &mut dyn FnMut(u32, Weight) -> Result<(), Failure>,
) -> Result<(), Failure> {
    numbered(graph, offer, &[rules], shares, &mut |at, w| {
        let weight = &graph.weights[at as usize];
        let want = pins.form(&weight.path);
        let dims = fixed_2d(graph, &weight.ty.shape.0);
        let repacked = repack_weight(w, dims, shares(at), want)?;
        take(at, repacked)
    })
}

/// Source view for row-parallel unpacking into f32 batches.
#[derive(Clone, Copy)]
enum Source<'a> {
    Bf16(&'a [u8]),
    F16(&'a [u8]),
    Floats(&'a [f32]),
    Form(&'static dyn Load, &'a [u8]),
}

/// Branchless IEEE binary16 to f32 widening, matching `F16::to_f32`.
#[inline(always)]
pub(crate) fn widen_f16(raw: u16) -> f32 {
    let sign = u32::from(raw & 0x8000) << 16;
    let mag = u32::from(raw & 0x7FFF);
    let inf = if mag >= 0x7C00 { 0x7F80_0000u32 } else { 0 };
    f32::from_bits(sign | (mag << 13) | inf) * f32::from_bits(0x7780_0000)
}

/// Branchless OCP E4M3 encoding, bit-exact with `E4M3::from`.
#[inline(always)]
fn encode_e4m3(x: f32) -> u8 {
    let bits = x.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    let mag = bits & 0x7FFF_FFFF;
    let odd = (mag >> 20) & 1;
    let norm = mag.wrapping_add(0xC407_FFFF).wrapping_add(odd) >> 20;
    let sub = f32::from_bits(mag).mul_add(512.0, 8_388_608.0).to_bits() - 0x4B00_0000;
    let code = if mag < 0x3C80_0000 {
        sub
    } else if mag < 0x43E0_0000 {
        norm
    } else if mag <= 0x7F80_0000 {
        0x7E
    } else {
        0x7F
    };
    sign | (code as u8)
}

/// Rounding-up OCP E8M0 exponent, bit-exact with `E8M0::from`.
#[inline(always)]
fn encode_e8m0(x: f32) -> u8 {
    if x.is_nan() {
        return 0xFF;
    }
    let bits = x.abs().to_bits();
    let exp = (bits >> 23) as i32;
    let man = bits & 0x7F_FFFF;
    (exp + i32::from(man != 0)).clamp(0, 254) as u8
}

/// Packs one 32-element MXFP8 block directly into `dst`.
#[inline(always)]
pub(crate) fn pack_mxfp8_scalar(vals: &[f32; 32], dst: &mut [u8; 33]) {
    let mut top = 0.0f32;
    for &v in vals {
        top = top.max(v.abs());
    }
    let e8m0 = encode_e8m0(top / 448.0);
    dst[0] = e8m0;
    if (1..=253).contains(&e8m0) {
        let inv = f32::from_bits((254 - u32::from(e8m0)) << 23);
        for (o, &v) in dst[1..].iter_mut().zip(vals) {
            *o = encode_e4m3(v * inv);
        }
    } else {
        let stored = f32::from_bits(u32::from(e8m0) << 23);
        for (o, &v) in dst[1..].iter_mut().zip(vals) {
            let scaled = if stored == 0.0 { 0.0 } else { v / stored };
            *o = encode_e4m3(scaled);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::encode_e8m0;
    use std::arch::aarch64::*;

    /// Widens four IEEE binary16 values in `h` into f32 lanes.
    #[inline(always)]
    unsafe fn widen_f16_x4(h: uint16x4_t) -> float32x4_t {
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

    /// Encodes four f32 lanes to OCP E4M3 codes.
    #[inline(always)]
    unsafe fn encode_e4m3_x4(v: float32x4_t) -> uint32x4_t {
        // SAFETY: aarch64 always supports NEON vector instructions.
        unsafe {
            let bits = vreinterpretq_u32_f32(v);
            let sign = vandq_u32(vshrq_n_u32::<24>(bits), vdupq_n_u32(0x80));
            let mag = vandq_u32(bits, vdupq_n_u32(0x7FFF_FFFF));
            let odd = vandq_u32(vshrq_n_u32::<20>(mag), vdupq_n_u32(1));
            let norm = vshrq_n_u32::<20>(vaddq_u32(vaddq_u32(mag, vdupq_n_u32(0xC407_FFFF)), odd));
            let sub_f = vfmaq_f32(
                vdupq_n_f32(8_388_608.0),
                vreinterpretq_f32_u32(mag),
                vdupq_n_f32(512.0),
            );
            let sub = vsubq_u32(vreinterpretq_u32_f32(sub_f), vdupq_n_u32(0x4B00_0000));
            let is_sub = vcltq_u32(mag, vdupq_n_u32(0x3C80_0000));
            let code = vbslq_u32(is_sub, sub, norm);
            let is_sat = vcgeq_u32(mag, vdupq_n_u32(0x43E0_0000));
            let is_nan = vcgtq_u32(mag, vdupq_n_u32(0x7F80_0000));
            let over = vsubq_u32(vdupq_n_u32(0x7E), is_nan);
            vorrq_u32(sign, vbslq_u32(is_sat, over, code))
        }
    }

    /// Packs four u32-lane E4M3 vectors into 16 contiguous bytes.
    #[inline(always)]
    unsafe fn store_16(vecs: [uint32x4_t; 4], out: *mut u8) {
        // SAFETY: caller provides 16 writable bytes at `out`.
        unsafe {
            let a = vuzp1q_u16(
                vreinterpretq_u16_u32(vecs[0]),
                vreinterpretq_u16_u32(vecs[1]),
            );
            let b = vuzp1q_u16(
                vreinterpretq_u16_u32(vecs[2]),
                vreinterpretq_u16_u32(vecs[3]),
            );
            let bytes = vuzp1q_u8(vreinterpretq_u8_u16(a), vreinterpretq_u8_u16(b));
            vst1q_u8(out, bytes);
        }
    }

    /// Packs eight f32x4 vectors directly into one 33-byte MXFP8 block.
    #[inline(always)]
    unsafe fn pack_vecs(v: [float32x4_t; 8], dst: &mut [u8; 33]) {
        // SAFETY: `dst` has 32 bytes after the scale.
        unsafe {
            let m0 = vmaxnmq_f32(vabsq_f32(v[0]), vabsq_f32(v[1]));
            let m1 = vmaxnmq_f32(vabsq_f32(v[2]), vabsq_f32(v[3]));
            let m2 = vmaxnmq_f32(vabsq_f32(v[4]), vabsq_f32(v[5]));
            let m3 = vmaxnmq_f32(vabsq_f32(v[6]), vabsq_f32(v[7]));
            let m01 = vmaxnmq_f32(m0, m1);
            let m23 = vmaxnmq_f32(m2, m3);
            let top = vmaxnmvq_f32(vmaxnmq_f32(m01, m23)).max(0.0);
            let e8m0 = encode_e8m0(top / 448.0);
            dst[0] = e8m0;
            let s = if (1..=253).contains(&e8m0) {
                let inv = vdupq_n_f32(f32::from_bits((254 - u32::from(e8m0)) << 23));
                [
                    vmulq_f32(v[0], inv),
                    vmulq_f32(v[1], inv),
                    vmulq_f32(v[2], inv),
                    vmulq_f32(v[3], inv),
                    vmulq_f32(v[4], inv),
                    vmulq_f32(v[5], inv),
                    vmulq_f32(v[6], inv),
                    vmulq_f32(v[7], inv),
                ]
            } else if e8m0 == 0 {
                let z = vdupq_n_f32(0.0);
                [z; 8]
            } else {
                let st = vdupq_n_f32(f32::from_bits(u32::from(e8m0) << 23));
                [
                    vdivq_f32(v[0], st),
                    vdivq_f32(v[1], st),
                    vdivq_f32(v[2], st),
                    vdivq_f32(v[3], st),
                    vdivq_f32(v[4], st),
                    vdivq_f32(v[5], st),
                    vdivq_f32(v[6], st),
                    vdivq_f32(v[7], st),
                ]
            };
            let p = dst.as_mut_ptr().add(1);
            store_16(
                [
                    encode_e4m3_x4(s[0]),
                    encode_e4m3_x4(s[1]),
                    encode_e4m3_x4(s[2]),
                    encode_e4m3_x4(s[3]),
                ],
                p,
            );
            store_16(
                [
                    encode_e4m3_x4(s[4]),
                    encode_e4m3_x4(s[5]),
                    encode_e4m3_x4(s[6]),
                    encode_e4m3_x4(s[7]),
                ],
                p.add(16),
            );
        }
    }

    pub(super) fn pack_bf16_block(raw: &[u8; 64], dst: &mut [u8; 33]) {
        // SAFETY: `raw` holds 32 little-endian u16 values.
        unsafe {
            let p = raw.as_ptr().cast::<u16>();
            let h0 = vld1q_u16(p);
            let h1 = vld1q_u16(p.add(8));
            let h2 = vld1q_u16(p.add(16));
            let h3 = vld1q_u16(p.add(24));
            let v = [
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(h0))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(h0)),
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(h1))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(h1)),
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(h2))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(h2)),
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(h3))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(h3)),
            ];
            pack_vecs(v, dst);
        }
    }

    pub(super) fn pack_f16_block(raw: &[u8; 64], dst: &mut [u8; 33]) {
        // SAFETY: `raw` holds 32 little-endian u16 values.
        unsafe {
            let p = raw.as_ptr().cast::<u16>();
            let h0 = vld1q_u16(p);
            let h1 = vld1q_u16(p.add(8));
            let h2 = vld1q_u16(p.add(16));
            let h3 = vld1q_u16(p.add(24));
            let v = [
                widen_f16_x4(vget_low_u16(h0)),
                widen_f16_x4(vget_high_u16(h0)),
                widen_f16_x4(vget_low_u16(h1)),
                widen_f16_x4(vget_high_u16(h1)),
                widen_f16_x4(vget_low_u16(h2)),
                widen_f16_x4(vget_high_u16(h2)),
                widen_f16_x4(vget_low_u16(h3)),
                widen_f16_x4(vget_high_u16(h3)),
            ];
            pack_vecs(v, dst);
        }
    }

    pub(super) fn pack_f32_block(vals: &[f32; 32], dst: &mut [u8; 33]) {
        // SAFETY: `vals` holds 32 contiguous f32 values.
        unsafe {
            let p = vals.as_ptr();
            let v = [
                vld1q_f32(p),
                vld1q_f32(p.add(4)),
                vld1q_f32(p.add(8)),
                vld1q_f32(p.add(12)),
                vld1q_f32(p.add(16)),
                vld1q_f32(p.add(20)),
                vld1q_f32(p.add(24)),
                vld1q_f32(p.add(28)),
            ];
            pack_vecs(v, dst);
        }
    }
}

impl Source<'_> {
    /// Copies `count` elements from `row` starting at `col` into `dst`.
    fn row(&self, row: usize, col: usize, stride: usize, dst: &mut [f32]) {
        let first = row * stride + col;
        match *self {
            Source::Bf16(bytes) => {
                let raw = &bytes[first * 2..(first + dst.len()) * 2];
                for (o, &pair) in dst.iter_mut().zip(raw.as_chunks::<2>().0) {
                    *o = f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16);
                }
            }
            Source::F16(bytes) => {
                let raw = &bytes[first * 2..(first + dst.len()) * 2];
                for (o, &pair) in dst.iter_mut().zip(raw.as_chunks::<2>().0) {
                    *o = widen_f16(u16::from_le_bytes(pair));
                }
            }
            Source::Floats(vals) => {
                dst.copy_from_slice(&vals[first..first + dst.len()]);
            }
            Source::Form(form, bytes) => {
                for (i, o) in dst.iter_mut().enumerate() {
                    *o = form.unpack(bytes, first + i);
                }
            }
        }
    }

    /// Packs `c_cnt` elements of one row directly into MXFP8 blocks.
    fn pack_mxfp8_row(&self, row: usize, col: usize, stride: usize, c_cnt: usize, out: &mut [u8]) {
        let first = row * stride + col;
        let (blocks, _) = out.as_chunks_mut::<33>();
        match *self {
            Source::Bf16(bytes) => {
                let raw = &bytes[first * 2..(first + c_cnt) * 2];
                #[cfg(target_arch = "aarch64")]
                for (dst, chunk) in blocks.iter_mut().zip(raw.as_chunks::<64>().0) {
                    neon::pack_bf16_block(chunk, dst);
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    let mut vals = [0.0f32; 32];
                    for (dst, chunk) in blocks.iter_mut().zip(raw.as_chunks::<64>().0) {
                        for (o, &pair) in vals.iter_mut().zip(chunk.as_chunks::<2>().0) {
                            *o = f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16);
                        }
                        pack_mxfp8_scalar(&vals, dst);
                    }
                }
            }
            Source::F16(bytes) => {
                let raw = &bytes[first * 2..(first + c_cnt) * 2];
                #[cfg(target_arch = "aarch64")]
                for (dst, chunk) in blocks.iter_mut().zip(raw.as_chunks::<64>().0) {
                    neon::pack_f16_block(chunk, dst);
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    let mut vals = [0.0f32; 32];
                    for (dst, chunk) in blocks.iter_mut().zip(raw.as_chunks::<64>().0) {
                        for (o, &pair) in vals.iter_mut().zip(chunk.as_chunks::<2>().0) {
                            *o = widen_f16(u16::from_le_bytes(pair));
                        }
                        pack_mxfp8_scalar(&vals, dst);
                    }
                }
            }
            Source::Floats(vals) => {
                let slice = &vals[first..first + c_cnt];
                for (dst, chunk) in blocks.iter_mut().zip(slice.as_chunks::<32>().0) {
                    #[cfg(target_arch = "aarch64")]
                    neon::pack_f32_block(chunk, dst);
                    #[cfg(not(target_arch = "aarch64"))]
                    pack_mxfp8_scalar(chunk, dst);
                }
            }
            Source::Form(form, bytes) => {
                let mut vals = [0.0f32; 32];
                for (b, dst) in blocks.iter_mut().enumerate() {
                    let base = first + b * 32;
                    for (i, o) in vals.iter_mut().enumerate() {
                        *o = form.unpack(bytes, base + i);
                    }
                    pack_mxfp8_scalar(&vals, dst);
                }
            }
        }
    }
}

/// Repacks a 2D weight slice into `target` across CPU threads.
pub(crate) fn repack_weight(
    w: Weight,
    dims: Option<[usize; 2]>,
    share: Option<(u32, usize, usize)>,
    target: &'static dyn Load,
) -> Result<Weight, Failure> {
    if !target.packs() {
        return Ok(w);
    }
    let same = match &w {
        Weight::Floats(_) => target.name() == "f32",
        Weight::Packed(src, _) | Weight::Filed(src, _) => src.name() == target.name(),
    };
    if same {
        return Ok(w);
    }
    let held = match w.held(Reads::default())? {
        Some(loaded) => loaded,
        None => w,
    };
    let Some([rows, cols]) = dims else {
        return Ok(held);
    };
    let (r0, r_cnt, c0, c_cnt) = match share {
        Some((0, first, count)) => (first, count, 0, cols),
        Some((1, first, count)) => (0, rows, first, count),
        _ => (0, rows, 0, cols),
    };
    let (src, len_elems) = match &held {
        Weight::Packed(form, bytes) if form.name() == "bf16" => {
            (Source::Bf16(bytes), bytes.len() / 2)
        }
        Weight::Packed(form, bytes) if form.name() == "f16" => {
            (Source::F16(bytes), bytes.len() / 2)
        }
        Weight::Packed(form, bytes) => (Source::Form(*form, bytes), form.elements(bytes.len())),
        Weight::Floats(t) => {
            let vals = t.floats().map_err(|e| e.to_string())?;
            (Source::Floats(vals), vals.len())
        }
        Weight::Filed(..) => unreachable!("filed weights materialized above"),
    };
    // Pre-sliced shares already hold only `[r_cnt, c_cnt]`.
    let (r0, c0, stride) = if share.is_some() && len_elems == r_cnt * c_cnt {
        (0, 0, c_cnt)
    } else {
        (r0, c0, cols)
    };
    Ok(Weight::Packed(
        target,
        repack_2d(src, stride, [r0, r_cnt, c0, c_cnt], target),
    ))
}

/// Packs a `[r_cnt, c_cnt]` window into `target` in parallel.
fn repack_2d(
    src: Source<'_>,
    stride: usize,
    [r0, r_cnt, c0, c_cnt]: [usize; 4],
    target: &'static dyn Load,
) -> Vec<u8> {
    if r_cnt == 0 || c_cnt == 0 {
        return Vec::new();
    }
    if target.bytes(0) != 0 || !c_cnt.is_multiple_of(target.block()) {
        let mut all = vec![0.0f32; r_cnt * c_cnt];
        for r in 0..r_cnt {
            src.row(r0 + r, c0, stride, &mut all[r * c_cnt..(r + 1) * c_cnt]);
        }
        return target.pack(&all);
    }
    let row_out = target.bytes(c_cnt);
    let mut out = vec![0u8; r_cnt * row_out];
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(r_cnt);
    let direct_mxfp8 = target.name() == "mxfp8" && c_cnt.is_multiple_of(32);
    // Dynamic row chunks keep fast and slow cores balanced.
    let tile = (r_cnt / (threads * 4)).clamp(1, 64);
    let queue = std::sync::Mutex::new(out.chunks_mut(tile * row_out).enumerate());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let queue = &queue;
            scope.spawn(move || {
                let mut buf = Vec::new();
                loop {
                    let next = queue.lock().unwrap().next();
                    let Some((ci, chunk)) = next else {
                        break;
                    };
                    let start_r = r0 + ci * tile;
                    let count_r = chunk.len() / row_out;
                    if direct_mxfp8 {
                        for (dr, row_dst) in chunk.chunks_exact_mut(row_out).enumerate() {
                            src.pack_mxfp8_row(start_r + dr, c0, stride, c_cnt, row_dst);
                        }
                        continue;
                    }
                    let batch = (65_536 / c_cnt.max(1)).clamp(1, count_r);
                    buf.resize(batch * c_cnt, 0.0);
                    let mut done = 0;
                    while done < count_r {
                        let take = batch.min(count_r - done);
                        let floats = &mut buf[..take * c_cnt];
                        for dr in 0..take {
                            let dst = &mut floats[dr * c_cnt..(dr + 1) * c_cnt];
                            src.row(start_r + done + dr, c0, stride, dst);
                        }
                        let packed = target.pack(floats);
                        chunk[done * row_out..(done + take) * row_out].copy_from_slice(&packed);
                        done += take;
                    }
                }
            });
        }
    });
    out
}
