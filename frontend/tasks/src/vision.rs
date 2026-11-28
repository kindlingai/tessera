//! Qwen3-VL vision preprocessing, M-RoPE positions, and multimodal chat formatting.

use crate::Error;
use crate::api::{EmbedMessage, EmbedPart};
use std::borrow::Cow;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};
use tessera_engine::{DType, Tensor};

pub(crate) const PATCH: usize = 16;
pub(crate) const MERGE: usize = 2;
pub(crate) const FACTOR: usize = PATCH * MERGE;
pub(crate) const MIN_PIXELS: usize = 4_096;
pub(crate) const MAX_PIXELS: usize = 1_310_720;
pub(crate) const IMAGE_TOKEN: u32 = 151_655;
pub(crate) const DEFAULT_SYSTEM: &str = "Represent the user's input.";

const PRECISION: u32 = 22;
const B64_PAD: u8 = 64;
const B64_WS: u8 = 254;
const B64_BAD: u8 = 255;

/// Packed RGB image, three bytes per pixel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>,
}

/// Precomputed vision buffers for one resized image.
#[derive(Clone, Debug)]
pub(crate) struct EncodedImage {
    pub(crate) vit_h: usize,
    pub(crate) vit_w: usize,
    pub(crate) rgb: Vec<u8>,
    pub(crate) pos_ids: Vec<u32>,
    pub(crate) pos_w: Vec<f32>,
    pub(crate) rot_pos: Vec<u32>,
}

impl EncodedImage {
    /// Merged visual tokens produced by this image.
    pub(crate) fn tokens(&self) -> usize {
        (self.vit_h / MERGE) * (self.vit_w / MERGE)
    }

    /// Raw ViT patches before spatial merge.
    pub(crate) fn patches(&self) -> usize {
        self.vit_h * self.vit_w
    }
}

/// One image placeholder span inside a token sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImageSpan {
    pub(crate) pos: usize,
    pub(crate) vit_h: usize,
    pub(crate) vit_w: usize,
}

impl ImageSpan {
    /// Merged visual token count for this span.
    pub(crate) fn tokens(&self) -> usize {
        (self.vit_h / MERGE) * (self.vit_w / MERGE)
    }
}

/// Worker count from host parallelism and minimum work per thread.
fn workers(work: usize, min_chunk: usize) -> usize {
    let cpus = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    (work / min_chunk.max(1)).clamp(1, cpus)
}

/// Processes mutable chunks with dynamic work stealing across threads.
fn parallel_chunks<T: Send>(
    slice: &mut [T],
    chunk_len: usize,
    threads: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    let it = Mutex::new(slice.chunks_mut(chunk_len).enumerate());
    let f = &f;
    std::thread::scope(|s| {
        for _ in 0..threads {
            let it = &it;
            s.spawn(move || {
                loop {
                    let next = it.lock().ok().and_then(|mut g| g.next());
                    let Some((idx, chunk)) = next else {
                        break;
                    };
                    f(idx, chunk);
                }
            });
        }
    });
}

/// Reinterprets a plain numeric slice as raw little-endian bytes.
fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), size_of_val(v)) }
}

/// Builds a 256-entry base64 lookup table at compile time.
const fn build_b64_lut() -> [u8; 256] {
    let mut lut = [B64_BAD; 256];
    let mut i = 0usize;
    while i < 26 {
        lut[b'A' as usize + i] = i as u8;
        lut[b'a' as usize + i] = (i + 26) as u8;
        i += 1;
    }
    i = 0;
    while i < 10 {
        lut[b'0' as usize + i] = (i + 52) as u8;
        i += 1;
    }
    lut[b'+' as usize] = 62;
    lut[b'-' as usize] = 62;
    lut[b'/' as usize] = 63;
    lut[b'_' as usize] = 63;
    lut[b'=' as usize] = B64_PAD;
    lut[b' ' as usize] = B64_WS;
    lut[b'\t' as usize] = B64_WS;
    lut[b'\n' as usize] = B64_WS;
    lut[b'\r' as usize] = B64_WS;
    lut[0x0C] = B64_WS;
    lut
}

const B64_LUT: [u8; 256] = build_b64_lut();

/// Precomputes normalized float bytes for `0..=255`.
const fn build_norm_lut() -> [[u8; 4]; 256] {
    let mut lut = [[0u8; 4]; 256];
    let mut i = 0usize;
    while i < 256 {
        lut[i] = ((i as f32 / 255.0 - 0.5) * 2.0).to_ne_bytes();
        i += 1;
    }
    lut
}

const NORM_LUT: [[u8; 4]; 256] = build_norm_lut();

/// Decodes unpadded 4-byte base64 chunks into 3-byte output chunks.
fn b64_chunk(src: &[u8], dst: &mut [u8]) -> Result<(), u8> {
    let (in_quads, _) = src.as_chunks::<4>();
    let (out_trips, _) = dst.as_chunks_mut::<3>();
    for (s, o) in in_quads.iter().zip(out_trips.iter_mut()) {
        let a = B64_LUT[s[0] as usize];
        let b = B64_LUT[s[1] as usize];
        let c = B64_LUT[s[2] as usize];
        let d = B64_LUT[s[3] as usize];
        if (a | b | c | d) >= 64 {
            for v in [a, b, c, d] {
                if v == B64_WS {
                    return Err(B64_WS);
                }
            }
            for v in [a, b, c, d] {
                if v == B64_PAD {
                    return Err(B64_PAD);
                }
            }
            return Err(B64_BAD);
        }
        let n = (u32::from(a) << 18) | (u32::from(b) << 12) | (u32::from(c) << 6) | u32::from(d);
        o[0] = (n >> 16) as u8;
        o[1] = (n >> 8) as u8;
        o[2] = n as u8;
    }
    Ok(())
}

/// Decodes a clean base64 buffer with optional parallel body pass.
fn decode_b64_clean(src: &[u8]) -> Result<Vec<u8>, u8> {
    if src.is_empty() {
        return Ok(Vec::new());
    }
    let groups = src.len() / 4;
    let body_groups = groups - 1;
    let body_in = &src[..body_groups * 4];
    let tail = &src[body_groups * 4..];
    let pad2 = tail[2] == b'=';
    let pad3 = tail[3] == b'=';
    let extra = if pad2 {
        1
    } else if pad3 {
        2
    } else {
        3
    };
    let mut out = vec![0u8; body_groups * 3 + extra];
    let body_out = &mut out[..body_groups * 3];
    let n = workers(body_in.len(), 256 * 1024);
    if n <= 1 {
        b64_chunk(body_in, body_out)?;
    } else {
        let g_per = 16_384usize;
        let err = AtomicU8::new(0);
        parallel_chunks(body_out, g_per * 3, n, |idx, out_s| {
            let start = idx * g_per * 4;
            let len = (out_s.len() / 3) * 4;
            if let Err(code) = b64_chunk(&body_in[start..start + len], out_s) {
                let _ = err.compare_exchange(0, code, Ordering::Relaxed, Ordering::Relaxed);
            }
        });
        let code = err.load(Ordering::Relaxed);
        if code != 0 {
            return Err(code);
        }
    }
    let a = B64_LUT[tail[0] as usize];
    let b = B64_LUT[tail[1] as usize];
    let c = B64_LUT[tail[2] as usize];
    let d = B64_LUT[tail[3] as usize];
    for v in [a, b, c, d] {
        if v == B64_WS {
            return Err(B64_WS);
        }
    }
    if a >= 64 || b >= 64 {
        return Err(if a == B64_PAD || b == B64_PAD {
            B64_PAD
        } else {
            B64_BAD
        });
    }
    if pad2 && !pad3 {
        return Err(B64_PAD);
    }
    let cv = if pad2 {
        0
    } else if c < 64 {
        c
    } else {
        return Err(B64_BAD);
    };
    let dv = if pad3 {
        0
    } else if d < 64 {
        d
    } else {
        return Err(B64_BAD);
    };
    let v = (u32::from(a) << 18) | (u32::from(b) << 12) | (u32::from(cv) << 6) | u32::from(dv);
    let base = body_groups * 3;
    out[base] = (v >> 16) as u8;
    if !pad2 {
        out[base + 1] = (v >> 8) as u8;
    }
    if !pad3 {
        out[base + 2] = v as u8;
    }
    Ok(out)
}

/// Decodes standard or URL-safe base64, ignoring ASCII whitespace.
pub fn decode_base64(text: &str) -> Result<Vec<u8>, Error> {
    let raw = text.as_bytes();
    if raw.len().is_multiple_of(4) {
        match decode_b64_clean(raw) {
            Ok(out) => return Ok(out),
            Err(B64_WS) => {}
            Err(B64_PAD) => return Err("unexpected base64 padding".into()),
            Err(_) => return Err("invalid base64 character".into()),
        }
    }
    let mut clean = Vec::with_capacity(raw.len());
    for &b in raw {
        if !b.is_ascii_whitespace() {
            clean.push(b);
        }
    }
    if clean.is_empty() {
        return Ok(Vec::new());
    }
    if !clean.len().is_multiple_of(4) {
        return Err("invalid base64 length".into());
    }
    decode_b64_clean(&clean).map_err(|code| match code {
        B64_PAD => "unexpected base64 padding".into(),
        _ => "invalid base64 character".into(),
    })
}

/// Decodes an image data URI or raw base64 string.
pub fn decode_uri(uri: &str) -> Result<Vec<u8>, Error> {
    let trimmed = uri.trim();
    if let Some(rest) = trimmed.strip_prefix("data:") {
        let (meta, payload) = rest.split_once(',').ok_or("malformed image data URI")?;
        if !meta.contains(";base64") {
            return Err("image data URI must use base64".into());
        }
        return decode_base64(payload);
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return Err("remote image URLs are not fetched; pass a data URI".into());
    }
    decode_base64(trimmed)
}

/// Decodes PNG, JPEG, WebP, or GIF bytes into packed RGB.
pub(crate) fn decode_image(bytes: &[u8]) -> Result<RgbImage, Error> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| format!("failed to decode image: {e}"))?
        .into_rgb8();
    let (width, height) = (img.width() as usize, img.height() as usize);
    if width == 0 || height == 0 {
        return Err("image has zero dimension".into());
    }
    Ok(RgbImage {
        width,
        height,
        pixels: img.into_raw(),
    })
}

/// Deterministic synthetic RGB image for benchmarks and tests.
pub fn synthetic_rgb(width: usize, height: usize, seed: usize) -> RgbImage {
    let mut pixels = vec![0u8; width * height * 3];
    let wm = width.saturating_sub(1).max(1);
    let hm = height.saturating_sub(1).max(1);
    for y in 0..height {
        for x in 0..width {
            let off = (y * width + x) * 3;
            pixels[off] = ((x * 255 / wm + seed * 17) & 255) as u8;
            pixels[off + 1] = ((y * 255 / hm + seed * 31) & 255) as u8;
            pixels[off + 2] = (((x ^ y) + seed * 53) & 255) as u8;
        }
    }
    RgbImage {
        width,
        height,
        pixels,
    }
}

fn py_round(x: f64) -> usize {
    x.round_ties_even().max(0.0) as usize
}

/// Qwen3-VL `smart_resize` returning `(target_h, target_w)`.
pub(crate) fn smart_resize(
    height: usize,
    width: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), Error> {
    if height == 0 || width == 0 {
        return Err("empty image".into());
    }
    let hi = height.max(width) as f64;
    let lo = height.min(width) as f64;
    if hi / lo > 200.0 {
        return Err(format!("image aspect ratio {:.1} exceeds 200", hi / lo).into());
    }
    let f = FACTOR as f64;
    let mut h_bar = py_round(height as f64 / f).max(1) * FACTOR;
    let mut w_bar = py_round(width as f64 / f).max(1) * FACTOR;
    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = ((height as f64 / beta / f).floor() as usize).max(1) * FACTOR;
        w_bar = ((width as f64 / beta / f).floor() as usize).max(1) * FACTOR;
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = ((height as f64 * beta / f).ceil() as usize).max(1) * FACTOR;
        w_bar = ((width as f64 * beta / f).ceil() as usize).max(1) * FACTOR;
    }
    Ok((h_bar, w_bar))
}

fn bicubic_tap(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

struct Coeffs {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    kk: Vec<i32>,
}

fn build_coeffs(in_size: usize, out_size: usize) -> Coeffs {
    let scale = (in_size as f32) as f64 / out_size as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * ksize];
    let mut tmp = vec![0.0f64; ksize];
    let inv = 1.0 / filter_scale;
    let shift = (1u32 << PRECISION) as f64;
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5) as i32).max(0);
        let xmax = ((center + support + 0.5) as i32).min(in_size as i32) - xmin;
        let mut sum = 0.0;
        for x in 0..xmax {
            let w = bicubic_tap(((x + xmin) as f64 - center + 0.5) * inv);
            tmp[x as usize] = w;
            sum += w;
        }
        let row = &mut kk[xx * ksize..(xx + 1) * ksize];
        for x in 0..xmax.max(0) as usize {
            if sum != 0.0 {
                tmp[x] /= sum;
            }
            let v = tmp[x] * shift;
            row[x] = if tmp[x] < 0.0 {
                (-0.5 + v) as i32
            } else {
                (0.5 + v) as i32
            };
        }
        bounds.push((xmin as usize, xmax.max(0) as usize));
    }
    Coeffs { ksize, bounds, kk }
}

#[inline]
fn clip8(v: i32) -> u8 {
    (v >> PRECISION).clamp(0, 255) as u8
}

/// Applies horizontal bicubic resampling to a contiguous stripe of rows.
fn horiz_band(src: &[u8], out: &mut [u8], w: usize, ow: usize, ch: &Coeffs) {
    let half = 1i32 << (PRECISION - 1);
    let in_row = w * 3;
    let out_row = ow * 3;
    for (irow, orow) in src.chunks_exact(in_row).zip(out.chunks_exact_mut(out_row)) {
        for xx in 0..ow {
            let (xmin, xmax) = ch.bounds[xx];
            let k = &ch.kk[xx * ch.ksize..xx * ch.ksize + xmax];
            let (mut s0, mut s1, mut s2) = (half, half, half);
            let slice = &irow[xmin * 3..(xmin + xmax) * 3];
            for (px, &kx) in slice.as_chunks::<3>().0.iter().zip(k) {
                s0 = s0.wrapping_add(i32::from(px[0]) * kx);
                s1 = s1.wrapping_add(i32::from(px[1]) * kx);
                s2 = s2.wrapping_add(i32::from(px[2]) * kx);
            }
            let dst = xx * 3;
            orow[dst] = clip8(s0);
            orow[dst + 1] = clip8(s1);
            orow[dst + 2] = clip8(s2);
        }
    }
}

/// Applies vertical bicubic resampling to a contiguous stripe of rows.
fn vert_band(tmp: &[u8], out: &mut [u8], tw: usize, y_start: usize, cv: &Coeffs, acc: &mut [i32]) {
    let half = 1i32 << (PRECISION - 1);
    let row_bytes = tw * 3;
    for (dy, orow) in out.chunks_exact_mut(row_bytes).enumerate() {
        let yy = y_start + dy;
        let (ymin, ymax) = cv.bounds[yy];
        let k = &cv.kk[yy * cv.ksize..yy * cv.ksize + ymax];
        acc.fill(half);
        for (y, &ky) in k.iter().enumerate() {
            let irow = &tmp[(ymin + y) * row_bytes..(ymin + y + 1) * row_bytes];
            for (a, &px) in acc.iter_mut().zip(irow) {
                *a = a.wrapping_add(i32::from(px) * ky);
            }
        }
        for (o, &a) in orow.iter_mut().zip(acc.iter()) {
            *o = clip8(a);
        }
    }
}

/// Pillow-exact 8-bit fixed-point bicubic resize on packed RGB.
pub(crate) fn resize_bicubic(src: &[u8], w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    assert_eq!(src.len(), w * h * 3);
    if (w, h) == (ow, oh) {
        return src.to_vec();
    }
    let ch = build_coeffs(w, ow);
    let mut cv = build_coeffs(h, oh);
    let y_first = cv.bounds[0].0;
    let y_last = cv.bounds[oh - 1].0 + cv.bounds[oh - 1].1;
    let (tmp, tw) = if ow != w {
        for b in &mut cv.bounds {
            b.0 -= y_first;
        }
        let rows = y_last - y_first;
        let src_band = &src[y_first * w * 3..y_last * w * 3];
        let mut out = vec![0u8; ow * rows * 3];
        let n = workers(rows * ow, 32_768);
        if n <= 1 {
            horiz_band(src_band, &mut out, w, ow, &ch);
        } else {
            let tile_rows = 16usize;
            parallel_chunks(&mut out, tile_rows * ow * 3, n, |idx, out_s| {
                let r0 = idx * tile_rows;
                let r_cnt = out_s.len() / (ow * 3);
                let in_s = &src_band[r0 * w * 3..(r0 + r_cnt) * w * 3];
                horiz_band(in_s, out_s, w, ow, &ch);
            });
        }
        (Cow::Owned(out), ow)
    } else {
        (Cow::Borrowed(src), w)
    };
    if oh == h {
        return tmp.into_owned();
    }
    let mut out = vec![0u8; tw * oh * 3];
    let n = workers(oh * tw, 32_768);
    if n <= 1 {
        let mut acc = vec![0i32; tw * 3];
        vert_band(&tmp, &mut out, tw, 0, &cv, &mut acc);
    } else {
        let tile_rows = 16usize;
        let it = Mutex::new(out.chunks_mut(tile_rows * tw * 3).enumerate());
        let tmp_ref: &[u8] = &tmp;
        let cv = &cv;
        std::thread::scope(|s| {
            for _ in 0..n {
                let it = &it;
                s.spawn(move || {
                    let mut acc = vec![0i32; tw * 3];
                    loop {
                        let next = it.lock().ok().and_then(|mut g| g.next());
                        let Some((idx, out_s)) = next else {
                            break;
                        };
                        vert_band(tmp_ref, out_s, tw, idx * tile_rows, cv, &mut acc);
                    }
                });
            }
        });
    }
    out
}

/// Resizes and encodes an RGB image off the GPU worker.
pub(crate) fn encode_image(
    im: &RgbImage,
    min_pixels: usize,
    max_pixels: usize,
    grid: usize,
) -> Result<EncodedImage, Error> {
    let (th, tw) = smart_resize(im.height, im.width, min_pixels, max_pixels)?;
    let rgb = resize_bicubic(&im.pixels, im.width, im.height, tw, th);
    let (vit_h, vit_w) = (th / PATCH, tw / PATCH);
    let (pos_ids, pos_w) = pos_bilinear(grid, vit_h, vit_w);
    let n = vit_h * vit_w;
    let mut rot_pos = Vec::with_capacity(n * 2);
    for p in 0..n {
        let (pr, pc) = patch_rc(p, vit_w);
        rot_pos.push(pr as u32);
        rot_pos.push(pc as u32);
    }
    Ok(EncodedImage {
        vit_h,
        vit_w,
        rgb,
        pos_ids,
        pos_w,
        rot_pos,
    })
}

/// Maps merge-block patch index `p` to `(row, col)`.
#[inline]
pub(crate) fn patch_rc(p: usize, vit_w: usize) -> (usize, usize) {
    let bw = vit_w / MERGE;
    let blk = p / 4;
    let sub = p & 3;
    ((blk / bw) * 2 + (sub >> 1), (blk % bw) * 2 + (sub & 1))
}

/// Fills patches starting at `p0` with normalized float bytes.
fn patchify_range(rgb: &[u8], out: &mut [[u8; 4]], vit_w: usize, p0: usize) {
    let w_px = vit_w * PATCH;
    for (dp, dst) in out.as_chunks_mut::<1536>().0.iter_mut().enumerate() {
        let (pr, pc) = patch_rc(p0 + dp, vit_w);
        let y0 = pr * PATCH;
        let x0 = pc * PATCH;
        for py in 0..PATCH {
            let row = &rgb[((y0 + py) * w_px + x0) * 3..][..PATCH * 3];
            let base = py * PATCH;
            for (px, ch) in row.as_chunks::<3>().0.iter().enumerate() {
                let r = NORM_LUT[ch[0] as usize];
                let g = NORM_LUT[ch[1] as usize];
                let b = NORM_LUT[ch[2] as usize];
                let o = base + px;
                dst[o] = r;
                dst[256 + o] = r;
                dst[512 + o] = g;
                dst[768 + o] = g;
                dst[1024 + o] = b;
                dst[1280 + o] = b;
            }
        }
    }
}

/// Writes `[N_patches, 1536]` normalized float bytes directly into `out`.
fn patchify_into(rgb: &[u8], out_bytes: &mut [u8], vit_h: usize, vit_w: usize) {
    let n = vit_h * vit_w;
    let (words, _) = out_bytes.as_chunks_mut::<4>();
    let threads = workers(n, 256);
    if threads <= 1 {
        patchify_range(rgb, words, vit_w, 0);
        return;
    }
    let tile_patches = 64usize;
    parallel_chunks(words, tile_patches * 1536, threads, |idx, slice| {
        patchify_range(rgb, slice, vit_w, idx * tile_patches);
    });
}

/// Bilinear `align_corners=True` corner indices and weights for `pos_embed`.
pub(crate) fn pos_bilinear(grid: usize, vit_h: usize, vit_w: usize) -> (Vec<u32>, Vec<f32>) {
    let n = vit_h * vit_w;
    let mut ids = Vec::with_capacity(n * 4);
    let mut weights = Vec::with_capacity(n * 4);
    let gmax = (grid - 1) as f32;
    for p in 0..n {
        let (pr, pc) = patch_rc(p, vit_w);
        let rh = if vit_h > 1 {
            pr as f32 * gmax / (vit_h - 1) as f32
        } else {
            0.0
        };
        let rw = if vit_w > 1 {
            pc as f32 * gmax / (vit_w - 1) as f32
        } else {
            0.0
        };
        let r0 = rh.floor() as usize;
        let c0 = rw.floor() as usize;
        let r1 = (r0 + 1).min(grid - 1);
        let c1 = (c0 + 1).min(grid - 1);
        let dr = rh - r0 as f32;
        let dc = rw - c0 as f32;
        ids.extend_from_slice(&[
            (r0 * grid + c0) as u32,
            (r0 * grid + c1) as u32,
            (r1 * grid + c0) as u32,
            (r1 * grid + c1) as u32,
        ]);
        weights.extend_from_slice(&[
            (1.0 - dr) * (1.0 - dc),
            (1.0 - dr) * dc,
            dr * (1.0 - dc),
            dr * dc,
        ]);
    }
    (ids, weights)
}

/// Packs pre-encoded multi-image inputs for `qwen3_vl::vision_program`.
pub(crate) fn pack_vision(images: &[&EncodedImage]) -> Vec<Tensor> {
    let merged: usize = images.iter().map(|im| im.tokens()).sum();
    let total_patches = merged * 4;
    let mut patches = Tensor::zeros(vec![merged, 4, 1536], DType::F32);
    let mut pos_ids = Tensor::zeros(vec![merged, 4, 4], DType::U32);
    let mut pos_w = Tensor::zeros(vec![merged, 4, 4], DType::F32);
    let mut rot_pos = Tensor::zeros(vec![merged, 4, 2], DType::U32);
    let mut bounds = Vec::with_capacity(total_patches * 2);
    let mut p_off = 0usize;
    let mut id_off = 0usize;
    let mut w_off = 0usize;
    let mut r_off = 0usize;
    let mut patch_base = 0u32;
    for im in images {
        let n = im.patches();
        let p_bytes = n * 1536 * 4;
        patchify_into(
            &im.rgb,
            &mut patches.bytes_mut()[p_off..p_off + p_bytes],
            im.vit_h,
            im.vit_w,
        );
        p_off += p_bytes;
        let ib = as_bytes(&im.pos_ids);
        pos_ids.bytes_mut()[id_off..id_off + ib.len()].copy_from_slice(ib);
        id_off += ib.len();
        let wb = as_bytes(&im.pos_w);
        pos_w.bytes_mut()[w_off..w_off + wb.len()].copy_from_slice(wb);
        w_off += wb.len();
        let rb = as_bytes(&im.rot_pos);
        rot_pos.bytes_mut()[r_off..r_off + rb.len()].copy_from_slice(rb);
        r_off += rb.len();
        let patch_end = patch_base + n as u32;
        for _ in 0..n {
            bounds.push(patch_base);
            bounds.push(patch_end);
        }
        patch_base = patch_end;
    }
    vec![
        patches,
        pos_ids,
        pos_w,
        rot_pos,
        Tensor::u32(vec![merged, 4, 2], bounds),
    ]
}

/// 3D interleaved M-RoPE `(t, h, w)` coordinates for one sequence.
pub(crate) fn mrope_ids(seq_len: usize, spans: &[ImageSpan]) -> Vec<[u32; 3]> {
    let mut out = Vec::with_capacity(seq_len);
    let mut img_idx = 0usize;
    let mut delta: isize = 0;
    for p in 0..seq_len {
        while img_idx < spans.len() && spans[img_idx].pos + spans[img_idx].tokens() <= p {
            let im = spans[img_idx];
            let lh = im.vit_h / MERGE;
            let lw = im.vit_w / MERGE;
            delta += lh.max(lw) as isize - im.tokens() as isize;
            img_idx += 1;
        }
        if img_idx < spans.len() && p >= spans[img_idx].pos {
            let im = spans[img_idx];
            let lw = im.vit_w / MERGE;
            let rel = p - im.pos;
            let r = rel / lw;
            let c = rel % lw;
            let base = (im.pos as isize + delta) as u32;
            out.push([base, base + r as u32, base + c as u32]);
        } else {
            let cp = (p as isize + delta) as u32;
            out.push([cp, cp, cp]);
        }
    }
    out
}

/// Packs variable-length sequences for `qwen3_vl::text_program`.
pub(crate) fn pack_text(seqs: &[&[u32]], spans: &[&[ImageSpan]]) -> Vec<Tensor> {
    let rows: usize = seqs.iter().map(|s| s.len()).sum();
    let mut tokens = Vec::with_capacity(rows);
    let mut positions = Vec::with_capacity(rows * 3);
    let mut bounds = Vec::with_capacity(rows * 2);
    let mut picks = Vec::with_capacity(seqs.len());
    let mut vis_idx = Vec::with_capacity(rows);
    let mut vis_cursor = 0u32;
    for (&seq, &seq_spans) in seqs.iter().zip(spans) {
        let seg_start = tokens.len() as u32;
        let seg_end = seg_start + seq.len() as u32;
        picks.push(seg_end - 1);
        let pos = mrope_ids(seq.len(), seq_spans);
        let mut sp_i = 0usize;
        for (p, &tok) in seq.iter().enumerate() {
            while sp_i < seq_spans.len() && seq_spans[sp_i].pos + seq_spans[sp_i].tokens() <= p {
                sp_i += 1;
            }
            let in_vis = sp_i < seq_spans.len() && p >= seq_spans[sp_i].pos;
            if in_vis {
                tokens.push(u32::MAX);
                vis_idx.push(vis_cursor);
                vis_cursor += 1;
            } else {
                tokens.push(tok);
                vis_idx.push(u32::MAX);
            }
            positions.extend_from_slice(&pos[p]);
            bounds.push(seg_start);
            bounds.push(seg_end);
        }
    }
    vec![
        Tensor::u32(vec![rows], tokens),
        Tensor::u32(vec![rows, 3], positions),
        Tensor::u32(vec![rows, 2], bounds),
        Tensor::u32(vec![seqs.len()], picks),
        Tensor::u32(vec![rows], vis_idx),
    ]
}

/// Formats multimodal messages with the Qwen3-VL-Embedding chat template.
pub(crate) fn format_chat(
    messages: &[EmbedMessage],
    add_generation_prompt: bool,
) -> (String, Vec<&[u8]>) {
    let mut out = String::new();
    let mut images = Vec::new();
    let has_system = messages.first().is_some_and(|m| m.role == "system");
    out.push_str("<|im_start|>system\n");
    if has_system {
        for part in &messages[0].parts {
            if let EmbedPart::Text(t) = part {
                out.push_str(t);
            }
        }
    } else {
        out.push_str(DEFAULT_SYSTEM);
    }
    out.push_str("<|im_end|>\n");
    for msg in messages {
        if msg.role == "system" {
            continue;
        }
        out.push_str("<|im_start|>");
        out.push_str(&msg.role);
        out.push('\n');
        for part in &msg.parts {
            match part {
                EmbedPart::Text(t) => out.push_str(t),
                EmbedPart::Image(bytes) => {
                    images.push(bytes.as_slice());
                    out.push_str("<|vision_start|><|image_pad|><|vision_end|>");
                }
            }
        }
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    (out, images)
}

/// Expands each image placeholder token to its visual span length.
pub(crate) fn expand_images(
    ids: &[u32],
    image_token: u32,
    images: &[EncodedImage],
) -> Result<(Vec<u32>, Vec<ImageSpan>), Error> {
    let extra: usize = images.iter().map(EncodedImage::tokens).sum();
    let mut out = Vec::with_capacity(ids.len() + extra);
    let mut spans = Vec::with_capacity(images.len());
    let mut img_idx = 0usize;
    for &tok in ids {
        if tok == image_token {
            let im = images
                .get(img_idx)
                .ok_or("more image placeholders than images")?;
            spans.push(ImageSpan {
                pos: out.len(),
                vit_h: im.vit_h,
                vit_w: im.vit_w,
            });
            for _ in 0..im.tokens() {
                out.push(image_token);
            }
            img_idx += 1;
        } else {
            out.push(tok);
        }
    }
    if img_idx != images.len() {
        return Err("fewer image placeholders than images".into());
    }
    Ok((out, spans))
}
