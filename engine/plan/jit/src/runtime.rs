//! Calls for compiled bodies: buffers, math, shuffles and tiles.

use std::sync::atomic::{AtomicU32, Ordering};
use tessera_codegen::Precision;
use tessera_codegen::form::{BF16_FORM, Bf16, F16, F32, F32_FORM, Form, Fp8, LOADS, Load, U8, U32};
use tessera_host::lockstep::{self, THREADS, Tiles};

/// A buffer as the compiled code sees it.
#[repr(C)]
pub struct Buf {
    pub ptr: *mut u8,
    pub len: usize,
}

/// A launch argument: buffer index and skip, or scalar bits.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Arg {
    pub buf: usize,
    pub skip: usize,
    pub bits: u32,
}

/// Values one launch hands every group.
#[repr(C)]
pub struct Ctx {
    pub bufs: *const Buf,
    pub args: *const Arg,
    pub threads: usize,
    pub lanes: usize,
    pub precision: *const Precision,
}

/// A compiled body over groups `lo..hi` of `ctx`.
pub type Entry = unsafe extern "C" fn(*const Ctx, usize, usize);

/// Set by a helper that could not do its work.
static FAILED: AtomicU32 = AtomicU32::new(0);

const PAST_END: u32 = 1;

/// Takes the failure code a launch left, if any.
pub fn failure() -> Option<&'static str> {
    match FAILED.swap(0, Ordering::Relaxed) {
        0 => None,
        PAST_END => Some("a read or write past a buffer's end"),
        _ => Some("a compiled body failed"),
    }
}

/// `len` bytes at `p`, when `end` lies within them.
unsafe fn within<'a>(p: *const u8, len: usize, end: usize) -> Option<&'a [u8]> {
    if end > len {
        FAILED.store(PAST_END, Ordering::Relaxed);
        return None;
    }
    // SAFETY: the caller's buffer holds `len` bytes at `p`.
    Some(unsafe { std::slice::from_raw_parts(p, len) })
}

macro_rules! readers {
    ($($name:ident = $read:ident : $ty:ty, $form:expr;)*) => {$(
        pub unsafe extern "C" fn $name(p: *const u8, len: usize, at: usize) -> $ty {
            match unsafe { within(p, len, Form::bytes(&$form, at + 1)) } {
                Some(b) => lockstep::$read(b, at),
                None => Default::default(),
            }
        }
    )*};
}

readers! {
    rd_f32 = rd_f32: f32, F32;
    rd_bf16 = rd_bf16: f32, Bf16;
    rd_f16 = rd_f16: f32, F16;
    rd_fp8 = rd_fp8: f32, Fp8;
    rd_u32 = rd_u32: u32, U32;
    rd_u8 = rd_u8: u32, U8;
}

/// Element `at` of a buffer in the registered form `form`.
pub unsafe extern "C" fn rd_form(form: usize, p: *const u8, len: usize, at: usize) -> f32 {
    let form = LOADS[form];
    match unsafe { within(p, len, form.bytes(at + 1)) } {
        Some(b) => form.unpack(b, at),
        None => 0.0,
    }
}

/// `lockstep::dot_form` after bounds checks on both rows.
#[allow(clippy::too_many_arguments, reason = "two rows and a length")]
unsafe fn dot_in(
    form: &dyn Load,
    a: *const u8,
    a_len: usize,
    a_at: usize,
    b: *const u8,
    b_len: usize,
    b_at: usize,
    n: usize,
) -> f32 {
    let (Some(a), Some(b)) = (
        unsafe { within(a, a_len, F32_FORM.bytes(a_at + n)) },
        unsafe { within(b, b_len, form.bytes(b_at + n)) },
    ) else {
        return 0.0;
    };
    lockstep::dot_form(form, a, a_at, b, b_at, n)
}

macro_rules! dots {
    ($($name:ident = $form:expr;)*) => {$(
        #[allow(clippy::too_many_arguments, reason = "two rows and a length")]
        pub unsafe extern "C" fn $name(
            a: *const u8,
            a_len: usize,
            a_at: usize,
            b: *const u8,
            b_len: usize,
            b_at: usize,
            n: usize,
        ) -> f32 {
            unsafe { dot_in($form, a, a_len, a_at, b, b_len, b_at, n) }
        }
    )*};
}

dots! {
    dot_f32 = F32_FORM;
    dot_bf16 = BF16_FORM;
}

/// A dot against a row in the registered form `form`.
#[allow(clippy::too_many_arguments, reason = "a form, two rows and a length")]
pub unsafe extern "C" fn dot_form(
    form: usize,
    a: *const u8,
    a_len: usize,
    a_at: usize,
    b: *const u8,
    b_len: usize,
    b_at: usize,
    n: usize,
) -> f32 {
    unsafe { dot_in(LOADS[form], a, a_len, a_at, b, b_len, b_at, n) }
}

macro_rules! writers {
    ($($name:ident = $write:ident : $ty:ty, $form:expr;)*) => {$(
        pub unsafe extern "C" fn $name(p: *mut u8, len: usize, at: usize, v: $ty) {
            if unsafe { within(p, len, Form::bytes(&$form, at + 1)) }.is_some() {
                // SAFETY: checked above.
                lockstep::$write(unsafe { std::slice::from_raw_parts_mut(p, len) }, at, v);
            }
        }
    )*};
}

writers! {
    st_f32 = st_f32: f32, F32;
    st_bf16 = st_bf16: f32, Bf16;
    st_fp8 = st_fp8: f32, Fp8;
    st_u32 = st_u32: u32, U32;
    st_u8 = st_u8: u32, U8;
}

macro_rules! tiles {
    ($($name:ident = $form:expr;)*) => {$(
        #[allow(clippy::too_many_arguments, reason = "three buffers and their offsets")]
        pub unsafe extern "C" fn $name(
            out: *mut u8,
            out_len: usize,
            at: usize,
            stride: usize,
            x: *const u8,
            x_len: usize,
            x_at: usize,
            w: *const u8,
            w_len: usize,
            w_at: usize,
            k: usize,
            rows: usize,
            cols: usize,
        ) {
            let cols = cols.min(lockstep::BLOCK);
            let last = (at + rows.saturating_sub(1) * stride + cols) * 4;
            let (Some(_), Some(x), Some(w)) = (
                unsafe { within(out, out_len, last) },
                unsafe { within(x, x_len, (x_at + rows * k) * 4) },
                unsafe { within(w, w_len, $form.bytes(w_at + cols * k)) },
            ) else {
                return;
            };
            // SAFETY: checked above.
            let out = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
            let d = lockstep::Dots { at, stride, x_at, w_at, k, cols };
            lockstep::dots_form($form, out, x, w, &d, rows);
        }
    )*};
}

tiles! {
    dots_f32 = F32_FORM;
    dots_bf16 = BF16_FORM;
}

/// One stage over a row. Only two stages read `b`.
#[allow(clippy::too_many_arguments, reason = "two rows, a stage and a length")]
pub unsafe extern "C" fn stage_f32(
    out: *mut u8,
    out_len: usize,
    at: usize,
    n: usize,
    code: u32,
    s: f32,
    b: *const u8,
    b_len: usize,
    b_at: usize,
) {
    if unsafe { within(out, out_len, (at + n) * 4) }.is_none() {
        return;
    }
    let reads = tessera_codegen::stage::Row::from_code(code).is_some_and(|r| r.reads());
    let operand = match reads {
        true => match unsafe { within(b, b_len, (b_at + n) * 4) } {
            Some(b) => Some((b, b_at)),
            None => return,
        },
        false => None,
    };
    // SAFETY: checked above. Operands never share the output's buffer.
    let out = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
    lockstep::stage_f32(out, at, n, (code, s), operand);
}

/// One query's attention. Each buffer comes as pointer, length, start.
#[allow(
    clippy::too_many_arguments,
    reason = "five buffers and the query's shape"
)]
pub unsafe extern "C" fn attend_f32(
    out: *mut u8,
    out_len: usize,
    out_at: usize,
    q: *const u8,
    q_len: usize,
    q_at: usize,
    k: *const u8,
    k_len: usize,
    k_at: usize,
    v: *const u8,
    v_len: usize,
    v_at: usize,
    ids: *const u8,
    ids_len: usize,
    list_at: usize,
    stride: usize,
    from: usize,
    to: usize,
    width: usize,
    scale: f32,
    picks: usize,
) {
    let last = to.saturating_sub(1) * stride + width;
    let (Some(_), Some(q), Some(k), Some(v), Some(ids)) = (
        unsafe { within(out, out_len, (out_at + width) * 4) },
        unsafe { within(q, q_len, (q_at + width) * 4) },
        unsafe { within(k, k_len, (k_at + last) * 4) },
        unsafe { within(v, v_len, (v_at + last) * 4) },
        unsafe { within(ids, ids_len, (list_at + picks) * 4) },
    ) else {
        return;
    };
    // SAFETY: checked above. The output has its own buffer.
    let out = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
    let rows = tessera_host::attend::Rows { out, q, k, v, ids };
    let query = tessera_host::attend::Query {
        out_at,
        q_at,
        k_at,
        v_at,
        stride,
        from,
        to,
        width,
        scale,
        list_at,
        picks,
    };
    tessera_host::attend::attend_f32(rows, &query);
}

#[allow(
    clippy::too_many_arguments,
    reason = "two rows, two scales and a length"
)]
pub unsafe extern "C" fn mix_f32(
    out: *mut u8,
    out_len: usize,
    at: usize,
    keep: f32,
    p: f32,
    b: *const u8,
    b_len: usize,
    b_at: usize,
    n: usize,
) {
    let fits = unsafe { within(out, out_len, (at + n) * 4) }.is_some();
    let Some(b) = (unsafe { within(b, b_len, (b_at + n) * 4) }) else {
        return;
    };
    if fits {
        // SAFETY: checked above.
        let out = unsafe { std::slice::from_raw_parts_mut(out, out_len) };
        lockstep::mix_f32(out, at, keep, p, b, b_at, n);
    }
}

pub unsafe extern "C" fn word(p: *const u8, len: usize, at: usize) -> u32 {
    match unsafe { within(p, len, at + 4) } {
        Some(b) => lockstep::word(b, at),
        None => 0,
    }
}

pub unsafe extern "C" fn half(p: *const u8, len: usize, at: usize) -> u32 {
    match unsafe { within(p, len, at + 2) } {
        Some(b) => lockstep::half(b, at),
        None => 0,
    }
}

macro_rules! unary {
    ($($name:ident = $f:expr;)*) => {$(
        pub extern "C" fn $name(x: f32) -> f32 {
            let f: fn(f32) -> f32 = $f;
            f(x)
        }
    )*};
}

unary! {
    exp = f32::exp;
    tanh = f32::tanh;
    cos = f32::cos;
    sin = f32::sin;
}

pub extern "C" fn max(a: f32, b: f32) -> f32 {
    a.max(b)
}

pub extern "C" fn log(x: f32) -> f32 {
    x.ln()
}

pub extern "C" fn pow(a: f32, b: f32) -> f32 {
    a.powf(b)
}

pub extern "C" fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    x.clamp(lo, hi)
}

pub extern "C" fn f16(w: u32, low: u32) -> f32 {
    lockstep::f16(w, low)
}

pub extern "C" fn e4m3(w: u32, low: u32) -> f32 {
    lockstep::e4m3(w, low)
}

pub extern "C" fn e2m1(w: u32, low: u32) -> f32 {
    lockstep::e2m1(w, low)
}

pub extern "C" fn bf16_pair(hi: f32, lo: f32) -> u32 {
    lockstep::bf16_pair(hi, lo)
}

pub extern "C" fn round_f16(x: f32) -> f32 {
    lockstep::round_f16(x)
}

pub extern "C" fn round_e4m3(x: f32) -> f32 {
    lockstep::round_e4m3(x)
}

pub extern "C" fn to_e4m3(x: f32) -> u32 {
    lockstep::to_e4m3(x)
}

/// Lane arrays, as compiled code keeps them on its stack.
type Lanes<T> = [T; THREADS];

/// `out` takes each lane's value from the lane `m` away.
pub unsafe extern "C" fn shfl_xor(
    v: *const Lanes<f32>,
    m: *const Lanes<u32>,
    out: *mut Lanes<f32>,
    n: usize,
    lanes: usize,
) {
    // SAFETY: the compiled code passes its own lane arrays.
    unsafe { *out = lockstep::shfl_xor(&*v, &*m, n, lanes) };
}

pub unsafe extern "C" fn sum32(
    v: *mut Lanes<f32>,
    mask: *const Lanes<bool>,
    n: usize,
    lanes: usize,
) {
    // SAFETY: the compiled code passes its own lane arrays.
    unsafe { lockstep::sum32(&mut *v, &*mask, n, lanes) };
}

pub extern "C" fn tiles_new(rows: usize, cols: usize, n: usize, lanes: usize) -> *mut Tiles {
    Box::into_raw(Box::new(Tiles::new(rows, cols, n, lanes)))
}

pub unsafe extern "C" fn tiles_free(t: *mut Tiles) {
    // SAFETY: `tiles_new` made it. Nothing holds it past the group.
    drop(unsafe { Box::from_raw(t) });
}

pub unsafe extern "C" fn tiles_zero(t: *mut Tiles) {
    unsafe { (*t).zero() }
}

pub unsafe extern "C" fn tiles_per(t: *const Tiles) -> usize {
    unsafe { (*t).per() }
}

pub unsafe extern "C" fn tiles_row(t: *const Tiles, lane: usize, e: usize) -> u32 {
    unsafe { (*t).row(lane, e) }
}

pub unsafe extern "C" fn tiles_scale(t: *mut Tiles, lane: usize, e: usize, by: f32) {
    unsafe { (*t).scale(lane, e, by) }
}

/// Lane sum `e`: row, column, value, and whether live.
pub unsafe extern "C" fn tiles_each(
    t: *const Tiles,
    lane: usize,
    e: usize,
    live: u8,
    r: *mut u32,
    c: *mut u32,
    v: *mut f32,
) -> u8 {
    let (on, row, col, value) = unsafe { (*t).each(lane, e, live != 0) };
    unsafe {
        *r = row;
        *c = col;
        *v = value;
    }
    u8::from(on)
}

/// Group memory that compiled code passes.
unsafe fn group<'a>(p: *const f32, len: usize) -> &'a [f32] {
    // SAFETY: compiled code passes its own memory.
    unsafe { std::slice::from_raw_parts(p, len) }
}

pub unsafe extern "C" fn tiles_hold(
    h: *mut Tiles,
    a: *const f32,
    a_len: usize,
    a_at: *const Lanes<u32>,
    stride: *const Lanes<u32>,
    mask: *const Lanes<bool>,
    n: usize,
) {
    // SAFETY: compiled code passes its own lane arrays.
    unsafe { (*h).hold(group(a, a_len), &*a_at, &*stride, &*mask, n) }
}

#[allow(clippy::too_many_arguments, reason = "mirrors the tile statement")]
pub unsafe extern "C" fn tiles_block_held(
    t: *mut Tiles,
    h: *const Tiles,
    b: *const f32,
    b_len: usize,
    b_at: *const Lanes<u32>,
    stride: *const Lanes<u32>,
    depth: usize,
    mask: *const Lanes<bool>,
    n: usize,
    precision: *const Precision,
) {
    // SAFETY: the tile objects outlive the group.
    unsafe {
        let b = group(b, b_len);
        (*t).block_held(&*h, b, &*b_at, &*stride, depth, &*mask, n, *precision);
    }
}

#[allow(clippy::too_many_arguments, reason = "mirrors the tile statement")]
pub unsafe extern "C" fn tiles_block(
    t: *mut Tiles,
    a: *const f32,
    a_len: usize,
    a_at: *const Lanes<u32>,
    b: *const f32,
    b_len: usize,
    b_at: *const Lanes<u32>,
    stride: *const Lanes<u32>,
    depth: usize,
    parts: usize,
    apart: *const Lanes<u32>,
    mask: *const Lanes<bool>,
    n: usize,
    precision: *const Precision,
) {
    // SAFETY: compiled code passes its own memory and lane arrays.
    unsafe {
        let (a, b) = (
            std::slice::from_raw_parts(a, a_len),
            std::slice::from_raw_parts(b, b_len),
        );
        let parts = (parts, &*apart);
        (*t).block(
            a, &*a_at, b, &*b_at, &*stride, depth, parts, &*mask, n, *precision,
        );
    }
}
