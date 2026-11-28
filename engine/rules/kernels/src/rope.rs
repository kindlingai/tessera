//! Rotary positions: a table of angles, then the rotation.

use tessera_codegen::prelude::*;
use tessera_graph::{DType, Op, Order, Pairs, Stage, View};
use tessera_lower::form::{F32_FORM, Load as Stored, plain, writable};
use tessera_lower::{
    Draft, Home, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Term, ToArg, Tree,
    knob, sites,
};

knob!("rope", "threads", 1, 1024);

fn line_or_grid(len: usize, threads: usize) -> Shape {
    match threads {
        0 => Shape::Line(len),
        t => Shape::Grid {
            across: len.div_ceil(t),
            down: 1,
            threads: t,
        },
    }
}

/// Maximum pairs per lane cached in registers for fused norm-rope.
const ROPE_CACHE: u32 = 8;

/// Per-lane register slots for cached norm-rope pairs.
const ROPE_SLOTS: usize = (ROPE_CACHE * 2) as usize;

/// Trip count cached in registers, or zero when too wide.
fn rope_trips(width: usize) -> u32 {
    let trips = (width / 2).div_ceil(LANES);
    if trips <= ROPE_CACHE as usize {
        trips as u32
    } else {
        0
    }
}

/// One thread per position, head and pair, rotating in place.
#[kernel]
fn rope<const ADJACENT: bool>(
    mut x: BufMut<F32>,
    freqs: Buf<F32>,
    positions: u32,
    heads: u32,
    width: u32,
) {
    let pairs = width / 2;
    if gid() >= positions * heads * pairs {
        return;
    }
    let pair = gid() % pairs;
    let h = (gid() / pairs) % heads;
    let s = gid() / pairs / heads;
    let at = ((s * heads + h) * width) as u64;
    let c = freqs[(s * width + 2 * pair) as u64];
    let sn = freqs[(s * width + 2 * pair + 1) as u64];
    // Pairs lie side by side, or one in each half.
    let a = if ADJACENT { 2 * pair } else { pair };
    let b = if ADJACENT { a + 1 } else { pairs + pair };
    let x0 = x[at + a as u64];
    let x1 = x[at + b as u64];
    x[at + a as u64] = x0 * c - x1 * sn;
    x[at + b as u64] = x0 * sn + x1 * c;
}

/// One thread per position and pair: its cosine and sine.
#[kernel]
fn angles(
    positions: Buf<U32>,
    mut out: BufMut<F32>,
    count: u32,
    width: u32,
    theta: f32,
    turned: u32,
) {
    let pairs = width / 2;
    let at = gid() / pairs;
    let pair = gid() % pairs;
    if at >= count {
        return;
    }
    let turn = pow(theta, -((2 * pair) as f32) / width as f32);
    // Pairs past `turned` keep angle zero.
    let angle = if pair < turned {
        positions[at as u64] as f32 * turn
    } else {
        0.0
    };
    out[(at * width + 2 * pair) as u64] = cos(angle);
    out[(at * width + 2 * pair + 1) as u64] = sin(angle);
}

/// Each position's cosine and sine, per pair.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    angles,
    threads = line_or_grid(self.count * self.width / 2, self.threads),
    shared,
)]
pub(crate) struct Angles {
    pub count: usize,
    pub width: usize,
    pub theta: f32,
    pub turned: usize,
    pub threads: usize,
}

/// One thread per position and pair: multi-axis cosine and sine.
#[kernel]
fn mrope_angles<const INTERLEAVED: bool>(
    coords: Buf<U32>,
    mut out: BufMut<F32>,
    count: u32,
    width: u32,
    axes: u32,
    theta: f32,
    s0: u32,
    s1: u32,
    s2: u32,
) {
    let pairs = width / 2;
    let at = gid() / pairs;
    let pair = gid() % pairs;
    if at >= count {
        return;
    }
    let mut axis = 0u32;
    let mut local = pair;
    let mut denom = width;
    if INTERLEAVED {
        let rem = pair % 3;
        if rem == 1 && pair < s1 * 3 {
            axis = 1;
        } else {
            if rem == 2 && pair < s2 * 3 {
                axis = 2;
            }
        }
    } else {
        if pair < s0 {
            axis = 0;
            local = pair;
            denom = 2 * s0;
        } else {
            if pair < s0 + s1 {
                axis = 1;
                local = pair - s0;
                denom = 2 * s1;
            } else {
                axis = 2;
                local = pair - s0 - s1;
                denom = 2 * s2;
            }
        }
    }
    let pos = coords[(at * axes + axis) as u64] as f32;
    let angle = pos * pow(theta, -((2 * local) as f32) / denom as f32);
    out[(at * width + 2 * pair) as u64] = cos(angle);
    out[(at * width + 2 * pair + 1) as u64] = sin(angle);
}

/// Multi-axis cosine and sine per position and pair.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    mrope_angles,
    variant = self.interleaved,
    name = match self.interleaved {
        true => "mrope_interleaved_angles",
        false => "mrope_angles",
    },
    threads = self.count * self.width / 2,
    shared,
)]
pub(crate) struct MropeAngles {
    pub count: usize,
    pub width: usize,
    pub axes: usize,
    pub theta: f32,
    pub s0: usize,
    pub s1: usize,
    pub s2: usize,
    pub interleaved: bool,
}

/// Rotates pairs in place, by the second input's angles.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    rope,
    variant = self.adjacent,
    name = match self.adjacent {
        true => "rope_adjacent",
        false => "rope",
    },
    threads = line_or_grid(self.positions * self.heads * self.width / 2, self.threads),
    overwrites,
)]
pub struct Rope {
    pub positions: usize,
    pub heads: usize,
    pub width: usize,
    pub adjacent: bool,
    pub threads: usize,
}

/// One warp per head: RMS norm followed by rotary embedding.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn norm_rope<const ADJACENT: bool>(
    #[form(x_form)] x: Buf<Dyn>,
    freqs: Buf<F32>,
    #[form(w_form)] w: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    positions: u32,
    heads: u32,
    width: u32,
    eps: f32,
    #[build] x_form: &'static dyn Stored,
    #[build] w_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
    #[build] weighted: bool,
    #[build] use_cache: bool,
    #[build] cached: u32,
    #[build] exact: bool,
) {
    let row = gid() / LANES as u32;
    if row >= positions * heads {
        return;
    }
    let s = row / heads;
    let pairs = width / 2;
    let at = row as u64 * width as u64;
    let f_at = s as u64 * width as u64;
    let mut vals = [0.0f32; ROPE_SLOTS];
    let mut acc = 0.0f32;
    if use_cache {
        for t in (0..cached).unroll(ROPE_CACHE) {
            let p = lane() + t * LANES as u32;
            if exact || p < pairs {
                let a = if ADJACENT { 2 * p } else { p };
                let b = if ADJACENT { a + 1 } else { pairs + p };
                let x0 = x[at + a as u64];
                let x1 = x[at + b as u64];
                let idx = (t * 2) as usize;
                vals[idx] = x0;
                vals[idx + 1] = x1;
                acc += x0 * x0 + x1 * x1;
            }
        }
    } else {
        for p in (lane()..pairs).by(LANES as u32) {
            let a = if ADJACENT { 2 * p } else { p };
            let b = if ADJACENT { a + 1 } else { pairs + p };
            let x0 = x[at + a as u64];
            let x1 = x[at + b as u64];
            acc += x0 * x0 + x1 * x1;
        }
    }
    acc = sum32(acc);
    let inv = rsqrt(acc / width as f32 + eps);
    if use_cache {
        for t in (0..cached).unroll(ROPE_CACHE) {
            let p = lane() + t * LANES as u32;
            if exact || p < pairs {
                let a = if ADJACENT { 2 * p } else { p };
                let b = if ADJACENT { a + 1 } else { pairs + p };
                let idx = (t * 2) as usize;
                let mut y0 = vals[idx] * inv;
                let mut y1 = vals[idx + 1] * inv;
                if weighted {
                    y0 = y0 * w[a as u64];
                    y1 = y1 * w[b as u64];
                }
                let c = freqs[f_at + (2 * p) as u64];
                let sn = freqs[f_at + (2 * p + 1) as u64];
                out[at + a as u64] = y0 * c - y1 * sn;
                out[at + b as u64] = y0 * sn + y1 * c;
            }
        }
    } else {
        for p in (lane()..pairs).by(LANES as u32) {
            let a = if ADJACENT { 2 * p } else { p };
            let b = if ADJACENT { a + 1 } else { pairs + p };
            let mut y0 = x[at + a as u64] * inv;
            let mut y1 = x[at + b as u64] * inv;
            if weighted {
                y0 = y0 * w[a as u64];
                y1 = y1 * w[b as u64];
            }
            let c = freqs[f_at + (2 * p) as u64];
            let sn = freqs[f_at + (2 * p + 1) as u64];
            out[at + a as u64] = y0 * c - y1 * sn;
            out[at + b as u64] = y0 * sn + y1 * c;
        }
    }
}

/// Fused RMS norm and rotary embedding per head.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = norm_rope_name(self),
    launches = norm_rope_launch,
    shape = Shape::Line(self.positions * self.heads * LANES),
    work = self.positions * self.heads * self.width,
    needs = Needs::lanes(),
    // Rows sum alike at any count. Other kernels differ.
    order = Order::PerConfig,
    accepts = norm_rope_accepts,
    emits = norm_rope_emits,
    source = norm_rope_source,
    load = norm_rope_load,
    unproven,
)]
pub(crate) struct NormRope {
    pub positions: usize,
    pub heads: usize,
    pub width: usize,
    pub eps: f32,
    pub adjacent: bool,
    pub weighted: bool,
    pub x_form: &'static dyn Stored,
    pub w_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

fn norm_rope_name(k: &NormRope) -> &'static str {
    match (k.adjacent, k.weighted) {
        (false, true) => "norm_rope",
        (false, false) => "rms_rope",
        (true, true) => "norm_rope_adj",
        (true, false) => "rms_rope_adj",
    }
}

fn norm_rope_accepts(k: &NormRope, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.x_form.name(),
        1 => form.name() == F32_FORM.name(),
        2 if k.weighted => form.name() == k.w_form.name(),
        _ => !form.narrow(),
    }
}

fn norm_rope_emits(k: &NormRope, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn norm_rope_body(k: &NormRope) -> tessera_codegen::ir::Body {
    let cached = rope_trips(k.width);
    let exact = (k.width / 2).is_multiple_of(LANES);
    let mut body = match k.adjacent {
        true => norm_rope_ir::<true>(
            k.x_form,
            k.w_form,
            k.out_form,
            k.weighted,
            cached > 0,
            cached,
            exact,
        ),
        false => norm_rope_ir::<false>(
            k.x_form,
            k.w_form,
            k.out_form,
            k.weighted,
            cached > 0,
            cached,
            exact,
        ),
    };
    if !k.weighted {
        body.params.retain(|p| p.name() != "w");
    }
    body
}

fn norm_rope_launch(k: &NormRope, step: &Step) -> Vec<Launch> {
    let body = norm_rope_body(k);
    let scalar = |name: &str| match name {
        "positions" => k.positions.arg(),
        "heads" => k.heads.arg(),
        "width" => k.width.arg(),
        "eps" => k.eps.arg(),
        other => panic!("norm_rope has no field `{other}`"),
    };
    vec![Launch::of(
        &body,
        step,
        false,
        &scalar,
        Shape::Line(k.positions * k.heads * LANES),
    )]
}

fn norm_rope_source(k: &NormRope) -> Source {
    let body = norm_rope_body(k);
    let base = k.name();
    let cached = rope_trips(k.width);
    let mut tag = String::new();
    if cached > 0 {
        tag.push_str(&format!("_c{cached}"));
        if (k.width / 2).is_multiple_of(LANES) {
            tag.push('x');
        }
    }
    let mut forms = vec![k.x_form.name()];
    if k.weighted {
        forms.push(k.w_form.name());
    }
    forms.push(k.out_form.name());
    let entry = match forms.iter().all(|&f| f == "f32") {
        true => format!("plan_{base}{tag}"),
        false => format!("plan_{base}{tag}_{}", forms.join("_")),
    };
    Source::named(body, entry)
}

fn norm_rope_load(k: &NormRope, _: usize) -> Option<Load> {
    let body = norm_rope_body(k);
    let elems = k.positions * k.heads * k.width;
    let x_bytes = k.x_form.bytes(elems) as f64;
    let cached = rope_trips(k.width) > 0;
    let mut reads = if cached {
        vec![x_bytes]
    } else {
        vec![x_bytes, x_bytes]
    };
    reads.push(4.0 * (k.positions * k.width) as f64);
    if k.weighted {
        reads.push(k.w_form.bytes(k.width) as f64);
    }
    let trips = if cached {
        1.0
    } else {
        (k.width / 2).div_ceil(LANES) as f64
    };
    Some(Load {
        threads: k.positions * k.heads * LANES,
        group: 256,
        trips,
        trip: body.trip(),
        reads,
        written: k.out_form.bytes(elems) as f64,
        flight: k.x_form.bytes(2) as f64 + 8.0,
        ..Load::default()
    })
}

/// Whether `form` supports four-wide float vector loads and stores.
fn float_wide(form: &dyn Stored) -> bool {
    matches!(
        form.dtype(),
        DType::F32 | DType::Bf16 | DType::F16 | DType::Fp8
    )
}

/// One warp per 128-wide head: vectorized in-place norm and RoPE.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn norm_rope4(
    #[form(x_form)] mut x: BufMut<Dyn>,
    freqs: Buf<F32>,
    #[form(w_form)] w: Buf<Dyn>,
    positions: u32,
    heads: u32,
    eps: f32,
    #[build] x_form: &'static dyn Stored,
    #[build] w_form: &'static dyn Stored,
    #[build] weighted: bool,
) {
    let row = gid() / LANES as u32;
    if row >= positions * heads {
        return;
    }
    let s = row / heads;
    let l = lane();
    let col = l * 4;
    let at = row as u64 * 128;
    let v = x.load4(at + col as u64);
    let mut acc = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    acc = sum32(acc);
    let inv = rsqrt(acc / 128.0f32 + eps);
    let mut y0 = v.x * inv;
    let mut y1 = v.y * inv;
    let mut y2 = v.z * inv;
    let mut y3 = v.w * inv;
    if weighted {
        let g = w.load4(col as u64);
        y0 *= g.x;
        y1 *= g.y;
        y2 *= g.z;
        y3 *= g.w;
    }
    let p0 = shfl_xor(y0, 16);
    let p1 = shfl_xor(y1, 16);
    let p2 = shfl_xor(y2, 16);
    let p3 = shfl_xor(y3, 16);
    let p4 = (l % 16) * 4;
    let f_at = s as u64 * 128 + (2 * p4) as u64;
    let f01 = freqs.load4(f_at);
    let f23 = freqs.load4(f_at + 4);
    let sgn = if l < 16 { -1.0f32 } else { 1.0f32 };
    let mut o = F32x4::zero();
    o.x = y0 * f01.x + sgn * p0 * f01.y;
    o.y = y1 * f01.z + sgn * p1 * f01.w;
    o.z = y2 * f23.x + sgn * p2 * f23.y;
    o.w = y3 * f23.z + sgn * p3 * f23.w;
    x.store4(at + col as u64, o);
}

/// Vectorized in-place RMS norm and half-split RoPE for `width == 128`.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.weighted { "norm_rope4" } else { "rms_rope4" },
    launches = norm_rope4_launch,
    shape = Shape::Line(self.positions * self.heads * LANES),
    work = self.positions * self.heads * 128,
    needs = Needs::lanes(),
    // Rows sum alike at any count. Other kernels differ.
    order = Order::PerConfig,
    accepts = norm_rope4_accepts,
    emits = norm_rope4_emits,
    source = norm_rope4_source,
    load = norm_rope4_load,
    overwrites,
    unproven,
)]
pub(crate) struct NormRope4 {
    pub positions: usize,
    pub heads: usize,
    pub eps: f32,
    pub weighted: bool,
    pub x_form: &'static dyn Stored,
    pub w_form: &'static dyn Stored,
}

fn norm_rope4_accepts(k: &NormRope4, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.x_form.name(),
        1 => form.name() == F32_FORM.name(),
        2 if k.weighted => form.name() == k.w_form.name(),
        _ => !form.narrow(),
    }
}

fn norm_rope4_emits(k: &NormRope4, form: &dyn Stored) -> bool {
    form.name() == k.x_form.name()
}

fn norm_rope4_body(k: &NormRope4) -> tessera_codegen::ir::Body {
    let mut body = norm_rope4_ir(k.x_form, k.w_form, k.weighted);
    if !k.weighted {
        body.params.retain(|p| p.name() != "w");
    }
    body
}

fn norm_rope4_launch(k: &NormRope4, step: &Step) -> Vec<Launch> {
    let body = norm_rope4_body(k);
    let scalar = |name: &str| match name {
        "positions" => k.positions.arg(),
        "heads" => k.heads.arg(),
        "eps" => k.eps.arg(),
        other => panic!("norm_rope4 has no field `{other}`"),
    };
    vec![Launch::of(
        &body,
        step,
        true,
        &scalar,
        Shape::Line(k.positions * k.heads * LANES),
    )]
}

fn norm_rope4_source(k: &NormRope4) -> Source {
    let body = norm_rope4_body(k);
    let base = k.name();
    let mut forms = vec![k.x_form.name()];
    if k.weighted {
        forms.push(k.w_form.name());
    }
    let entry = match forms.iter().all(|&f| f == "f32") {
        true => format!("plan_{base}"),
        false => format!("plan_{base}_{}", forms.join("_")),
    };
    Source::named(body, entry)
}

fn norm_rope4_load(k: &NormRope4, _: usize) -> Option<Load> {
    let body = norm_rope4_body(k);
    let elems = k.positions * k.heads * 128;
    let x_bytes = k.x_form.bytes(elems) as f64;
    let mut reads = vec![x_bytes, 4.0 * (k.positions * 128) as f64];
    if k.weighted {
        reads.push(k.w_form.bytes(128) as f64);
    }
    Some(Load {
        threads: k.positions * k.heads * LANES,
        group: 256,
        trips: 1.0,
        trip: body.trip(),
        reads,
        written: x_bytes,
        flight: k.x_form.bytes(4) as f64 + 32.0,
        ..Load::default()
    })
}

/// Fused strided slice and rotary embedding into a dense output.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn slice_rope(
    #[form(in_form)] x: Buf<Dyn>,
    freqs: Buf<F32>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    positions: u32,
    heads: u32,
    width: u32,
    src_stride: u32,
    col_off: u32,
    #[build] in_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
    #[build] wide: bool,
) {
    let pairs = width / 2;
    if wide {
        let pairs4 = pairs / 4;
        if gid() >= positions * heads * pairs4 {
            return;
        }
        let p4 = gid() % pairs4;
        let h = (gid() / pairs4) % heads;
        let s = gid() / (pairs4 * heads);
        let p = p4 * 4;
        let src_at = (s * src_stride + col_off + h * width) as u64;
        let dst_at = ((s * heads + h) * width) as u64;
        let f_at = (s * width + 2 * p) as u64;
        let a = x.load4(src_at + p as u64);
        let b = x.load4(src_at + (pairs + p) as u64);
        let f01 = freqs.load4(f_at);
        let f23 = freqs.load4(f_at + 4);
        let mut oa = F32x4::zero();
        let mut ob = F32x4::zero();
        oa.x = a.x * f01.x - b.x * f01.y;
        ob.x = a.x * f01.y + b.x * f01.x;
        oa.y = a.y * f01.z - b.y * f01.w;
        ob.y = a.y * f01.w + b.y * f01.z;
        oa.z = a.z * f23.x - b.z * f23.y;
        ob.z = a.z * f23.y + b.z * f23.x;
        oa.w = a.w * f23.z - b.w * f23.w;
        ob.w = a.w * f23.w + b.w * f23.z;
        out.store4(dst_at + p as u64, oa);
        out.store4(dst_at + (pairs + p) as u64, ob);
    } else {
        if gid() >= positions * heads * pairs {
            return;
        }
        let pair = gid() % pairs;
        let h = (gid() / pairs) % heads;
        let s = gid() / (pairs * heads);
        let src_at = (s * src_stride + col_off + h * width) as u64;
        let dst_at = ((s * heads + h) * width) as u64;
        let c = freqs[(s * width + 2 * pair) as u64];
        let sn = freqs[(s * width + 2 * pair + 1) as u64];
        let a = pair;
        let b = pairs + pair;
        let x0 = x[src_at + a as u64];
        let x1 = x[src_at + b as u64];
        out[dst_at + a as u64] = x0 * c - x1 * sn;
        out[dst_at + b as u64] = x0 * sn + x1 * c;
    }
}

/// Fused strided slice and RoPE, vectorized when four-aligned.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = slice_rope_name(self),
    launches = slice_rope_launch,
    shape = slice_rope_shape(self),
    work = self.positions * self.heads * self.width,
    accepts = slice_rope_accepts,
    emits = slice_rope_emits,
    source = slice_rope_source,
    load = slice_rope_load,
    unproven,
)]
pub(crate) struct SliceRope {
    pub positions: usize,
    pub heads: usize,
    pub width: usize,
    pub src_stride: usize,
    pub col_off: usize,
    pub in_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
    pub wide: bool,
}

fn slice_rope_name(k: &SliceRope) -> &'static str {
    if k.wide {
        "slice_rope_x4"
    } else {
        "slice_rope"
    }
}

fn slice_rope_shape(k: &SliceRope) -> Shape {
    let pairs = k.width / 2;
    let per = if k.wide { pairs / 4 } else { pairs };
    Shape::Line(k.positions * k.heads * per)
}

fn slice_rope_accepts(k: &SliceRope, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.in_form.name(),
        1 => form.name() == F32_FORM.name(),
        _ => !form.narrow(),
    }
}

fn slice_rope_emits(k: &SliceRope, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn slice_rope_body(k: &SliceRope) -> tessera_codegen::ir::Body {
    slice_rope_ir(k.in_form, k.out_form, k.wide)
}

fn slice_rope_launch(k: &SliceRope, step: &Step) -> Vec<Launch> {
    let body = slice_rope_body(k);
    let scalar = |name: &str| match name {
        "positions" => k.positions.arg(),
        "heads" => k.heads.arg(),
        "width" => k.width.arg(),
        "src_stride" => k.src_stride.arg(),
        "col_off" => k.col_off.arg(),
        other => panic!("slice_rope has no field `{other}`"),
    };
    vec![Launch::of(&body, step, false, &scalar, slice_rope_shape(k))]
}

fn slice_rope_source(k: &SliceRope) -> Source {
    let body = slice_rope_body(k);
    let base = k.name();
    let entry = match (k.in_form.name(), k.out_form.name()) {
        ("f32", "f32") => format!("plan_{base}"),
        (a, b) => format!("plan_{base}_{a}_{b}"),
    };
    Source::named(body, entry)
}

fn slice_rope_load(k: &SliceRope, _: usize) -> Option<Load> {
    let body = slice_rope_body(k);
    let elems = k.positions * k.heads * k.width;
    let per = if k.wide { k.width / 8 } else { k.width / 2 };
    Some(Load {
        threads: k.positions * k.heads * per,
        group: 256,
        trips: 1.0,
        trip: body.trip(),
        reads: vec![
            k.in_form.bytes(elems) as f64,
            4.0 * (k.positions * k.width) as f64,
        ],
        written: k.out_form.bytes(elems) as f64,
        flight: k.in_form.bytes(if k.wide { 8 } else { 2 }) as f64
            + if k.wide { 32.0 } else { 8.0 },
        ..Load::default()
    })
}

/// Unshared slice (past any reshapes) feeding `x`, if row-aligned.
fn fused_slice(
    tree: &Tree,
    mut cur: T,
    positions: usize,
    heads: usize,
    width: usize,
) -> Option<(T, usize, usize)> {
    while matches!(tree.node(cur).term, Term::Op(Op::View(View::Reshape(_)))) {
        if tree.root(cur) {
            return None;
        }
        cur = tree.node(cur).args[0];
    }
    if tree.root(cur) {
        return None;
    }
    let node = tree.node(cur);
    let Term::Op(Op::View(View::Slice { axis, start, .. })) = &node.term else {
        return None;
    };
    let src = node.args[0];
    let src_node = tree.node(src);
    if !src_node.form.dtype().is_float()
        || matches!(src_node.term, Term::Leaf(Home::Weight(_) | Home::Part(..)))
    {
        return None;
    }
    let a = *axis as usize;
    let outer: usize = src_node.shape[..a].iter().product();
    let inner: usize = src_node.shape[a + 1..].iter().product();
    let start = tree.eval(start)? as usize;
    if outer != positions || tree.elements(cur) != positions * heads * width {
        return None;
    }
    Some((src, src_node.shape[a] * inner, start * inner))
}

/// Unshared RMS norm feeding `x`, if its row width matches.
fn fused_norm(tree: &Tree, x: T, width: usize) -> Option<(T, Option<T>, f32)> {
    if tree.root(x) || !width.is_multiple_of(2) || width == 0 {
        return None;
    }
    let node = tree.node(x);
    if node.shape.last() != Some(&width) {
        return None;
    }
    match node.term {
        Term::Stage(Stage::RmsNorm { eps }) => Some((node.args[0], Some(node.args[1]), eps)),
        Term::Stage(Stage::Rms { eps }) => Some((node.args[0], None, eps)),
        _ => None,
    }
}

/// Appends fused norm-rope or slice-rope drafts when patterns match.
fn push_fused(
    tree: &Tree,
    at: T,
    x: T,
    angles: Draft,
    rope: Rope,
    len: usize,
    drafts: &mut Vec<Draft>,
) {
    if let Some((src, weight, eps)) = fused_norm(tree, x, rope.width) {
        let x_form = plain(tree.node(src).form);
        let w_form = weight.map_or(F32_FORM, |w| plain(tree.node(w).form));
        let out_form = writable(tree.node(at).form);
        let mut reads = vec![Read::Node(src), Read::Draft(0)];
        if let Some(w) = weight {
            reads.push(Read::Node(w));
        }
        drafts.push(angles.clone().choice(1));
        let mut base = src;
        while matches!(tree.node(base).term, Term::Op(Op::View(View::Reshape(_)))) {
            if tree.root(base) {
                break;
            }
            base = tree.node(base).args[0];
        }
        let fast4 = !rope.adjacent
            && rope.width == 128
            && x_form.name() == out_form.name()
            && float_wide(x_form)
            && float_wide(w_form)
            && !tree.root(src)
            && !tree.root(base)
            && !matches!(tree.node(base).term, Term::Leaf(_));
        if fast4 {
            let kernel = NormRope4 {
                positions: rope.positions,
                heads: rope.heads,
                eps,
                weighted: weight.is_some(),
                x_form,
                w_form,
            };
            drafts.push(Draft::new(kernel, reads, len).choice(1));
        } else {
            let kernel = NormRope {
                positions: rope.positions,
                heads: rope.heads,
                width: rope.width,
                eps,
                adjacent: rope.adjacent,
                weighted: weight.is_some(),
                x_form,
                w_form,
                out_form,
            };
            drafts.push(Draft::new(kernel, reads, len).choice(1));
        }
    }
    if !rope.adjacent
        && let Some((src, src_stride, col_off)) =
            fused_slice(tree, x, rope.positions, rope.heads, rope.width)
    {
        let in_form = plain(tree.node(src).form);
        let out_form = writable(tree.node(at).form);
        let wide = float_wide(in_form)
            && float_wide(out_form)
            && rope.width.is_multiple_of(8)
            && src_stride.is_multiple_of(4)
            && col_off.is_multiple_of(4);
        let kernel = SliceRope {
            positions: rope.positions,
            heads: rope.heads,
            width: rope.width,
            src_stride,
            col_off,
            in_form,
            out_form,
            wide,
        };
        drafts.push(angles.choice(2));
        drafts.push(Draft::new(kernel, [Read::Node(src), Read::Draft(0)], len).choice(2));
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Stage {
        stage,
        x,
        operand: Some((op, op_len)),
        len,
        width,
    } = tree.site(at)?
    else {
        return None;
    };
    let threads = tree.knobs.get("rope", "threads").unwrap_or(0);
    match stage {
        Stage::Rope {
            theta,
            pairs,
            turned,
        } => {
            let angles = Angles {
                count: op_len,
                width,
                theta,
                turned: turned.map_or(width / 2, |t| t as usize),
                threads,
            };
            let rope = Rope {
                positions: op_len,
                heads: len / (op_len * width),
                width,
                adjacent: pairs == Pairs::Adjacent,
                threads,
            };
            let a_draft = Draft::new(angles, [op], op_len * width);
            let mut drafts = vec![
                a_draft.clone(),
                Draft::new(rope, [Read::Node(x), Read::Draft(0)], len),
            ];
            push_fused(tree, at, x, a_draft, rope, len, &mut drafts);
            Some(drafts)
        }
        Stage::Mrope {
            theta,
            sections,
            interleaved,
        } => {
            let axes = *tree.node(op).shape.last()?;
            let positions = op_len / axes.max(1);
            let angles = MropeAngles {
                count: positions,
                width,
                axes,
                theta,
                s0: sections[0] as usize,
                s1: sections[1] as usize,
                s2: sections[2] as usize,
                interleaved,
            };
            let rope = Rope {
                positions,
                heads: len / (positions * width),
                width,
                adjacent: false,
                threads,
            };
            let a_draft = Draft::new(angles, [op], positions * width);
            let mut drafts = vec![
                a_draft.clone(),
                Draft::new(rope, [Read::Node(x), Read::Draft(0)], len),
            ];
            push_fused(tree, at, x, a_draft, rope, len, &mut drafts);
            Some(drafts)
        }
        _ => None,
    }
}

tessera_lower::rule!("rope", propose_all);

mod metal;

fn propose_all(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let mut drafts = propose(tree, at)?;
    if let Some(d) = metal::draft(tree, at) {
        drafts.push(d);
    }
    Some(drafts)
}
