//! Softmax over rows, one warp or block per row.

use crate::{AHEAD, WIDE};
use tessera_codegen::prelude::*;
use tessera_graph::{Order, Stage};
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::{
    Backend, Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T, ToArg, Tree, sites,
};

/// SIMD groups sharing one wide row in block mode.
const WARPS: u32 = 8;

/// Threads per row in block mode.
const BLOCK_THREADS: u32 = WARPS * LANES as u32;

/// Minimum width to split across a whole block.
const BLOCK_MIN: usize = 512;

/// Minimum width that prefers block mode for prefill batches.
const WIDE_BLOCK_MIN: usize = 1024;

/// Maximum rows where narrow block-wide softmax wins.
const BLOCK_ROWS: usize = 16;

/// Maximum rows where wide block-wide softmax wins.
const WIDE_BLOCK_ROWS: usize = 2048;

/// Width above which block mode always runs.
const HUGE_WIDTH: usize = 4096;

/// Maximum wide trips held in registers across passes.
const CACHE_TRIPS: u32 = 8;

/// Per-thread register slots for cached pass-1 values.
const CACHE_SLOTS: usize = (CACHE_TRIPS * WIDE) as usize;

/// Maximum across 32 lanes via butterfly shuffles.
#[kernel]
fn warp_max(mut v: f32) -> f32 {
    v = max(v, shfl_xor(v, 16));
    v = max(v, shfl_xor(v, 8));
    v = max(v, shfl_xor(v, 4));
    v = max(v, shfl_xor(v, 2));
    v = max(v, shfl_xor(v, 1));
    v
}

/// Reduces per-warp values across a block via sum or max.
#[kernel]
fn block_reduce<const SUM: bool>(v: f32) -> f32 {
    let mut red = shared::<f32>(WARPS);
    if lane() == 0 {
        red[lid() / LANES as u32] = v;
    }
    barrier();
    let mut acc = if SUM { 0.0f32 } else { red[0] };
    for w_i in 0..WARPS {
        if SUM {
            acc += red[w_i];
        } else {
            acc = max(acc, red[w_i]);
        }
    }
    barrier();
    acc
}

/// Trip count cached in registers, or zero when uncached.
fn cache_trips(width: usize, block: bool, wide: bool) -> u32 {
    if !wide {
        return 0;
    }
    let stride = if block { BLOCK_THREADS as usize } else { LANES } * WIDE as usize;
    let trips = width.div_ceil(stride);
    if trips <= CACHE_TRIPS as usize {
        trips as u32
    } else {
        0
    }
}

/// Whether `rows` of `width` should run one block per row.
fn use_block(rows: usize, width: usize) -> bool {
    width >= HUGE_WIDTH
        || (rows <= BLOCK_ROWS && width >= BLOCK_MIN)
        || (rows <= WIDE_BLOCK_ROWS && width >= WIDE_BLOCK_MIN)
}

/// Parallel row softmax, register-cached or two-pass online.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn softmax_lanes<const WIDE: u32, const AHEAD: u32>(
    #[form(x_form)] x: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    rows: u32,
    width: u32,
    #[build] x_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
    #[build] block: bool,
    #[build] wide: bool,
    #[build] use_cache: bool,
    #[build] cached: u32,
) {
    let group_w = if block { BLOCK_THREADS } else { LANES as u32 };
    let row = gid() / group_w;
    if row >= rows {
        return;
    }
    let at = row as u64 * width as u64;
    let first = if block { lid() } else { lane() };
    let mut hi = f32::NEG_INFINITY;
    if use_cache {
        let mut vals = [0.0f32; CACHE_SLOTS];
        for t in (0..cached).unroll(CACHE_TRIPS) {
            let i = first * WIDE + t * (group_w * WIDE);
            if i < width {
                let v = x.load4(at + i as u64);
                let idx = (t * WIDE) as usize;
                vals[idx] = v.x;
                vals[idx + 1] = v.y;
                vals[idx + 2] = v.z;
                vals[idx + 3] = v.w;
                hi = max(hi, max(max(v.x, v.y), max(v.z, v.w)));
            }
        }
        let w_hi = warp_max(hi);
        let top = if block {
            block_reduce::<false>(w_hi)
        } else {
            w_hi
        };
        let mut sum = 0.0f32;
        for t in (0..cached).unroll(CACHE_TRIPS) {
            let i = first * WIDE + t * (group_w * WIDE);
            if i < width {
                let idx = (t * WIDE) as usize;
                let e0 = exp(vals[idx] - top);
                let e1 = exp(vals[idx + 1] - top);
                let e2 = exp(vals[idx + 2] - top);
                let e3 = exp(vals[idx + 3] - top);
                vals[idx] = e0;
                vals[idx + 1] = e1;
                vals[idx + 2] = e2;
                vals[idx + 3] = e3;
                sum += (e0 + e1) + (e2 + e3);
            }
        }
        sum = sum32(sum);
        let tot = if block {
            block_reduce::<true>(sum)
        } else {
            sum
        };
        let inv = 1.0f32 / tot;
        for t in (0..cached).unroll(CACHE_TRIPS) {
            let i = first * WIDE + t * (group_w * WIDE);
            if i < width {
                let idx = (t * WIDE) as usize;
                let o = F32x4::new(
                    vals[idx] * inv,
                    vals[idx + 1] * inv,
                    vals[idx + 2] * inv,
                    vals[idx + 3] * inv,
                );
                out.store4(at + i as u64, o);
            }
        }
    } else {
        let mut sum = 0.0f32;
        if wide {
            for i in (first * WIDE..width).by(group_w * WIDE).unroll(AHEAD) {
                let v = x.load4(at + i as u64);
                let qm = max(max(v.x, v.y), max(v.z, v.w));
                if qm > hi {
                    if sum != 0.0f32 {
                        sum *= exp(hi - qm);
                    }
                    hi = qm;
                }
                if qm > f32::NEG_INFINITY {
                    sum += (exp(v.x - hi) + exp(v.y - hi)) + (exp(v.z - hi) + exp(v.w - hi));
                }
            }
        } else {
            for i in (first..width).by(group_w) {
                let v = x[at + i as u64];
                if v > hi {
                    if sum != 0.0f32 {
                        sum *= exp(hi - v);
                    }
                    hi = v;
                }
                if v > f32::NEG_INFINITY {
                    sum += exp(v - hi);
                }
            }
        }
        let w_hi = warp_max(hi);
        if sum != 0.0f32 {
            sum *= exp(hi - w_hi);
        }
        sum = sum32(sum);
        let mut top = w_hi;
        let mut tot = sum;
        if block {
            let mut red_hi = shared::<f32>(WARPS);
            let mut red_sum = shared::<f32>(WARPS);
            if lane() == 0 {
                red_hi[lid() / LANES as u32] = w_hi;
                red_sum[lid() / LANES as u32] = sum;
            }
            barrier();
            top = red_hi[0];
            for w_i in 1..WARPS {
                top = max(top, red_hi[w_i]);
            }
            tot = 0.0f32;
            for w_i in 0..WARPS {
                let ws = red_sum[w_i];
                if ws != 0.0f32 {
                    tot += ws * exp(red_hi[w_i] - top);
                }
            }
        }
        let inv = 1.0f32 / tot;
        if wide {
            for i in (first * WIDE..width).by(group_w * WIDE).unroll(AHEAD) {
                let v = x.load4(at + i as u64);
                let o = F32x4::new(
                    exp(v.x - top) * inv,
                    exp(v.y - top) * inv,
                    exp(v.z - top) * inv,
                    exp(v.w - top) * inv,
                );
                out.store4(at + i as u64, o);
            }
        } else {
            for i in (first..width).by(group_w) {
                out[at + i as u64] = exp(x[at + i as u64] - top) * inv;
            }
        }
    }
}

/// Warp- or block-parallel softmax over `rows` of `width`.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.block { "block_softmax" } else { "lane_softmax" },
    launches = softmax_launches,
    shape = softmax_shape(self.rows, self.block),
    work = self.rows * self.width,
    needs = softmax_needs(self.block),
    order = Order::PerConfig,
    accepts = softmax_accepts,
    emits = softmax_emits,
    source = softmax_source,
    load = softmax_load,
    unproven,
)]
pub(crate) struct LaneSoftmax {
    pub rows: usize,
    pub width: usize,
    pub block: bool,
    pub wide: bool,
    pub x_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

fn softmax_shape(rows: usize, block: bool) -> Shape {
    if block {
        Shape::Grid {
            across: rows,
            down: 1,
            threads: BLOCK_THREADS as usize,
        }
    } else {
        Shape::Line(rows * LANES)
    }
}

fn softmax_needs(block: bool) -> Needs {
    if block {
        Needs {
            lanes: LANES,
            group_bytes: 8 * WARPS as usize,
            group_threads: BLOCK_THREADS as usize,
            lacks: &[Backend::Cuda],
            ..Needs::default()
        }
    } else {
        Needs {
            lacks: &[Backend::Cuda],
            ..Needs::lanes()
        }
    }
}

fn softmax_accepts(k: &LaneSoftmax, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.x_form.name(),
        _ => !form.narrow(),
    }
}

fn softmax_emits(k: &LaneSoftmax, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn softmax_body(k: &LaneSoftmax) -> tessera_codegen::ir::Body {
    let cached = cache_trips(k.width, k.block, k.wide);
    softmax_lanes_ir::<WIDE, AHEAD>(k.x_form, k.out_form, k.block, k.wide, cached > 0, cached)
}

fn softmax_launches(k: &LaneSoftmax, step: &Step) -> Vec<Launch> {
    let body = softmax_body(k);
    let scalar = |name: &str| match name {
        "rows" => k.rows.arg(),
        "width" => k.width.arg(),
        other => panic!("LaneSoftmax has no field `{other}`"),
    };
    vec![Launch::of(
        &body,
        step,
        false,
        &scalar,
        softmax_shape(k.rows, k.block),
    )]
}

fn softmax_source(k: &LaneSoftmax) -> Source {
    let body = softmax_body(k);
    let base = k.name();
    let mode = match (k.wide, cache_trips(k.width, k.block, k.wide)) {
        (true, n) if n > 0 => format!("_c{n}"),
        (true, _) => "_w4".into(),
        (false, _) => String::new(),
    };
    let entry = match (k.x_form.name(), k.out_form.name()) {
        ("f32", "f32") => format!("plan_{base}{mode}"),
        (x, o) => format!("plan_{base}{mode}_{x}_{o}"),
    };
    Source::named(body, entry)
}

fn softmax_load(k: &LaneSoftmax, _: usize) -> Option<Load> {
    let body = softmax_body(k);
    let elems = k.rows * k.width;
    let x_bytes = k.x_form.bytes(elems) as f64;
    let cached = cache_trips(k.width, k.block, k.wide) > 0;
    let reads = if cached {
        vec![x_bytes]
    } else {
        vec![x_bytes, x_bytes]
    };
    let per_row = if k.block {
        BLOCK_THREADS as usize
    } else {
        LANES
    };
    let step = if k.wide { WIDE as usize } else { 1 };
    let trips = if cached {
        1.0
    } else {
        (k.width as f64 / (per_row * step) as f64).ceil()
    };
    let flight_elems = if k.wide { (WIDE * AHEAD) as usize } else { 1 };
    Some(Load {
        threads: k.rows * per_row,
        group: BLOCK_THREADS as usize,
        trips,
        trip: body.trip(),
        reads,
        written: k.out_form.bytes(elems) as f64,
        flight: k.x_form.bytes(flight_elems) as f64,
        ..Load::default()
    })
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    if site.stage != Stage::Softmax {
        return None;
    }
    let x_form = plain(tree.node(site.x).form);
    let out_form = writable(tree.node(at).form);
    let rows = site.len / site.width;
    let width = site.width;
    let kernel = LaneSoftmax {
        rows,
        width,
        block: use_block(rows, width),
        wide: width.is_multiple_of(WIDE as usize),
        x_form,
        out_form,
    };
    Some(vec![Draft::new(kernel, [site.x], site.len)])
}

tessera_lower::rule!("lane_softmax", propose);
