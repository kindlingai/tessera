//! RMS norm, one SIMD group or one block per row.

use crate::{AHEAD, WIDE};
use tessera_codegen::prelude::*;
use tessera_graph::Order;
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::{
    Draft, Kernel, Knobs, Launch, Load, Needs, Shape, Source, Step, T, ToArg, Tree, sites,
};

/// SIMD groups sharing one wide row in block mode.
pub(crate) const WARPS: u32 = 8;

/// Maximum SIMD groups sharing one wide row.
pub(crate) const MAX_WARPS: u32 = 16;

/// Minimum width to split across a whole block.
pub(crate) const BLOCK_MIN: usize = 512;

/// Minimum width that prefers block mode for prefill batches.
const WIDE_BLOCK_MIN: usize = 1024;

/// Minimum width that widens block mode to 16 warps.
const WIDE_WARPS_MIN: usize = 4096;

/// Maximum rows where narrow block-wide RMS norm wins.
pub(crate) const BLOCK_ROWS: usize = 16;

/// Maximum rows where wide block-wide RMS norm wins.
const WIDE_BLOCK_ROWS: usize = 2048;

/// Maximum wide trips held in registers across passes.
pub(crate) const CACHE_TRIPS: u32 = 8;

/// Declares row-norm tuning knobs for one kernel rule.
macro_rules! norm_knobs {
    ($k:literal) => {
        $crate::__knob!($k, "warps", 1, 32);
        $crate::__knob!($k, "wide_warps", 1, 32);
        $crate::__knob!($k, "block_min", 32, 65536);
        $crate::__knob!($k, "wide_block_min", 32, 65536);
        $crate::__knob!($k, "wide_warps_min", 32, 131072);
        $crate::__knob!($k, "block_rows", 1, 65536);
        $crate::__knob!($k, "wide_block_rows", 1, 65536);
        $crate::__knob!($k, "cache_trips", 0, 32);
        $crate::__knob!($k, "ahead", 1, 16);
    };
}

pub(crate) use norm_knobs;
norm_knobs!("lane_norm");

/// Tunable shape parameters for row RMS norm kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowCfg {
    pub warps: u32,
    pub wide_warps: u32,
    pub block_min: usize,
    pub wide_block_min: usize,
    pub wide_warps_min: usize,
    pub block_rows: usize,
    pub wide_block_rows: usize,
    pub cache_trips: u32,
    pub ahead: u32,
}

impl Default for RowCfg {
    fn default() -> Self {
        Self {
            warps: WARPS,
            wide_warps: MAX_WARPS,
            block_min: BLOCK_MIN,
            wide_block_min: WIDE_BLOCK_MIN,
            wide_warps_min: WIDE_WARPS_MIN,
            block_rows: BLOCK_ROWS,
            wide_block_rows: WIDE_BLOCK_ROWS,
            cache_trips: CACHE_TRIPS,
            ahead: AHEAD,
        }
    }
}

impl RowCfg {
    /// Reads shape knobs for `kernel`, keeping defaults when unset.
    pub(crate) fn of(knobs: &Knobs, kernel: &str) -> Self {
        let mut cfg = Self::default();
        if let Some(v) = knobs.get(kernel, "warps") {
            cfg.warps = v as u32;
        }
        if let Some(v) = knobs.get(kernel, "wide_warps") {
            cfg.wide_warps = v as u32;
        }
        if let Some(v) = knobs.get(kernel, "block_min") {
            cfg.block_min = v;
        }
        if let Some(v) = knobs.get(kernel, "wide_block_min") {
            cfg.wide_block_min = v;
        }
        if let Some(v) = knobs.get(kernel, "wide_warps_min") {
            cfg.wide_warps_min = v;
        }
        if let Some(v) = knobs.get(kernel, "block_rows") {
            cfg.block_rows = v;
        }
        if let Some(v) = knobs.get(kernel, "wide_block_rows") {
            cfg.wide_block_rows = v;
        }
        if let Some(v) = knobs.get(kernel, "cache_trips") {
            cfg.cache_trips = v as u32;
        }
        if let Some(v) = knobs.get(kernel, "ahead") {
            cfg.ahead = v as u32;
        }
        cfg
    }

    pub(crate) fn warps(self, width: usize) -> u32 {
        if width >= self.wide_warps_min {
            self.wide_warps
        } else {
            self.warps
        }
    }

    pub(crate) fn use_block(self, rows: usize, width: usize) -> bool {
        (rows <= self.block_rows && width >= self.block_min)
            || (rows <= self.wide_block_rows && width >= self.wide_block_min)
    }
}

/// Thread count in block mode for `warps`.
pub(crate) fn block_threads(warps: u32) -> u32 {
    warps * LANES as u32
}

/// Trip count cached in registers, or zero when too wide.
pub(crate) fn cache_trips(width: usize, block: bool, warps: u32, max_trips: u32) -> u32 {
    if max_trips == 0 {
        return 0;
    }
    let stride = if block {
        block_threads(warps) as usize
    } else {
        LANES
    } * WIDE as usize;
    let trips = width.div_ceil(stride);
    if trips <= max_trips as usize {
        trips as u32
    } else {
        0
    }
}

/// Whether `width` divides the row tile cleanly.
pub(crate) fn exact_trips(width: usize, block: bool, warps: u32) -> bool {
    let stride = if block {
        block_threads(warps) as usize
    } else {
        LANES
    } * WIDE as usize;
    width.is_multiple_of(stride)
}

/// Source entry tag capturing block warps, cached trips, and alignment.
pub(crate) fn row_tag(width: usize, block: bool, warps: u32, max_trips: u32, ahead: u32) -> String {
    let mut tag = String::new();
    if block && warps != WARPS {
        tag.push_str(&format!("_w{warps}"));
    }
    let cached = cache_trips(width, block, warps, max_trips);
    if cached > 0 {
        tag.push_str(&format!("_c{cached}"));
        if exact_trips(width, block, warps) {
            tag.push('x');
        }
    } else if max_trips != CACHE_TRIPS {
        tag.push_str("_nc");
    }
    if ahead != AHEAD {
        tag.push_str(&format!("_a{ahead}"));
    }
    tag
}

/// Reduces per-thread partial sums across all warps in a block.
#[kernel]
pub(crate) fn block_sum(acc: f32, #[build] warps: u32, #[build] cap: u32) -> f32 {
    let mut red = shared::<f32>(cap);
    if lane() == 0 {
        red[lid() / LANES as u32] = acc;
    }
    barrier();
    let mut tot = 0.0f32;
    for w_i in 0..warps {
        tot += red[w_i];
    }
    tot
}

/// Loads up to three four-wide vectors and sums them.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn load_sum4(
    #[form(a_form)] a: Buf<Dyn>,
    #[form(b_form)] b: Buf<Dyn>,
    #[form(c_form)] c: Buf<Dyn>,
    off: u64,
    #[build] a_form: &'static dyn Stored,
    #[build] b_form: &'static dyn Stored,
    #[build] c_form: &'static dyn Stored,
    #[build] add2: bool,
    #[build] add3: bool,
) -> F32x4 {
    // A packed load decodes per lane and cannot take `+=`.
    let va = a.load4(off);
    let mut s = F32x4::new(va.x, va.y, va.z, va.w);
    if add2 {
        let vb = b.load4(off);
        s.x += vb.x;
        s.y += vb.y;
        s.z += vb.z;
        s.w += vb.w;
    }
    if add3 {
        let vc = c.load4(off);
        s.x += vc.x;
        s.y += vc.y;
        s.z += vc.z;
        s.w += vc.w;
    }
    s
}

/// Scales four values by `inv` and optional weights, then stores.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn scale_store4(
    #[form(w_form)] w: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    off: u64,
    col: u64,
    x: f32,
    y: f32,
    z: f32,
    v: f32,
    inv: f32,
    #[build] w_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
    #[build] weighted: bool,
) {
    let mut o = F32x4::zero();
    if weighted {
        let g = w.load4(col);
        o.x = x * inv * g.x;
        o.y = y * inv * g.y;
        o.z = z * inv * g.z;
        o.w = v * inv * g.w;
    } else {
        o.x = x * inv;
        o.y = y * inv;
        o.z = z * inv;
        o.w = v * inv;
    }
    out.store4(off, o);
}

/// Sums rows, optionally saves the sum, and norms.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn rms_norm_lanes<const WIDE: u32>(
    #[form(a_form)] a: Buf<Dyn>,
    #[form(b_form)] b: Buf<Dyn>,
    #[form(c_form)] c: Buf<Dyn>,
    #[form(w_form)] w: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    #[form(sum_form)] mut sum_out: BufMut<Dyn>,
    rows: u32,
    width: u32,
    eps: f32,
    #[build] a_form: &'static dyn Stored,
    #[build] b_form: &'static dyn Stored,
    #[build] c_form: &'static dyn Stored,
    #[build] w_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
    #[build] sum_form: &'static dyn Stored,
    #[build] add2: bool,
    #[build] add3: bool,
    #[build] weighted: bool,
    #[build] save_sum: bool,
    #[build] reload_sum: bool,
    #[build] block: bool,
    #[build] warps: u32,
    #[build] shared_cap: u32,
    #[build] use_cache: bool,
    #[build] cached: u32,
    #[build] cache_cap: u32,
    #[build] cache_slots: usize,
    #[build] ahead: u32,
    #[build] exact: bool,
) {
    let group_w = if block {
        warps * LANES as u32
    } else {
        LANES as u32
    };
    let row = gid() / group_w;
    if row >= rows {
        return;
    }
    let at = row as u64 * width as u64;
    let first = if block { lid() } else { lane() };
    let mut vals = local::<f32>(cache_slots);
    let mut acc = 0.0f32;
    if use_cache {
        for t in (0..cached).unroll(cache_cap) {
            let i = first * WIDE + t * (group_w * WIDE);
            if exact || i < width {
                let off = at + i as u64;
                let s = load_sum4(a, b, c, off, a_form, b_form, c_form, add2, add3);
                if save_sum {
                    sum_out.store4(off, s);
                }
                let idx = (t * WIDE) as usize;
                vals[idx] = s.x;
                vals[idx + 1] = s.y;
                vals[idx + 2] = s.z;
                vals[idx + 3] = s.w;
                acc += s.x * s.x + s.y * s.y + s.z * s.z + s.w * s.w;
            }
        }
    } else {
        for i in (first * WIDE..width).by(group_w * WIDE).unroll(ahead) {
            let off = at + i as u64;
            let s = load_sum4(a, b, c, off, a_form, b_form, c_form, add2, add3);
            if save_sum {
                sum_out.store4(off, s);
            }
            acc += s.x * s.x + s.y * s.y + s.z * s.z + s.w * s.w;
        }
    }
    acc = sum32(acc);
    let tot = if block {
        block_sum(acc, warps, shared_cap)
    } else {
        acc
    };
    let inv = rsqrt(tot / width as f32 + eps);
    if use_cache {
        for t in (0..cached).unroll(cache_cap) {
            let i = first * WIDE + t * (group_w * WIDE);
            if exact || i < width {
                let idx = (t * WIDE) as usize;
                scale_store4(
                    w,
                    out,
                    at + i as u64,
                    i as u64,
                    vals[idx],
                    vals[idx + 1],
                    vals[idx + 2],
                    vals[idx + 3],
                    inv,
                    w_form,
                    out_form,
                    weighted,
                );
            }
        }
    } else {
        for i in (first * WIDE..width).by(group_w * WIDE).unroll(ahead) {
            let off = at + i as u64;
            let mut s = F32x4::zero();
            if reload_sum {
                s = sum_out.load4(off);
            } else {
                s = load_sum4(a, b, c, off, a_form, b_form, c_form, add2, add3);
            }
            scale_store4(
                w, out, off, i as u64, s.x, s.y, s.z, s.w, inv, w_form, out_form, weighted,
            );
        }
    }
}

/// An RMS norm over `rows` of `width`, one launch.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = match (self.weighted, self.block) {
        (true, true) => "block_norm",
        (false, true) => "block_rms",
        (true, false) => "lane_norm",
        (false, false) => "lane_rms",
    },
    launches = norm_launches,
    shape = row_shape(self.rows, self.block, self.warps),
    work = self.rows * self.width,
    needs = row_needs(self.block, self.warps),
    // Rows sum alike at any count. Other kernels differ.
    order = Order::PerConfig,
    accepts = norm_accepts,
    emits = norm_emits,
    source = norm_source,
    load = norm_load,
    unproven,
)]
pub(crate) struct LaneNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
    pub weighted: bool,
    pub block: bool,
    pub warps: u32,
    pub cache_trips: u32,
    pub ahead: u32,
    pub x_form: &'static dyn Stored,
    pub w_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

/// Launch grid for `rows` in block or lane mode.
pub(crate) fn row_shape(rows: usize, block: bool, warps: u32) -> Shape {
    if block {
        Shape::Grid {
            across: rows,
            down: 1,
            threads: block_threads(warps) as usize,
        }
    } else {
        Shape::Line(rows * LANES)
    }
}

/// Device requirements for block or lane row kernels.
pub(crate) fn row_needs(block: bool, warps: u32) -> Needs {
    if block {
        Needs {
            lanes: LANES,
            group_bytes: 4 * warps.max(MAX_WARPS) as usize,
            group_threads: block_threads(warps) as usize,
            ..Needs::default()
        }
    } else {
        Needs::lanes()
    }
}

fn norm_accepts(k: &LaneNorm, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.x_form.name(),
        1 => form.name() == k.w_form.name(),
        _ => !form.narrow(),
    }
}

fn norm_emits(k: &LaneNorm, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

/// Parameters for lowering a row-wise RMS norm body.
pub(crate) struct RowSpec {
    pub width: usize,
    pub block: bool,
    pub warps: u32,
    pub cache_trips: u32,
    pub ahead: u32,
    pub a_form: &'static dyn Stored,
    pub b_form: &'static dyn Stored,
    pub c_form: &'static dyn Stored,
    pub w_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
    pub sum_form: &'static dyn Stored,
    pub add2: bool,
    pub add3: bool,
    pub weighted: bool,
    pub keep_w: bool,
    pub save_sum: bool,
    pub reload_sum: bool,
}

/// Builds the IR body for a row-wise RMS norm kernel.
pub(crate) fn row_body(s: RowSpec) -> tessera_codegen::ir::Body {
    let cached = cache_trips(s.width, s.block, s.warps, s.cache_trips);
    let exact = exact_trips(s.width, s.block, s.warps);
    let shared_cap = s.warps.max(MAX_WARPS);
    let cache_cap = s.cache_trips.max(1);
    let cache_slots = (cache_cap * WIDE) as usize;
    let mut body = rms_norm_lanes_ir::<WIDE>(
        s.a_form,
        s.b_form,
        s.c_form,
        s.w_form,
        s.out_form,
        s.sum_form,
        s.add2,
        s.add3,
        s.weighted,
        s.save_sum,
        s.reload_sum,
        s.block,
        s.warps,
        shared_cap,
        cached > 0,
        cached,
        cache_cap,
        cache_slots,
        s.ahead,
        exact,
    );
    if !s.add2 {
        body.params.retain(|p| p.name() != "b");
    }
    if !s.add3 {
        body.params.retain(|p| p.name() != "c");
    }
    if !s.keep_w {
        body.params.retain(|p| p.name() != "w");
    }
    if !s.save_sum {
        body.params.retain(|p| p.name() != "sum_out");
    }
    body
}

fn norm_body(k: &LaneNorm) -> tessera_codegen::ir::Body {
    row_body(RowSpec {
        width: k.width,
        block: k.block,
        warps: k.warps,
        cache_trips: k.cache_trips,
        ahead: k.ahead,
        a_form: k.x_form,
        b_form: k.x_form,
        c_form: k.x_form,
        w_form: k.w_form,
        out_form: k.out_form,
        sum_form: k.out_form,
        add2: false,
        add3: false,
        weighted: k.weighted,
        keep_w: true,
        save_sum: false,
        reload_sum: false,
    })
}

/// Builds the single launch for a row-wise RMS norm kernel.
pub(crate) fn row_launches(
    body: &tessera_codegen::ir::Body,
    step: &Step,
    rows: usize,
    width: usize,
    eps: f32,
    block: bool,
    warps: u32,
) -> Vec<Launch> {
    let scalar = |name: &str| match name {
        "rows" => rows.arg(),
        "width" => width.arg(),
        "eps" => eps.arg(),
        other => panic!("row norm has no field `{other}`"),
    };
    vec![Launch::of(
        body,
        step,
        false,
        &scalar,
        row_shape(rows, block, warps),
    )]
}

fn norm_launches(k: &LaneNorm, step: &Step) -> Vec<Launch> {
    let body = norm_body(k);
    row_launches(&body, step, k.rows, k.width, k.eps, k.block, k.warps)
}

fn norm_source(k: &LaneNorm) -> Source {
    let body = norm_body(k);
    let base = k.name();
    let tag = row_tag(k.width, k.block, k.warps, k.cache_trips, k.ahead);
    let entry = match (k.x_form.name(), k.w_form.name(), k.out_form.name()) {
        ("f32", "f32", "f32") => format!("plan_{base}{tag}"),
        (x, w, o) => format!("plan_{base}{tag}_{x}_{w}_{o}"),
    };
    Source::named(body, entry)
}

fn norm_load(k: &LaneNorm, _: usize) -> Option<Load> {
    let body = norm_body(k);
    let elems = k.rows * k.width;
    let x_bytes = k.x_form.bytes(elems) as f64;
    let cached = cache_trips(k.width, k.block, k.warps, k.cache_trips) > 0;
    let mut reads = if cached {
        vec![x_bytes]
    } else {
        vec![x_bytes, x_bytes]
    };
    if k.weighted {
        reads.push(k.w_form.bytes(k.width) as f64);
    }
    let threads = block_threads(k.warps) as usize;
    let per_row = if k.block { threads } else { LANES };
    let trips = if cached {
        1.0
    } else {
        (k.width as f64 / (per_row * WIDE as usize) as f64).ceil()
    };
    Some(Load {
        threads: k.rows * per_row,
        group: threads,
        trips,
        trip: body.trip(),
        reads,
        written: k.out_form.bytes(elems) as f64,
        flight: k.x_form.bytes((WIDE * k.ahead) as usize) as f64,
        ..Load::default()
    })
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    let (eps, weighted) = site.rms()?;
    if !site.width.is_multiple_of(WIDE as usize) {
        return None;
    }
    let x_form = plain(tree.node(site.x).form);
    let w_form = site
        .operand
        .map_or(x_form, |(o, _)| plain(tree.node(o).form));
    let out_form = writable(tree.node(at).form);
    let rows = site.len / site.width;
    let cfg = RowCfg::of(&tree.knobs, "lane_norm");
    let block = cfg.use_block(rows, site.width);
    let kernel = LaneNorm {
        rows,
        width: site.width,
        eps,
        weighted,
        block,
        warps: cfg.warps(site.width),
        cache_trips: cfg.cache_trips,
        ahead: cfg.ahead,
        x_form,
        w_form,
        out_form,
    };
    Some(vec![Draft::new(kernel, site.padded(), site.len)])
}

tessera_lower::rule!("lane_norm", propose);
