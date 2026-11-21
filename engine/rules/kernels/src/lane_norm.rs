//! RMS norm, a SIMD group per row, summed by shuffles.

use crate::{AHEAD, WIDE};
use tessera_codegen::prelude::*;
use tessera_graph::Order;
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::{Draft, Kernel, Load, Needs, Source, T, Tree, sites};

/// Sums a row's squares, then scales the row.
#[kernel]
fn norm_lanes<const WIDE: u32, const AHEAD: u32, const WEIGHTED: bool>(
    #[form(x_form)] x: Buf<Dyn>,
    #[form(w_form)] w: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    rows: u32,
    width: u32,
    eps: f32,
    #[build] x_form: &'static dyn Stored,
    #[build] w_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    let row = gid() / LANES as u32;
    if row >= rows {
        return;
    }
    let at = row as u64 * width as u64;
    let mut acc = 0.0f32;
    for i in (lane() * WIDE..width).by(LANES as u32 * WIDE).unroll(AHEAD) {
        let v = x.load4(at + i as u64);
        acc += v.x * v.x;
        acc += v.y * v.y;
        acc += v.z * v.z;
        acc += v.w * v.w;
    }
    acc = sum32(acc);
    let inv = rsqrt(acc / width as f32 + eps);
    for i in (lane() * WIDE..width).by(LANES as u32 * WIDE).unroll(AHEAD) {
        let v = x.load4(at + i as u64);
        let mut g = F32x4::zero();
        if WEIGHTED {
            g = w.load4(i as u64);
        } else {
            g.x = 1.0;
            g.y = 1.0;
            g.z = 1.0;
            g.w = 1.0;
        }
        let mut o = F32x4::zero();
        o.x = v.x * inv * g.x;
        o.y = v.y * inv * g.y;
        o.z = v.z * inv * g.z;
        o.w = v.w * inv * g.w;
        out.store4(at + i as u64, o);
    }
}

/// An RMS norm over `rows` of `width`, one launch.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    norm_lanes::<WIDE, AHEAD>(self.x_form, self.w_form, self.out_form),
    variant = self.weighted,
    name = match self.weighted {
        true => "lane_norm",
        false => "lane_rms",
    },
    threads = self.rows * LANES,
    work = self.rows * self.width,
    needs = Needs::lanes(),
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
    pub x_form: &'static dyn Stored,
    pub w_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
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

fn norm_source(k: &LaneNorm) -> Source {
    let body = match k.weighted {
        true => norm_lanes_ir::<WIDE, AHEAD, true>(k.x_form, k.w_form, k.out_form),
        false => norm_lanes_ir::<WIDE, AHEAD, false>(k.x_form, k.w_form, k.out_form),
    };
    let base = k.name();
    let entry = match (k.x_form.name(), k.w_form.name(), k.out_form.name()) {
        ("f32", "f32", "f32") => format!("plan_{base}"),
        (x, w, o) => format!("plan_{base}_{x}_{w}_{o}"),
    };
    Source::named(body, entry)
}

fn norm_load(k: &LaneNorm, _: usize) -> Option<Load> {
    let body = match k.weighted {
        true => norm_lanes_ir::<WIDE, AHEAD, true>(k.x_form, k.w_form, k.out_form),
        false => norm_lanes_ir::<WIDE, AHEAD, false>(k.x_form, k.w_form, k.out_form),
    };
    let elems = k.rows * k.width;
    let x_bytes = k.x_form.bytes(elems) as f64;
    let mut reads = vec![x_bytes];
    if k.weighted {
        reads.push(k.w_form.bytes(k.width) as f64);
    }
    Some(Load {
        threads: k.rows * LANES,
        group: 256,
        trips: (k.width as f64 / (LANES * WIDE as usize) as f64).ceil(),
        trip: body.trip(),
        reads,
        written: k.out_form.bytes(elems) as f64,
        flight: k.x_form.bytes((WIDE * AHEAD) as usize) as f64,
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
    let kernel = LaneNorm {
        rows: site.len / site.width,
        width: site.width,
        eps,
        weighted,
        x_form,
        w_form,
        out_form,
    };
    Some(vec![Draft::new(kernel, site.padded(), site.len)])
}

tessera_lower::rule!("lane_norm", propose);
