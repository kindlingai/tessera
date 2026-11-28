//! Native Metal simdgroup_matrix Lightning Indexer scoring kernel.

use super::Sizes;
use tessera_codegen::kept::Layout;
use tessera_graph::Order;
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T};

const THREADS: usize = 128;
const SOURCE: &str = concat!(include_str!("../kv.metal"), include_str!("metal.metal"));
/// Low default guess so Metal prefers native MMA indexer scoring.
const GUESS: f64 = 1.0e-12;

pub(crate) fn plain_fmt(form: &dyn Stored) -> Option<usize> {
    match form.name() {
        "f32" => Some(0),
        "bf16" => Some(1),
        "f16" => Some(2),
        _ => None,
    }
}

pub(crate) fn k_fmt_idx(form: &dyn Stored) -> Option<usize> {
    match form.name() {
        "f32" => Some(0),
        "bf16" => Some(1),
        "f16" => Some(2),
        "fp8" => Some(3),
        _ => None,
    }
}

pub(crate) fn kept_params(
    layout: &'static dyn Layout,
    context: usize,
) -> Option<(usize, usize, usize, usize)> {
    if let Some(page) = layout.page() {
        if !page.is_power_of_two() {
            return None;
        }
        let pages = context.div_ceil(page);
        Some((1, pages, page.trailing_zeros() as usize, page - 1))
    } else if layout.name() == "dense" {
        Some((0, context, 0, 0))
    } else if layout.name() == "ring" {
        Some((2, context, 0, 0))
    } else {
        None
    }
}

fn pick_entry(dim: usize, k_form: &dyn Stored) -> Option<(&'static str, usize)> {
    let fi = k_fmt_idx(k_form)?;
    if dim <= 128 {
        const NAMES: [&str; 4] = ["idx_128_f32", "idx_128_bf16", "idx_128_f16", "idx_128_fp8"];
        return Some((NAMES[fi], 28_672));
    }
    if dim <= 256 {
        const NAMES: [&str; 4] = ["idx_256_f32", "idx_256_bf16", "idx_256_f16", "idx_256_fp8"];
        let smem = if fi == 0 { 29_696 } else { 27_648 };
        return Some((NAMES[fi], smem));
    }
    None
}

fn tile_keys(s: &Sizes) -> usize {
    if s.rows == 1 && s.out_keys() <= 2048 {
        64
    } else if s.rows <= 4 {
        128
    } else {
        256
    }
}

/// Returns the Metal MMA indexer draft when `sizes` fits.
pub(super) fn drafts(sizes: Sizes, reads: &[T], out: usize) -> Vec<Draft> {
    if fits(&sizes) {
        vec![Draft::new(IndexerMetal(sizes), reads.iter().copied(), out).choice(2)]
    } else {
        Vec::new()
    }
}

/// Whether `s` fits the native Metal MMA Lightning Indexer kernel.
fn fits(s: &Sizes) -> bool {
    s.rows > 0
        && s.out_keys() > 0
        && s.heads >= 16
        && s.dim >= 32
        && s.dim.is_multiple_of(8)
        && pick_entry(s.dim, s.k_form).is_some()
        && plain_fmt(s.q_form).is_some()
        && plain_fmt(s.w_form).is_some()
        && plain_fmt(s.out_form).is_some()
        && (!s.ragged || kept_params(s.kept, s.context).is_some())
}

/// Native Metal simdgroup_matrix Lightning Indexer scoring kernel.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "indexer_metal",
    order = Order::PerConfig,
    sizes = [self.0.rows, self.0.out_keys(), self.0.heads, self.0.dim],
    work = self.0.out_len() * self.0.heads * self.0.dim,
    guess = GUESS,
    needs = metal_needs(&self.0),
    accepts = metal_accepts,
    emits = metal_emits,
    source = metal_source(&self.0),
    launches = metal_launches,
    load = metal_load,
)]
struct IndexerMetal(Sizes);

fn metal_needs(s: &Sizes) -> Needs {
    let (_, smem) = pick_entry(s.dim, s.k_form).expect("verified entry");
    Needs::metal(THREADS, smem)
}

fn metal_accepts(k: &IndexerMetal, input: usize, form: &dyn Stored) -> bool {
    k.0.accepts(input, form)
}

fn metal_emits(k: &IndexerMetal, form: &dyn Stored) -> bool {
    k.0.emits(form)
}

fn metal_source(s: &Sizes) -> Source {
    let (entry, _) = pick_entry(s.dim, s.k_form).expect("verified entry");
    Source::metal(SOURCE, entry)
}

fn metal_load(k: &IndexerMetal, _: usize) -> Option<Load> {
    let s = &k.0;
    let tk = tile_keys(s);
    let blocks_x = s.out_keys().div_ceil(tk);
    let q_bytes = s.q_form.bytes(s.rows * s.heads * s.dim) as f64;
    let k_bytes = s.k_form.bytes(s.rows * s.out_keys() * s.dim) as f64;
    let out_bytes = s.out_form.bytes(s.out_len()) as f64;
    Some(Load {
        threads: blocks_x * s.rows * THREADS,
        group: THREADS,
        trips: tk.div_ceil(64) as f64 * s.heads.div_ceil(8) as f64,
        reads: vec![k_bytes, q_bytes],
        written: out_bytes,
        flight: 512.0,
        ..Load::default()
    })
}

fn metal_launches(k: &IndexerMetal, step: &Step) -> Vec<Launch> {
    let s = &k.0;
    let tk = tile_keys(s);
    let (kept_mode, kept_param, page_shift, page_mask) = if s.ragged {
        kept_params(s.kept, s.context).expect("verified layout")
    } else {
        (0, s.out_keys(), 0, 0)
    };
    let slots = if s.ragged { step.read(3) } else { step.read(0) };
    let positions = if s.ragged { step.read(4) } else { step.read(0) };
    let page_table = if s.ragged {
        step.kept_for(s.kept, true).unwrap_or_else(|| step.read(3))
    } else {
        step.read(0)
    };
    let q_fmt = plain_fmt(s.q_form).expect("verified q_form");
    let w_fmt = plain_fmt(s.w_form).expect("verified w_form");
    let out_fmt = plain_fmt(s.out_form).expect("verified out_form");
    let flags = kept_mode
        | (usize::from(s.ragged) << 2)
        | (usize::from(s.masked) << 3)
        | (q_fmt << 4)
        | (w_fmt << 6)
        | (out_fmt << 8);
    let past = s.out_keys().saturating_sub(s.queries);
    let args = vec![
        step.read(0),
        step.read(1),
        step.read(2),
        slots,
        positions,
        page_table,
        step.out(),
        Arg::count(s.rows),
        Arg::count(s.out_keys()),
        Arg::size(s.keys),
        Arg::count(s.heads),
        Arg::count(s.dim),
        Arg::count(tk),
        Arg::count(kept_param),
        Arg::count(page_shift),
        Arg::count(page_mask),
        Arg::count(s.queries),
        Arg::count(past),
        Arg::count(s.window.0),
        Arg::count(s.window.1),
        Arg::count(flags),
    ];
    let shape = Shape::Grid {
        across: s.out_keys().div_ceil(tk),
        down: s.rows,
        threads: THREADS,
    };
    vec![Launch { args, shape }]
}
