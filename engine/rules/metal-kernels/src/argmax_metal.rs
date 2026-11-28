//! Native Metal coalesced argmax and small-k top-k kernels.

use tessera_lower::Backend;
use tessera_lower::form::Load as Stored;
use tessera_lower::sites;
use tessera_lower::{Draft, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Tree};

const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];
const SOURCE: &str = include_str!("argmax_metal.metal");
const SMALL_MAX_WIDTH: usize = 2048;
const SMALL_MAX_K: usize = 32;
const SPLIT_MIN_WIDTH: usize = 8192;

fn fixed(entry: &'static str) -> Source {
    Source::metal(SOURCE, entry)
}

fn plain_f32<K>(_: &K, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == "f32",
        _ => !form.narrow(),
    }
}

fn u32_out<K>(_: &K, form: &dyn Stored) -> bool {
    !form.narrow()
}

fn metal_needs(threads: usize, smem: usize) -> Needs {
    Needs {
        lanes: 32,
        group_bytes: smem,
        group_threads: threads,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn grid_launch(
    step: &Step,
    counts: &[usize],
    across: usize,
    down: usize,
    threads: usize,
) -> Vec<Launch> {
    let shape = Shape::Grid {
        across,
        down,
        threads,
    };
    vec![Launch::plain(step, counts, shape)]
}

fn grid_load(
    groups: usize,
    threads: usize,
    trips: f64,
    read_bytes: usize,
    write_bytes: usize,
    smem: usize,
) -> Option<Load> {
    Some(Load {
        threads: groups * threads,
        group: threads,
        trips,
        reads: vec![read_bytes as f64],
        written: write_bytes as f64,
        flight: threads as f64,
        group_bytes: smem as f64,
        ..Load::default()
    })
}

/// Single-launch coalesced Metal argmax, one block per row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "argmax_row_metal",
    sizes = [self.rows, self.width],
    work = self.rows * self.width,
    needs = metal_needs(256, 64),
    accepts = plain_f32,
    emits = u32_out,
    source = fixed("argmax_row_mtl"),
    launches = row_launches,
    load = row_load,
    unproven,
)]
struct ArgmaxRow {
    rows: usize,
    width: usize,
    first: usize,
    whole: usize,
}

fn row_launches(k: &ArgmaxRow, s: &Step) -> Vec<Launch> {
    grid_launch(s, &[k.rows, k.width, k.first, k.whole], k.rows, 1, 256)
}

fn row_load(k: &ArgmaxRow, _: usize) -> Option<Load> {
    grid_load(
        k.rows,
        256,
        k.width.div_ceil(1024) as f64,
        4 * k.rows * k.width,
        4 * k.rows,
        64,
    )
}

/// Stage 1 coalesced peak finder across split-width chunks.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "argmax_part_metal",
    unproven,
    sizes = [self.rows, self.width, self.parts],
    work = self.rows * self.width,
    needs = metal_needs(256, 64),
    accepts = plain_f32,
    emits = u32_out,
    source = fixed("argmax_part_mtl"),
    launches = part_launches,
    load = part_load,
)]
struct ArgmaxPart {
    rows: usize,
    width: usize,
    parts: usize,
}

fn part_launches(k: &ArgmaxPart, s: &Step) -> Vec<Launch> {
    grid_launch(s, &[k.rows, k.width, k.parts], k.parts, k.rows, 256)
}

fn part_load(k: &ArgmaxPart, _: usize) -> Option<Load> {
    grid_load(
        k.rows * k.parts,
        256,
        k.width.div_ceil(k.parts).div_ceil(1024) as f64,
        4 * k.rows * k.width,
        4 * k.rows * k.parts,
        64,
    )
}

/// Stage 2 argmax reduction across split-width chunk peaks.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "argmax_fin_metal",
    sizes = [self.rows, self.width, self.parts],
    work = self.rows * self.parts,
    needs = metal_needs(64, 16),
    accepts = plain_f32,
    emits = u32_out,
    source = fixed("argmax_fin_mtl"),
    launches = fin_launches,
    load = fin_load,
    unproven,
)]
struct ArgmaxFin {
    rows: usize,
    width: usize,
    parts: usize,
    first: usize,
    whole: usize,
}

fn fin_launches(k: &ArgmaxFin, s: &Step) -> Vec<Launch> {
    grid_launch(
        s,
        &[k.rows, k.width, k.parts, k.first, k.whole],
        k.rows,
        1,
        64,
    )
}

fn fin_load(k: &ArgmaxFin, _: usize) -> Option<Load> {
    grid_load(k.rows, 64, 1.0, 8 * k.rows * k.parts, 4 * k.rows, 16)
}

/// Single-launch shared-memory top-k for widths up to 2048.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "topk_small_metal",
    source = fixed("topk_small_mtl"),
    unproven,
    sizes = [self.rows, self.width, self.k],
    work = self.rows * self.width * self.k,
    needs = metal_needs(128, 8_224),
    accepts = plain_f32,
    emits = u32_out,
    launches = small_launches,
    load = small_load,
)]
struct TopKSmall {
    rows: usize,
    width: usize,
    k: usize,
    first: usize,
    whole: usize,
}

fn small_launches(k: &TopKSmall, s: &Step) -> Vec<Launch> {
    grid_launch(s, &[k.rows, k.width, k.k, k.first, k.whole], k.rows, 1, 128)
}

fn small_load(k: &TopKSmall, _: usize) -> Option<Load> {
    grid_load(
        k.rows,
        128,
        (k.k * k.width.div_ceil(128)) as f64,
        4 * k.rows * k.width,
        4 * k.rows * k.k,
        8_224,
    )
}

/// Stage 2 lazy chunk-pop top-k from stage 1 peaks.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "topk_select_metal",
    sizes = [self.rows, self.width, self.parts, self.k],
    work = self.rows * self.k * self.width.div_ceil(self.parts.max(1)),
    needs = metal_needs(256, 2_124),
    accepts = plain_f32,
    emits = u32_out,
    source = fixed("topk_select_mtl"),
    launches = select_launches,
    load = select_load,
    unproven,
)]
struct TopKSelect {
    rows: usize,
    width: usize,
    parts: usize,
    k: usize,
    first: usize,
    whole: usize,
}

fn select_launches(k: &TopKSelect, s: &Step) -> Vec<Launch> {
    grid_launch(
        s,
        &[k.rows, k.width, k.parts, k.k, k.first, k.whole],
        k.rows,
        1,
        256,
    )
}

fn select_load(k: &TopKSelect, _: usize) -> Option<Load> {
    let chunk = k.width.div_ceil(k.parts.max(1));
    grid_load(
        k.rows,
        256,
        (k.k * chunk.div_ceil(1024)) as f64,
        4 * k.rows * k.k * chunk,
        4 * k.rows * k.k,
        2_124,
    )
}

/// Returns Metal drafts for `argmax` (`k == 1`) and small-`k` `top_k`.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let sites::Top {
        x,
        rows,
        width,
        k,
        first,
        whole,
    } = tree.site(at)?;
    if rows == 0 || width == 0 || k == 0 || k > SMALL_MAX_K {
        return None;
    }
    if k == 1 {
        if rows < 32 && width > SPLIT_MIN_WIDTH {
            let parts = (64 / rows).clamp(2, width.div_ceil(1024)).min(64);
            return Some(vec![
                Draft::new(ArgmaxPart { rows, width, parts }, [x], rows * parts),
                Draft::new(
                    ArgmaxFin {
                        rows,
                        width,
                        parts,
                        first,
                        whole,
                    },
                    [Read::Node(x), Read::Draft(0)],
                    rows,
                ),
            ]);
        }
        return Some(vec![Draft::new(
            ArgmaxRow {
                rows,
                width,
                first,
                whole,
            },
            [x],
            rows,
        )]);
    }
    if width <= SMALL_MAX_WIDTH {
        return Some(vec![Draft::new(
            TopKSmall {
                rows,
                width,
                k,
                first,
                whole,
            },
            [x],
            rows * k,
        )]);
    }
    let parts = width.div_ceil(256).clamp(2, 256);
    Some(vec![
        Draft::new(ArgmaxPart { rows, width, parts }, [x], rows * parts),
        Draft::new(
            TopKSelect {
                rows,
                width,
                parts,
                k,
                first,
                whole,
            },
            [Read::Node(x), Read::Draft(0)],
            rows * k,
        ),
    ])
}

tessera_lower::rule!("argmax_metal", propose);
