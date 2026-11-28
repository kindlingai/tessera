//! Metal bf16 matrix-unit matmul for dense prefill.

use tessera_graph::Order;
use tessera_linear::Product;
use tessera_lower::form::BF16_FORM;
use tessera_lower::{
    Backend, ByBackend, Draft, GUESS, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Tree,
};

const BLOCK: (usize, usize, usize) = (64, 64, 32);
const THREADS: usize = 256;
/// Staged bf16 elements a row holds, padding included.
const STRIDE: usize = 40;
const SMEM: usize = 2 * (BLOCK.0 + BLOCK.1) * STRIDE;
/// Row blocks in one band when `x` outweighs `w`.
const BAND: usize = 8;
const MIN_ROWS: usize = 32;
const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];
const SOURCE: &str = include_str!("tiled_metal.metal");

/// `x[m, k]` in f32 times bf16 `w[n, k]` transposed.
#[derive(Clone, Debug, Kernel)]
#[launch(
    name = self.product.form.suffixed("tiled_metal"),
    sizes = self.product.sizes(),
    work = self.product.work(),
    guess = GUESS,
    accepts = self.product.accepts(input, form),
    emits = self.product.emits(form),
    needs = needs(),
    order = Order::PerConfig,
    source = source(),
    launches = launches,
    load = load,
)]
struct TiledMetal {
    product: Product,
}

fn source() -> Source {
    Source::Fixed {
        text: ByBackend([SOURCE, "", ""]),
        entry: "tiled_metal_bf16",
    }
}

fn needs() -> Needs {
    Needs {
        lanes: 32,
        group_bytes: SMEM,
        group_threads: THREADS,
        tile: true,
        compiler: true,
        msl_bfloat: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn groups(p: &Product) -> (usize, usize) {
    (p.m.div_ceil(BLOCK.0), p.n.div_ceil(BLOCK.1))
}

fn load(t: &TiledMetal, _: usize) -> Option<Load> {
    let Product { m, n, k, .. } = t.product;
    let (gm, gn) = groups(&t.product);
    let (bm, bn, bk) = BLOCK;
    // `x` stays cached. `w` streams once per band.
    let band = if n >= 2 * m { gm } else { BAND.min(gm) };
    let mut reads = vec![(4 * m * k) as f64];
    reads.extend((0..gm.div_ceil(band)).map(|_| (2 * n * k) as f64));
    let mut load = Load {
        threads: gm * gn * THREADS,
        group: THREADS,
        trips: k.div_ceil(bk) as f64,
        reads,
        written: (4 * m * n) as f64,
        flight: 16.0,
        group_bytes: SMEM as f64,
        ..Load::default()
    };
    load.trip.bf16_flops = (2 * bm * bn * bk / THREADS) as f64;
    // Staged bf16 pairs fill one float each.
    load.trip.shared = ((bm + bn) * bk / 2 / THREADS) as f64;
    // Two float4 loads of `x`, two 8-byte loads of `w`.
    load.trip.loads = 4.0;
    load.trip.width = 12.0;
    load.trip.fresh = 1.0;
    load.trip.stores = 4.0;
    load.trip.held = (bm * bn / THREADS) as f64;
    Some(load)
}

fn launches(t: &TiledMetal, step: &Step) -> Vec<Launch> {
    let Product { m, n, k, .. } = t.product;
    let (gm, gn) = groups(&t.product);
    let shape = Shape::Grid {
        across: gm * gn,
        down: 1,
        threads: THREADS,
    };
    t.product.launch(step, &[m, n, k], shape)
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (product, reads) = Product::find(tree, at)?;
    let plain = product.picked.is_none() && product.near.is_none();
    let bare = product.after.stages.is_empty() && product.other_forms.is_empty();
    let forms = product.x_form.name() == "f32"
        && product.form.name() == BF16_FORM.name()
        && product.out_form.name() == "f32";
    let fits = product.m >= MIN_ROWS && product.k.is_multiple_of(8);
    if !(plain && bare && forms && fits) {
        return None;
    }
    let outputs = product.outputs();
    Some(vec![Draft::new(TiledMetal { product }, reads, outputs)])
}

tessera_lower::rule!("tiled_metal", propose);
