//! Metal bf16 matrix-unit tiles for wide ragged prefill.

use tessera_attention::flash::THREADS;
use tessera_attention::flash::sunk::{
    BK, FlashSunk, GUESS, LACKS, check_sunk, sited, sunk_accepts, sunk_emits, sunk_launches,
    why_not,
};
use tessera_attention::listed::Sizes;
use tessera_codegen::prelude::*;
use tessera_graph::Order;
use tessera_lower::{ByBackend, Draft, Kernel, Launch, Needs, Source, Step, T, Tree};

/// Queries and keys per block.
const BLOCK: usize = 32;
/// The one head width the tiles are cut for.
const WIDTH: usize = 128;
/// Staged bf16 elements per row of `WIDTH`, padding included.
const WIDE: usize = 136;
/// Staged elements per row of scores, padding included.
const NEAR: usize = 40;
/// Bytes of queries, keys, weights, scores and row state.
const SMEM: usize = BLOCK * (2 * 2 * WIDE + (2 + 4) * NEAR + 4 * 4);
/// bf16 tiles run four times the f32 rate.
const SPEEDUP: f64 = 4.0;
const SOURCE: &str = include_str!("sunk_metal.metal");

/// `FlashSunk` at width 128, on bf16 matrix units.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "flash_sunk_metal",
    order = Order::PerConfig,
    sizes = [self.0.0.len, self.0.0.keys.most(), self.0.seen(), self.0.1],
    work = self.0.0.len * self.0.seen() * self.0.1,
    guess = GUESS / SPEEDUP,
    needs = needs(),
    accepts = sunk_accepts(&self.0, input, form),
    emits = sunk_emits(&self.0, form),
    source = source(),
    launches = launches,
)]
struct SunkMetal(FlashSunk);

fn source() -> Source {
    Source::Fixed {
        text: ByBackend([SOURCE, "", ""]),
        entry: "flash_sunk_metal",
    }
}

fn needs() -> Needs {
    Needs {
        lanes: LANES,
        group_bytes: SMEM,
        group_threads: THREADS as usize,
        tile: true,
        compiler: true,
        msl_bfloat: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn launches(k: &SunkMetal, step: &Step) -> Vec<Launch> {
    sunk_launches(&k.0, step)
}

/// Dense and ring layouts read one scalar alike.
fn fits(s: &Sizes) -> bool {
    let plain = ["dense", "ring"].contains(&s.kept.name());
    plain && s.width == WIDTH && check_sunk(s, BLOCK as u32).is_ok()
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (sunk, reads) = sited(tree, at, |s| check_sunk(s, BK).is_ok() && fits(s))?;
    Some(vec![Draft::new(SunkMetal(sunk), reads, sunk.0.len)])
}

tessera_lower::rule!("flash_sunk_metal", propose, why = why_not, lacks = LACKS);
