//! Decoders over the shared decoder config, one line each.

use crate::arch::{Arch, Ask, Built, Spot, arch, find, known, unknown};
use crate::parts::{Cut, Part};
use crate::{Overrides, config, held, pull};
use std::error::Error;
use tessera_checkpoint::Shards;
use tessera_engine::decoder::{Config, Moe, Swiglu, stepper, taps};
use tessera_engine::{Graph, Norm};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// Graphs over the shared decoder config.
pub(crate) struct Family {
    name: &'static str,
    pub step: fn(&Config) -> Graph,
    pub taps: fn(&Config, &[u64]) -> Graph,
}

/// The decoder family registered as `name`.
pub(crate) fn family(name: &str) -> Result<&'static Family> {
    let arch = find(name).ok_or_else(|| unknown(name))?;
    FAMILIES
        .iter()
        .find(|f| f.name == arch.name)
        .ok_or_else(|| format!("{name} is not a decoder. Known: {}", known()).into())
}

/// The step graph and held weights `ask` names.
fn step(arch: &Arch, ask: &Ask) -> Result<Built> {
    let (_, mut c) = config(ask.dir, ask.sizes, ask.overrides)?;
    c.stack.steps = ask.steps;
    c.stack.taps = ask.taps.to_vec();
    let graph = (family(arch.name)?.step)(&c);
    held(ask, graph, c.stack.layers, &[])
}

/// A part's step graph, its weights from shards.
fn part(arch: &Arch, cut: &Cut) -> Result<Part> {
    let (_, c) = config(&cut.dir, cut.sized(256), &Overrides::default())?;
    let (dir, layers) = (cut.dir.clone(), c.stack.layers);
    Ok(Part::new((family(arch.name)?.step)(&c), move |g| {
        pull(g, &Shards::open(&dir)?, layers, &[])
    }))
}

/// Registers decoders by `model_type`, head norm, MLP and extra fields.
macro_rules! decoders {
    ($($name:literal: $norm:ty, $mlp:ty $(, $field:ident: $value:expr)*;)*) => {
        $(arch!($name, [Spot::Type($name)], step: step, part: part $(, $field: $value)*);)*

        const FAMILIES: &[Family] = &[$(Family {
            name: $name,
            step: stepper::<$norm, $mlp>,
            taps: taps::<$norm, $mlp>,
        }),*];
    };
}

decoders! {
    "qwen3": Norm, Swiglu, response: include_str!("response/qwen3.json");
    "llama": (), Swiglu;
    "granitemoe": (), Moe, response: include_str!("response/granite.json");
}
