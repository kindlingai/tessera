//! Decoders over the shared decoder config, one line each.

use crate::arch::{Arch, Ask, Built, Spot, arch, find, known, unknown};
use crate::parts::{Cut, Part, from_offer};
use crate::{config, held};
use std::error::Error;
use tessera_engine::decoder::{Config, Moe, SparseMoe, Swiglu, stepper_with, taps_with};
use tessera_engine::{Graph, Norm, Vector};

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
    held(ask, graph, c.stack.layers, &arch.rules())
}

/// A part's step graph, its weights from the checkpoint offer.
fn part(arch: &Arch, cut: &Cut) -> Result<Part> {
    let (_, c) = config(&cut.dir, cut.sized(256), cut.overrides)?;
    from_offer(
        (family(arch.name)?.step)(&c),
        cut.dir.clone(),
        cut.manifest,
        c.stack.layers,
        arch.rules(),
        cut.user.to_vec(),
        cut.json.to_string(),
    )
}

/// Registers decoders by model type, head norm, MLP, bias.
macro_rules! decoders {
    ($($name:literal: $norm:ty, $mlp:ty, $bias:ty $(, $field:ident: $value:expr)*;)*) => {
        $(arch!($name, [Spot::Type($name)], step: step, part: part $(, $field: $value)*);)*

        const FAMILIES: &[Family] = &[$(Family {
            name: $name,
            step: stepper_with::<$norm, $mlp, $bias>,
            taps: taps_with::<$norm, $mlp, $bias>,
        }),*];
    };
}

decoders! {
    "qwen2": (), Swiglu, Vector;
    "qwen3": Norm, Swiglu, ();
    "qwen3_moe": Norm, SparseMoe, (), rules: include_str!("../names/qwen3_moe.json");
    "llama": (), Swiglu, ();
    "granitemoe": (), Moe, ();
}
