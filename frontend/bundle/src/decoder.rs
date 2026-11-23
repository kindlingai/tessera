//! Decoders over the shared decoder config, one line each.

use crate::arch::{Arch, Ask, Built};
use crate::parts::{Cut, Part};
use crate::{Overrides, config, held, pull};
use std::error::Error;
use tessera_checkpoint::Shards;
use tessera_engine::decoder::{Config, Moe, Swiglu};
use tessera_engine::{Graph, Norm};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn graph(arch: &Arch, c: &Config) -> Graph {
    let family = arch.family.as_ref().expect("a decoder has a family");
    (family.step)(c)
}

/// The step graph and held weights `ask` names.
pub(crate) fn step(arch: &Arch, ask: &Ask) -> Result<Built> {
    let (_, mut c) = config(ask.dir, ask.sizes, ask.overrides)?;
    c.stack.steps = ask.steps;
    c.stack.taps = ask.taps.to_vec();
    held(ask, graph(arch, &c), c.stack.layers, &[])
}

/// A part's step graph, its weights from shards.
pub(crate) fn part(arch: &Arch, cut: &Cut) -> Result<Part> {
    let (_, c) = config(&cut.dir, cut.sized(256), &Overrides::default())?;
    let (dir, layers) = (cut.dir.clone(), c.stack.layers);
    Ok(Part::new(graph(arch, &c), move |g| {
        pull(g, &Shards::open(&dir)?, layers, &[])
    }))
}

/// Registers a decoder by `model_type`, head norm and MLP.
macro_rules! decoder {
    ($name:literal, $norm:ty, $mlp:ty) => {
        $crate::arch::arch!(
            $name,
            [$crate::arch::Spot::Type($name)],
            step: $crate::decoder::step,
            part: $crate::decoder::part,
            family: $crate::arch::Family {
                step: tessera_engine::decoder::stepper::<$norm, $mlp>,
                taps: tessera_engine::decoder::taps::<$norm, $mlp>,
            },
        );
    };
}

decoder!("qwen3", Norm, Swiglu);
decoder!("llama", (), Swiglu);
decoder!("granitemoe", (), Moe);
