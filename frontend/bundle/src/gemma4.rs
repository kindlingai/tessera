//! A Gemma 4 Hugging Face directory as a step graph.

use crate::arch::{Arch, Ask, Built, Spot, arch};
use crate::held;
use crate::parts::{Cut, Part, from_offer};
use std::error::Error;
use tessera_engine::Linear;
use tessera_engine::gemma4::{Config, stepper};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn step(arch: &Arch, ask: &Ask) -> Result<Built> {
    if !ask.taps.is_empty() {
        return Err("gemma4 does not support step taps yet".into());
    }
    let [slots, context] = ask.sizes;
    let mut c = (ask.overrides).config(ask.dir, |k| Config::read(k, slots, context))?;
    c.stack.steps = ask.steps;
    held(ask, stepper::<Linear>(&c), c.stack.layers, &arch.rules())
}

fn part(arch: &Arch, cut: &Cut) -> Result<Part> {
    let [slots, context] = cut.sized(256);
    let c = (cut.overrides).config(&cut.dir, |k| Config::read(k, slots, context))?;
    from_offer(
        stepper::<Linear>(&c),
        cut.dir.clone(),
        cut.manifest,
        c.stack.layers,
        arch.rules(),
        cut.user.to_vec(),
        cut.json.to_string(),
    )
}

arch!(
    "gemma4",
    [Spot::Type("gemma4"), Spot::Type("gemma4_unified")],
    rules: include_str!("../names/gemma4.json"),
    step: step,
    part: part,
);
