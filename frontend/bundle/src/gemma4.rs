//! A Gemma 4 Hugging Face directory as a step graph.

use crate::arch::{Arch, Ask, Built, Spot, arch};
use crate::held;
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

arch!(
    "gemma4",
    [Spot::Type("gemma4")],
    rules: include_str!("../names/gemma4.json"),
    step: step,
);
