//! Times kernels on this machine and stores what they cost.

mod grid;

// Linking the crate registers its rules.
use linear as _;

pub use grid::Grid;

use graph::{DimName, Graph};
use lower::{Device, Error, Feed, Run, Step, lower, select};
use std::path::PathBuf;

/// Whole passes over a plan, each step timed once.
const PASSES: usize = 3;

/// Times every launch of the plan, its inputs already fed.
pub fn time(runner: &dyn Run) -> Result<Vec<(Step, f64)>, Error> {
    let steps = &runner.plan().steps;
    // The first run pays for anything set up lazily.
    runner.submit(0..steps.len())?;
    let mut timed = Vec::with_capacity(PASSES * steps.len());
    // In plan order, so a launch finds its weights cold.
    for _ in 0..PASSES {
        for (i, step) in steps.iter().enumerate() {
            timed.push((step.clone(), runner.repeat(i, 1)?));
        }
    }
    Ok(timed)
}

/// Times `entry` under each competing rule. Returns the new launches.
pub fn measure(
    runner: &mut dyn Run,
    (g, entry): (&Graph, usize),
    sizes: &[(DimName, u64)],
    feeds: &[Feed],
    device: &mut Device,
) -> Result<Vec<(Step, f64)>, Error> {
    let contested = select(g, entry, sizes, &Device::guess())?.contested;
    let favored = contested.into_iter().map(|r| Device::guess().favoring(r));
    let mut added = Vec::new();
    for choice in [Device::guess()].into_iter().chain(favored) {
        let plan = lower(g, entry, sizes, &choice)?;
        if plan.steps.iter().all(|s| device.known(s)) {
            continue;
        }
        runner.replan(plan)?;
        runner.feed(feeds)?;
        let timed = time(runner)?;
        device.fit(timed.iter().map(|(step, seconds)| (step, *seconds)));
        added.extend(timed);
    }
    device.name = runner.device();
    Ok(added)
}

/// Where this machine keeps the samples for `device`.
pub fn path(device: &str) -> PathBuf {
    let plain = |c: char| if c.is_ascii_alphanumeric() { c } else { '-' };
    let file: String = device.chars().map(plain).collect();
    let dir = std::env::var_os("TESSERA_COSTS").map(PathBuf::from);
    let home = || std::env::home_dir().unwrap_or_else(std::env::temp_dir);
    let dir = dir.unwrap_or_else(|| home().join(".cache/tessera/costs"));
    dir.join(format!("{file}.json"))
}

/// The samples stored for the runner's device, or a guess.
pub fn stored(runner: &dyn Run) -> Result<Device, std::io::Error> {
    match Device::load(&path(&runner.device())) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Device::guess()),
        loaded => loaded,
    }
}

#[cfg(test)]
mod calibrate_test;
