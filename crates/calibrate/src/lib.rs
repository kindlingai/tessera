//! Times kernels on this machine and stores what they cost.

mod grid;

pub use grid::Grid;

use graph::{DimName, Graph};
use lower::{Device, Error, Feed, Run, Step, lower, select};
use std::path::PathBuf;

/// Seconds a short step repeats to fill.
const TARGET: f64 = 2e-3;

/// The most times one step repeats.
const MOST: usize = 64;

/// Steps this short are timed three times over.
const SHORT: f64 = 0.05;

/// Times the plan's distinct launches, its inputs already fed.
pub fn time(runner: &dyn Run) -> Result<Vec<(Step, f64)>, Error> {
    let steps = runner.plan().steps.clone();
    // The first run pays for anything set up lazily.
    runner.submit(0..steps.len())?;
    let mut timed: Vec<(Step, f64)> = Vec::new();
    for (i, step) in steps.into_iter().enumerate() {
        let same = |(s, _): &(Step, f64)| s.kernel == step.kernel && s.write.len == step.write.len;
        if timed.iter().any(same) {
            continue;
        }
        let mut least = runner.repeat(i, 1)?;
        let times = ((TARGET / least.max(1e-9)) as usize).clamp(1, MOST);
        let trials = if least < SHORT { 3 } else { 0 };
        // An in-place step repeats on its own output.
        for _ in 0..trials {
            least = least.min(runner.repeat(i, times)?);
        }
        timed.push((step, least));
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
