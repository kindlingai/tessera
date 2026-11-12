//! Times kernels on this machine and stores what they cost.

mod grid;
pub mod probe;

// Linking the crates registers their rules.
use kernels as _;
use linear as _;

pub use grid::Grid;

use graph::{DimName, Graph};
use lower::{Device, Error, Feed, Run, Step, lower, select};
use sim::Chip;
use std::collections::BTreeMap;
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
    // The same pins, or the weights would load again.
    let pins = device.pins.clone();
    let guess = || Device::guess().pinned(pins.clone());
    let contested = select(g, entry, sizes, &guess())?.contested;
    let favored = contested.into_iter().map(|r| guess().favoring(r));
    let mut added = Vec::new();
    for choice in [guess()].into_iter().chain(favored) {
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

/// A chip model against the timed launches of one kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    pub kernel: &'static str,
    pub launches: usize,
    /// Seconds the launches took, all together.
    pub seconds: f64,
    /// Estimate over timed: the lowest, median and highest.
    pub ratio: [f64; 3],
}

/// Compares `chip`'s estimates with timed launches, by kernel.
pub fn judge(chip: &dyn Chip, timed: &[(Step, f64)]) -> Vec<Verdict> {
    // Each distinct launch counts once, at its median time.
    let mut device = Device::guess();
    device.fit(timed.iter().map(|(step, seconds)| (step, *seconds)));
    let mut seen: Vec<&Step> = Vec::new();
    let mut by: BTreeMap<&'static str, (f64, Vec<f64>)> = BTreeMap::new();
    for (step, _) in timed {
        let same = |s: &&Step| s.kernel == step.kernel && s.write.len == step.write.len;
        let load = step.kernel.load(step.write.len);
        let Some(load) = load.filter(|_| !seen.iter().any(same)) else {
            continue;
        };
        seen.push(step);
        let seconds = device.seconds(&*step.kernel, step.write.len);
        let (total, ratios) = by.entry(step.kernel.name()).or_default();
        *total += seconds;
        ratios.push(chip.estimate(&load).seconds / seconds);
    }
    let verdict = |(kernel, (seconds, mut ratios)): (&'static str, (f64, Vec<f64>))| {
        ratios.sort_by(f64::total_cmp);
        let ratio = [
            ratios[0],
            ratios[ratios.len() / 2],
            ratios[ratios.len() - 1],
        ];
        Verdict {
            kernel,
            launches: ratios.len(),
            seconds,
            ratio,
        }
    };
    by.into_iter().map(verdict).collect()
}

/// One kernel's timed launches against the chip's best case.
#[derive(Clone, Debug, PartialEq)]
pub struct Headroom {
    pub kernel: &'static str,
    pub launches: usize,
    /// Seconds the launches took, all together.
    pub seconds: f64,
    /// The least seconds their work needs, where a load says.
    pub bound: Option<f64>,
}

/// Sets each kernel's time beside the least its work needs.
pub fn headroom(chip: &dyn Chip, timed: &[(Step, f64)]) -> Vec<Headroom> {
    let mut by: BTreeMap<&'static str, Headroom> = BTreeMap::new();
    for (step, seconds) in timed {
        let kernel = step.kernel.name();
        let bound = step.kernel.load(step.write.len).map(|l| chip.bound(&l));
        let h = by.entry(kernel).or_insert(Headroom {
            kernel,
            launches: 0,
            seconds: 0.0,
            bound: bound.map(|_| 0.0),
        });
        (h.launches, h.seconds) = (h.launches + 1, h.seconds + seconds);
        h.bound = h.bound.zip(bound).map(|(sum, b)| sum + b);
    }
    by.into_values().collect()
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
