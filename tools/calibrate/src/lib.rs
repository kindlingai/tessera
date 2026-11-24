//! Times kernels on this machine and stores what they cost.

pub mod bench;
mod grid;
pub mod guided;
pub mod plans;
pub mod probe;
pub mod watch;

// Linking the crates registers their rules.
use tessera_rules as _;

pub use grid::Grid;

use std::collections::BTreeMap;
use std::time::Instant;
use tessera_engine::Error;
use tessera_graph::{DimName, Graph};
use tessera_lower::{Device, Settings, Step, View, ZoneKind, lower, select};
use tessera_run::Run;
use tessera_sim::Chip;

/// Timed tries of each step. `fit` keeps a low one.
pub(crate) const TRIES: usize = 5;
/// Launches back to back in one try, at most.
pub(crate) const BURST: usize = 16;
/// Seconds one try aims to fill.
const AIM: f64 = 1e-3;

/// `tries` bursts of step `i`, each about `AIM` long.
fn bursts(runner: &mut dyn Run, i: usize, tries: usize, burst: usize) -> Result<Vec<f64>, Error> {
    let once = runner.repeat(i, 1)?;
    if tries == 1 && burst == 1 {
        return Ok(vec![once]);
    }
    let times = (AIM / once.max(1e-7)).clamp(1.0, burst as f64) as usize;
    let mut got = Vec::with_capacity(tries);
    for _ in 0..tries {
        got.push(runner.repeat(i, times)?);
    }
    Ok(got)
}

/// The low quartile: noise only adds time.
fn low(mut tries: Vec<f64>) -> f64 {
    tries.sort_by(f64::total_cmp);
    tries[(tries.len() - 1) / 4]
}

/// Times launches of the plan, skipping steps `skip` already knows.
fn time_steps(
    runner: &mut dyn Run,
    skip: Option<&Device>,
    (tries, burst): (usize, usize),
) -> Result<Vec<(Step, f64)>, Error> {
    let steps = runner.plan().steps.clone();
    // The first run pays for anything set up lazily.
    runner.submit(0..steps.len())?;
    let mut timed = Vec::new();
    let mut seen: Vec<(String, usize)> = Vec::new();
    // Bursts keep the clocks up, as a running plan does.
    for (i, step) in steps.iter().enumerate() {
        if let Some(device) = skip {
            let key = (format!("{:?}", step.kernel), step.write.len);
            if device.known(step) || seen.contains(&key) {
                continue;
            }
            seen.push(key);
        }
        for seconds in bursts(runner, i, tries, burst)? {
            timed.push((step.clone(), seconds));
        }
    }
    Ok(timed)
}

/// Times every launch of the plan, its inputs already fed.
pub fn time(runner: &mut dyn Run) -> Result<Vec<(Step, f64)>, Error> {
    time_steps(runner, None, (TRIES, BURST))
}

/// Each distinct launch's seconds with its weights read cold.
fn time_cold(
    runner: &mut dyn Run,
    (tries, burst): (usize, usize),
) -> Result<Vec<(Step, f64)>, Error> {
    let steps = runner.plan().steps.clone();
    let all = 0..steps.len();
    runner.submit(all.clone())?;
    // Each distinct launch alone, warm: its weights stay cached.
    let mut kinds: Vec<(String, usize, f64)> = Vec::new();
    let mut warm = Vec::with_capacity(steps.len());
    for (i, step) in steps.iter().enumerate() {
        let key = format!("{:?}", step.kernel);
        let seconds = match kinds.iter().find(|(k, ..)| *k == key) {
            Some((_, _, seconds)) => *seconds,
            None => {
                let seconds = low(bursts(runner, i, tries, burst)?);
                kinds.push((key, i, seconds));
                seconds
            }
        };
        warm.push(seconds);
    }
    let mut whole = Vec::with_capacity(tries);
    for _ in 0..tries {
        let start = Instant::now();
        let clocked = runner.clocked(all.clone())?;
        whole.push(clocked.unwrap_or_else(|| start.elapsed().as_secs_f64()));
    }
    // Both matmuls of a link read equal bytes.
    let extra = (low(whole) - warm.iter().sum::<f64>()).max(0.0) / steps.len() as f64;
    let cold = kinds
        .into_iter()
        .map(|(_, i, seconds)| (steps[i].clone(), seconds + extra));
    Ok(cold.collect())
}

/// Whether `device` already has a sample for `step`.
fn known(device: &Device, step: &Step, cold: bool) -> bool {
    match cold {
        true => device.known_cold(step),
        false => device.known(step),
    }
}

/// Plans for `(g, entry)` at `sizes` with unknown steps.
fn pending(
    (g, entry): (&Graph, usize),
    sizes: &[(DimName, u64)],
    device: &Device,
    settings: &Settings,
    cold: bool,
) -> Result<Vec<tessera_lower::Plan>, Error> {
    let (pins, platform) = (device.pins.clone(), device.platform.clone());
    let guess = || {
        Device::guess(settings)
            .on(platform.clone())
            .pinned(pins.clone())
    };
    let mut plans = Vec::new();
    for choice in choices((g, entry), sizes, &guess)? {
        let plan = lower(g, entry, sizes, &choice)?;
        if !plan.steps.iter().all(|s| known(device, s, cold)) {
            plans.push(plan);
        }
    }
    Ok(plans)
}

/// Times each plan in `plans` that still has unknown steps.
fn time_plans(
    runner: &mut dyn Run,
    plans: &[tessera_lower::Plan],
    feeds: &[View],
    device: &mut Device,
    (cold, tries, burst): (bool, usize, usize),
) -> Result<Vec<(Step, f64)>, Error> {
    let mut added = Vec::new();
    for plan in plans {
        if plan.steps.iter().all(|s| known(device, s, cold)) {
            continue;
        }
        runner.replan(plan)?;
        runner.feed(feeds)?;
        let timed = if cold {
            time_cold(runner, (tries, burst))?
        } else {
            time_steps(runner, Some(device), (tries, burst))?
        };
        let pairs = timed.iter().map(|(step, seconds)| (step, *seconds));
        match cold {
            true => device.fit_cold(pairs),
            false => device.fit(pairs),
        }
        added.extend(timed);
    }
    device.name = runner.device();
    Ok(added)
}

/// Times `entry` under each competing rule, cold if asked.
fn measure(
    runner: &mut dyn Run,
    (g, entry): (&Graph, usize),
    sizes: &[(DimName, u64)],
    feeds: &[View],
    device: &mut Device,
    (settings, cold): (&Settings, bool),
) -> Result<Vec<(Step, f64)>, Error> {
    measure_with(
        runner,
        (g, entry),
        sizes,
        feeds,
        device,
        (settings, cold),
        (TRIES, BURST),
    )
}

/// Times `entry` with custom `tries` and `burst` limits.
fn measure_with(
    runner: &mut dyn Run,
    (g, entry): (&Graph, usize),
    sizes: &[(DimName, u64)],
    feeds: &[View],
    device: &mut Device,
    (settings, cold): (&Settings, bool),
    (tries, burst): (usize, usize),
) -> Result<Vec<(Step, f64)>, Error> {
    if device.platform != runner.platform() {
        *device = device.clone().on(runner.platform());
    }
    let plans = pending((g, entry), sizes, device, settings, cold)?;
    time_plans(runner, &plans, feeds, device, (cold, tries, burst))
}

/// The guess, each competing rule, then each rule's ways.
fn choices(
    (g, entry): (&Graph, usize),
    sizes: &[(DimName, u64)],
    guess: &dyn Fn() -> Device,
) -> Result<Vec<Device>, Error> {
    let picked = select(g, entry, sizes, &guess())?;
    let mut all: Vec<Device> = [guess()].into_iter().collect();
    all.extend(picked.contested.iter().map(|r| guess().favoring(*r)));
    // A host emulates GPU ways slowly. They never win there.
    let gpu = guess().cluster.kind(0) == ZoneKind::Gpu;
    for (rule, ways) in &picked.ways {
        if ways.len() < 2 || !gpu {
            continue;
        }
        for way in ways {
            // Bars every kernel the other ways launch.
            let mut device = guess().favoring(*rule);
            for name in ways.iter().flatten() {
                if !way.contains(name) {
                    device = device.without(*name);
                }
            }
            all.push(device);
        }
    }
    Ok(all)
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
pub fn judge(chip: &dyn Chip, timed: &[(Step, f64)], settings: &Settings) -> Vec<Verdict> {
    // Each distinct launch counts once, at its median time.
    let mut device = Device::guess(settings);
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

#[cfg(test)]
mod calibrate_test;
