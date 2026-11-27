//! Runners that count the bytes of plans, and allocate nothing.

use crate::Error;
use crate::run::{Base, Run};
use std::collections::HashMap;
use std::ops::{Add, Range};
use std::sync::Mutex;
use tessera_codegen::Precision;
use tessera_engine_settings::Settings;
use tessera_gpukit::platform::{host_available, host_memory};
use tessera_gpukit::{Limits, Platform};
use tessera_lower::{Plan, Weight};
use tessera_mem::View;

/// Bytes of runner buffers, by contents.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sizes {
    pub weights: u64,
    pub kept: u64,
    /// Scratch, in power-of-two allocations.
    pub scratch: u64,
    /// The largest weight, which a load may copy.
    pub largest: u64,
}

impl Add for Sizes {
    type Output = Sizes;

    fn add(self, o: Sizes) -> Sizes {
        Sizes {
            weights: self.weights + o.weights,
            kept: self.kept + o.kept,
            scratch: self.scratch + o.scratch,
            largest: self.largest.max(o.largest),
        }
    }
}

impl Sizes {
    fn total(self) -> u64 {
        self.weights + self.kept + self.scratch + self.largest
    }
}

/// Live runner byte counts and their peak sum.
#[derive(Default)]
struct Tracked {
    live: Vec<Option<Sizes>>,
    peak: Sizes,
}

impl Tracked {
    fn record(&mut self, at: usize, sizes: Sizes) {
        if let Some(entry) = self.live.get_mut(at) {
            let merged = match *entry {
                Some(prev) => Sizes {
                    weights: sizes.weights.max(prev.weights),
                    kept: sizes.kept.max(prev.kept),
                    scratch: sizes.scratch,
                    largest: sizes.largest.max(prev.largest),
                },
                None => sizes,
            };
            *entry = Some(merged);
            let now = (self.live.iter().flatten().copied()).fold(Sizes::default(), Add::add);
            if now.total() >= self.peak.total() {
                self.peak = now;
            }
        }
    }
}

/// Each sizing runner's sizes, once sizing is on.
static SIZED: Mutex<Option<Tracked>> = Mutex::new(None);

/// From now on, `open` gives runners that only count bytes.
pub fn sizing() {
    let mut sized = SIZED.lock().expect("sizes");
    *sized = Some(Tracked::default());
}

/// Whether `open` gives runners that only count bytes.
pub fn counting() -> bool {
    SIZED.lock().expect("sizes").is_some() || crate::check::active()
}

/// The peak bytes of runners live together since `sizing`.
pub fn sized() -> Sizes {
    let sized = SIZED.lock().expect("sizes");
    sized.as_ref().map_or(Sizes::default(), |t| t.peak)
}

/// Host memory: total, then available without swapping.
pub fn host() -> [u64; 2] {
    [host_memory(), host_available()]
}

/// The platform a runner for `settings` would report.
pub fn platform(settings: &Settings) -> Platform {
    let host = || Platform::host(Limits::host(settings.host_lanes));
    crate::asked(settings).unwrap_or_else(host)
}

/// Free bytes on the GPU `settings` picks, if known.
pub fn free(settings: &Settings) -> Option<u64> {
    crate::free(settings)
}

/// Free display-carveout bytes `atticd` can lend now.
pub fn attic(settings: &Settings) -> u64 {
    if settings.gpu() != Some(tessera_engine_settings::Backend::Cuda) {
        return 0;
    }
    tessera_gpukit::attic::available(settings)
}

/// Dry-run runner that adopts plans through `on_plan` without executing.
struct Passive<F> {
    base: Base,
    platform: Platform,
    label: &'static str,
    on_plan: F,
}

impl<F: FnMut(&Plan) + Send> Passive<F> {
    fn adopt(&mut self, plan: &Plan) {
        self.base.adopt(plan);
        (self.on_plan)(plan);
    }
}

/// Opens a dry-run runner that never allocates or executes.
pub(crate) fn passive<F>(
    plan: &Plan,
    platform: Platform,
    label: &'static str,
    on_plan: F,
) -> Box<dyn Run>
where
    F: FnMut(&Plan) + Send + 'static,
{
    let mut runner = Passive {
        base: Base::default(),
        platform,
        label,
        on_plan,
    };
    runner.adopt(plan);
    Box::new(runner)
}

/// Byte counter dropped when its sizing runner closes.
struct Counter {
    at: usize,
    scratch: HashMap<u32, Vec<u64>>,
}

impl Counter {
    fn count(&mut self, plan: &Plan) {
        let mut owned = vec![0u8; plan.buffers.len()];
        for l in &plan.weights {
            owned[l.slot.buffer as usize] = 1;
        }
        for (_, s) in &plan.kept {
            owned[s.buffer as usize] = 2;
        }
        if let Some(s) = plan.table {
            owned[s.buffer as usize] = 2;
        }
        let mut sizes = Sizes::default();
        let mut seen: HashMap<u32, usize> = HashMap::new();
        for (b, (&bytes, kind)) in plan.buffers.iter().zip(owned).enumerate() {
            let bytes = bytes as u64;
            match kind {
                1 => sizes.weights += bytes,
                2 => sizes.kept += bytes,
                _ => {
                    let zone = plan.zone(b as u32);
                    let slot = seen.entry(zone).or_insert(0);
                    let held = self.scratch.entry(zone).or_default();
                    if held.len() <= *slot {
                        held.push(0);
                    }
                    let want = bytes.max(4).next_power_of_two();
                    held[*slot] = held[*slot].max(want);
                    *slot += 1;
                }
            }
        }
        sizes.scratch = self.scratch.values().flatten().sum();
        let largest = plan.weights.iter().map(|l| l.slot.bytes() as u64).max();
        sizes.largest = largest.unwrap_or(0);
        let mut sized = SIZED.lock().expect("sizes");
        if let Some(tracked) = sized.as_mut() {
            tracked.record(self.at, sizes);
        }
    }
}

impl Drop for Counter {
    fn drop(&mut self) {
        if let Ok(mut sized) = SIZED.lock()
            && let Some(tracked) = sized.as_mut()
            && let Some(entry) = tracked.live.get_mut(self.at)
        {
            *entry = None;
        }
    }
}

/// Opens a sizing-only runner for `plan`.
pub(crate) struct Sizer;

impl Sizer {
    /// A sizer for `plan`, if sizing is on.
    pub(crate) fn open(plan: &Plan, settings: &Settings) -> Option<Box<dyn Run>> {
        let at = {
            let mut sized = SIZED.lock().expect("sizes");
            let tracked = sized.as_mut()?;
            tracked.live.push(None);
            tracked.live.len() - 1
        };
        let mut counter = Counter {
            at,
            scratch: HashMap::new(),
        };
        Some(passive(plan, platform(settings), "sizing only", move |p| {
            counter.count(p);
        }))
    }
}

impl<F: FnMut(&Plan) + Send> Run for Passive<F> {
    fn base(&self) -> &Base {
        &self.base
    }

    fn base_mut(&mut self) -> &mut Base {
        &mut self.base
    }

    fn replan(&mut self, plan: &Plan) -> Result<(), Error> {
        self.adopt(plan);
        Ok(())
    }

    fn submit(&mut self, _: Range<usize>) -> Result<(), Error> {
        Err(Error::Sizing)
    }

    fn repeat(&mut self, _: usize, _: usize) -> Result<f64, Error> {
        Err(Error::Sizing)
    }

    fn device(&self) -> String {
        format!(
            "{} {}, {}",
            self.platform.backend, self.platform.name, self.label
        )
    }

    fn platform(&self) -> Platform {
        self.platform.clone()
    }

    fn precision(&self) -> Precision {
        Precision::EXACT
    }

    fn load_weight(&mut self, _: u32, _: &Weight) -> Result<(), Error> {
        Ok(())
    }

    fn load(&mut self, _: u32, _: View) -> Result<(), Error> {
        Ok(())
    }
}
