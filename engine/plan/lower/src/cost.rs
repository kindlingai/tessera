//! What launches cost on one device, from timed samples.

use crate::form::Pins;
use crate::kernel::Kernel;
use crate::plan::{Plan, Step};
use crate::recipe::Recipe;
use crate::run::Run;
use codegen::kept::Layout;
use gpukit::{Limits, Platform, Settings};
use serde::{Deserialize, Serialize};
use sim::{Chip, Cluster, Fit, Pool};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Bumped whenever a stored device changes meaning.
const VERSION: u32 = 1;

/// What a plan uses of each resource.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cost {
    pub seconds: f64,
    /// Bytes its launches read and write.
    pub bytes: u64,
    pub launches: usize,
}

/// One kernel's timed launch.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Sample {
    /// The kernel as it prints: its name and every size.
    pub kernel: String,
    /// A hash of its source when it was timed.
    pub source: u64,
    pub sizes: Vec<usize>,
    pub work: f64,
    pub seconds: f64,
    /// Logarithms of the sizes and the work.
    #[serde(skip)]
    at: Vec<f64>,
}

/// Where a launch lies among samples: nearby ones cost alike.
fn place(sizes: &[usize], work: f64) -> Vec<f64> {
    let sizes = sizes.iter().map(|s| (*s).max(1) as f64);
    sizes.chain([work.max(1.0)]).map(f64::ln).collect()
}

impl Sample {
    pub fn new(step: &Step, seconds: f64) -> Self {
        let (sizes, work) = (step.kernel.sizes(), step.kernel.work(step.write.len));
        Self {
            kernel: format!("{:?}", step.kernel),
            source: step.kernel.source().hash(),
            at: place(&sizes, work),
            sizes,
            work,
            seconds,
        }
    }

    /// The same kernel and sizes, whatever its source or time.
    fn same(&self, other: &Sample) -> bool {
        self.kernel == other.kernel && self.work == other.work
    }
}

/// One device's timed launches, by kernel name.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Device {
    /// The hardware that ran the samples.
    pub name: String,
    /// The chip's constants, as probes fitted them.
    pub fit: Option<Fit>,
    /// The model those constants make.
    #[serde(skip)]
    pool: Option<Pool>,
    /// The zones plans may span. Zone 0 is this GPU.
    #[serde(default)]
    pub cluster: Cluster,
    /// The hardware as asked, with its table row.
    #[serde(default)]
    pub platform: Platform,
    timed: BTreeMap<String, Vec<Sample>>,
    /// A rule to pick wherever it applies, whatever it costs.
    #[serde(skip)]
    favor: Option<String>,
    /// Kernels never to choose.
    #[serde(skip)]
    barred: Vec<String>,
    /// The form each weight is stored in.
    #[serde(skip)]
    pub pins: Pins,
    /// Where kept rows lie.
    #[serde(skip, default = "dense")]
    pub kept: &'static dyn Layout,
    /// Fresh after every change that could alter a plan.
    #[serde(skip)]
    generation: u64,
    /// Estimates already made, by kernel and sizes.
    #[serde(skip)]
    memo: RefCell<Memo>,
}

/// Seconds by kernel name, sizes and work.
type Memo = HashMap<(&'static str, Vec<usize>, u64), f64>;

/// Each generation is handed out once per process.
static GENERATIONS: AtomicU64 = AtomicU64::new(1);

fn fresh() -> u64 {
    GENERATIONS.fetch_add(1, Ordering::Relaxed)
}

/// Equal in all they hold. The generation is not held.
impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.fit == other.fit
            && self.cluster == other.cluster
            && self.platform == other.platform
            && self.timed == other.timed
            && self.favor == other.favor
            && self.barred == other.barred
            && self.pins == other.pins
            && self.kept == other.kept
    }
}

fn dense() -> &'static dyn Layout {
    codegen::kept::DENSE
}

impl Default for Device {
    fn default() -> Self {
        Self {
            name: String::new(),
            fit: None,
            pool: None,
            cluster: Cluster::default(),
            platform: Platform::default(),
            timed: BTreeMap::new(),
            favor: None,
            barred: Vec::new(),
            pins: Pins::default(),
            kept: dense(),
            generation: 0,
            memo: RefCell::default(),
        }
    }
}

/// What every launch is guessed to cost before any work.
const FIXED: f64 = 1e-4;

/// How many samples one estimate blends.
const NEAR: usize = 3;

impl Device {
    /// The device `open` picks under `settings`, before any samples.
    pub fn guess(settings: &Settings) -> Self {
        let platform = match settings.gpu() {
            None => Platform::host(crate::cpu::limits(settings.host_lanes)),
            Some(_) => Platform::assumed(Limits::assumed()),
        };
        Self {
            platform,
            ..Self::default()
        }
    }

    /// Where this machine keeps the samples for `device`.
    pub fn path(settings: &Settings, device: &str) -> Option<PathBuf> {
        let plain = |c: char| if c.is_ascii_alphanumeric() { c } else { '-' };
        let file: String = device.chars().map(plain).collect();
        Some(settings.costs.as_ref()?.join(format!("{file}.json")))
    }

    /// The samples stored for the runner's platform, or a guess.
    pub fn stored(runner: &dyn Run, settings: &Settings) -> Result<Self, Error> {
        let platform = runner.platform();
        let loaded = Self::path(settings, &platform.id()).map(|p| Self::load(&p));
        let device = match loaded {
            Some(Err(e)) if e.kind() == ErrorKind::NotFound => Self::guess(settings),
            Some(loaded) => loaded?,
            None => Self::guess(settings),
        };
        Ok(device.on(platform))
    }

    /// The same device, choosing as `recipe` says.
    pub fn with(mut self, recipe: &Recipe) -> Self {
        self.pins = recipe.pins.clone();
        self.favor = recipe.favor.clone();
        self.barred = recipe.barred.clone();
        self.kept = recipe.kept;
        self.changed()
    }

    /// The same device, on this hardware.
    pub fn on(mut self, platform: Platform) -> Self {
        self.platform = platform;
        self.changed()
    }

    /// The same device, with a chip model for unmeasured kernels.
    pub fn fitted(mut self, fit: Fit) -> Self {
        (self.fit, self.pool) = (Some(fit), Some(fit.pool("fitted")));
        self.changed()
    }

    fn changed(mut self) -> Self {
        self.generation = fresh();
        self.memo.get_mut().clear();
        self
    }

    /// Changes with anything that could alter a plan.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The same device, with weights stored as `pins` say.
    pub fn pinned(mut self, pins: Pins) -> Self {
        self.pins = pins;
        self.changed()
    }

    /// The same device, over these zones and links.
    pub fn zoned(mut self, cluster: Cluster) -> Self {
        self.cluster = cluster;
        self.changed()
    }

    pub fn zones(&self) -> u32 {
        self.cluster.zones.len() as u32
    }

    /// Seconds to move `bytes` between zones, if linked.
    pub fn transfer(&self, from: u32, to: u32, bytes: usize) -> Option<f64> {
        self.cluster
            .transfer(from as usize, to as usize, bytes as f64)
    }

    /// The same device, never choosing `kernel`.
    pub fn without(mut self, kernel: impl Into<String>) -> Self {
        self.barred.push(kernel.into());
        self.changed()
    }

    /// The same device, always choosing `rule` where it applies.
    pub fn favoring(mut self, rule: impl Into<String>) -> Self {
        self.favor = Some(rule.into());
        self.changed()
    }

    pub fn favors(&self, rule: &str) -> bool {
        self.favor.as_deref() == Some(rule)
    }

    /// Seconds for one launch in `zone`. Samples cover zone 0.
    pub fn seconds_in(&self, zone: u32, k: &dyn Kernel, len: usize) -> f64 {
        match zone {
            0 => self.seconds(k, len),
            _ => FIXED + k.guess() * k.work(len),
        }
    }

    /// Estimated seconds for one launch writing `len` elements.
    pub fn seconds(&self, k: &dyn Kernel, len: usize) -> f64 {
        if self.barred.iter().any(|b| b == k.name()) {
            return f64::INFINITY;
        }
        let work = k.work(len);
        let key = (k.name(), k.sizes(), work.to_bits());
        if let Some(seconds) = self.memo.borrow().get(&key) {
            return *seconds;
        }
        let timed = self.timed.get(k.name());
        let near = timed.and_then(|t| estimate(t, &place(&key.1, work), work));
        // With no sample of this kernel, the chip model speaks.
        let modelled = || Some(self.pool.as_ref()?.estimate(&k.load(len)?).seconds);
        let seconds = near.or_else(modelled).unwrap_or(FIXED + k.guess() * work);
        self.memo.borrow_mut().insert(key, seconds);
        seconds
    }

    /// The least seconds the chip needs for this launch's work.
    pub fn bound(&self, k: &dyn Kernel, len: usize) -> Option<f64> {
        Some(self.pool.as_ref()?.bound(&k.load(len)?))
    }

    /// The plan's estimated use of each resource.
    pub fn cost(&self, plan: &Plan) -> Cost {
        let mut total = Cost::default();
        for step in &plan.steps {
            let moved: usize = step.reads.iter().map(|r| r.bytes()).sum();
            // A read in another zone makes the step a move.
            let from = step.reads.first().map(|r| plan.zone(r.buffer));
            total.seconds += match from.filter(|z| *z != step.zone) {
                Some(from) => self.transfer(from, step.zone, step.write.bytes()),
                None => Some(self.seconds_in(step.zone, &*step.kernel, step.write.len)),
            }
            .unwrap_or(f64::INFINITY);
            total.bytes += (moved + step.write.bytes()) as u64;
            total.launches += step.launches().len();
        }
        total
    }

    /// Whether this launch was timed with its current source.
    pub fn known(&self, step: &Step) -> bool {
        let now = Sample::new(step, 0.0);
        let timed = self.timed.get(step.kernel.name());
        let fresh = |s: &Sample| s.same(&now) && s.source == now.source;
        timed.is_some_and(|t| t.iter().any(fresh))
    }

    pub fn samples(&self) -> usize {
        self.timed.values().map(Vec::len).sum()
    }

    /// Adds timed launches. A launch timed before is replaced.
    pub fn fit<'a>(&mut self, samples: impl IntoIterator<Item = (&'a Step, f64)>) {
        self.generation = fresh();
        self.memo.get_mut().clear();
        let mut seen: Vec<(&'static str, Sample, Vec<f64>)> = Vec::new();
        for (step, seconds) in samples {
            let sample = Sample::new(step, seconds);
            match seen.iter_mut().find(|(_, s, _)| s.same(&sample)) {
                Some((_, _, all)) => all.push(seconds),
                None => seen.push((step.kernel.name(), sample, vec![seconds])),
            }
        }
        for (name, mut sample, mut all) in seen {
            // The median shrugs off one slow launch.
            all.sort_by(f64::total_cmp);
            sample.seconds = all[all.len() / 2];
            let timed = self.timed.entry(name.to_string()).or_default();
            timed.retain(|old| !old.same(&sample));
            timed.push(sample);
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), Error> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string(&(VERSION, self))?)
    }

    /// A stored device. Another version's file is invalid data.
    pub fn load(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path)?;
        let (version, _): (u32, serde::de::IgnoredAny) = serde_json::from_str(&text)?;
        if version != VERSION {
            let found = format!("device version {version}");
            return Err(Error::new(ErrorKind::InvalidData, found));
        }
        let (_, mut device): (u32, Device) = serde_json::from_str(&text)?;
        for s in device.timed.values_mut().flatten() {
            s.at = place(&s.sizes, s.work);
        }
        device.pool = device.fit.map(|fit| fit.pool("fitted"));
        device.generation = fresh();
        Ok(device)
    }
}

/// Seconds at `at`: the nearest samples, each scaled to `work`.
fn estimate(timed: &[Sample], at: &[f64], work: f64) -> Option<f64> {
    let mut near = [(f64::INFINITY, 0.0); NEAR];
    for s in timed.iter().filter(|s| s.at.len() == at.len()) {
        let apart = |(a, b): (&f64, &f64)| (a - b) * (a - b);
        let far: f64 = s.at.iter().zip(at).map(apart).sum();
        let scaled = if s.work > 0.0 {
            s.seconds * work / s.work
        } else {
            s.seconds
        };
        // Keeps `near` sorted, nearest first.
        let mut held = (far, scaled);
        for slot in &mut near {
            if held.0 < slot.0 {
                std::mem::swap(slot, &mut held);
            }
        }
    }
    // Far samples fade fast. An exact match outweighs the rest.
    let weigh = |(far, _): &(f64, f64)| 1.0 / (far * far + 1e-12);
    let found = near.iter().filter(|(far, _)| far.is_finite());
    let total: f64 = found.clone().map(weigh).sum();
    let sum: f64 = found.map(|n| weigh(n) * n.1).sum();
    (total > 0.0).then(|| sum / total)
}
