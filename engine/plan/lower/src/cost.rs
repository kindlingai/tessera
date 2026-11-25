//! Launch costs on one device, from timed samples.

use crate::form::Pins;
use crate::kernel::Kernel;
use crate::kernels::{crosses, moved};
use crate::plan::{Plan, Step};
use crate::recipe::{Compute, Knobs, Recipe};
use crate::rule::{Rule, gather, rules_with};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{Error, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};
use tessera_codegen::kept::Layout;
use tessera_engine_settings::{Backend, Pick, Settings};
use tessera_graph::Parts;
use tessera_platform::{Limits, Platform};
use tessera_sim::{Chip, Cluster, Cost, FIXED, Fit, Sample, Timings, ZoneKind, generic, known};

/// Bumped whenever a stored device changes meaning.
const VERSION: u32 = 1;

/// One device's timed launches, by kernel name.
#[derive(Serialize, Deserialize, Clone, Debug, derive_more::PartialEq)]
pub struct Device {
    /// The hardware that ran the samples.
    pub name: String,
    /// The chip's constants, fitted from probes.
    pub fit: Option<Fit>,
    /// A known chip's table, else the fitted model.
    #[serde(skip)]
    #[partial_eq(skip)]
    model: Option<Arc<dyn Chip>>,
    /// The zones plans may span. Zone 0 is this GPU.
    #[serde(default)]
    pub cluster: Cluster,
    /// Zones that each run the plan. Empty means one.
    #[serde(default)]
    pub ranks: Vec<u32>,
    /// Each rank's weight of split axes. Empty means equal.
    #[serde(default)]
    pub parts: Vec<u32>,
    /// The rank this device lowers for.
    #[serde(default)]
    pub rank: u32,
    /// The hardware as asked, with its table row.
    #[serde(default)]
    pub platform: Platform,
    timed: Timings,
    /// Timed launches on a host zone, by kernel name.
    #[serde(default)]
    host: Timings,
    /// A rule to pick wherever it applies, whatever it costs.
    #[serde(skip)]
    favor: Option<String>,
    /// Fail lowering unless `favor` covers at least one step.
    #[serde(skip)]
    require_favored: bool,
    /// Kernels never to choose.
    #[serde(skip)]
    barred: Vec<String>,
    /// Rules made at run time, beside the linked ones.
    #[serde(skip)]
    made: Vec<&'static Rule>,
    /// The form each weight, activation and kept row uses.
    #[serde(skip)]
    pub pins: Pins,
    /// Operand precision for matrix-unit kernels.
    #[serde(skip)]
    pub compute: Compute,
    /// Kernel shape values the recipe set.
    #[serde(skip)]
    pub knobs: Knobs,
    /// Where kept rows lie.
    #[serde(skip, default = "dense")]
    pub kept: &'static dyn Layout,
    /// Shared paged KV row capacity across all slots.
    #[serde(skip)]
    pub kept_rows: Option<usize>,
    /// Model config JSON for runtime rules.
    #[serde(skip)]
    pub config: Option<String>,
    /// Fresh after every change that could alter a plan.
    #[serde(skip)]
    #[partial_eq(skip)]
    generation: u64,
    /// Estimates already made, by kernel and sizes.
    #[serde(skip)]
    #[partial_eq(skip)]
    memo: RefCell<Memo>,
}

/// Source hash for `step`, cached by its printed kernel.
fn source_hash(step: &Step, kernel: &str) -> u64 {
    static HASHES: OnceLock<Mutex<HashMap<(&'static str, String), u64>>> = OnceLock::new();
    let cache = HASHES.get_or_init(Mutex::default);
    let key = (step.kernel.name(), kernel.to_string());
    if let Some(&hash) = cache.lock().expect("source hashes").get(&key) {
        return hash;
    }
    let hash = step.kernel.source().hash();
    cache.lock().expect("source hashes").insert(key, hash);
    hash
}

/// A timed launch of `step`.
fn sample(step: &Step, seconds: f64) -> Sample {
    let (sizes, work) = (step.kernel.sizes(), step.kernel.work(step.write.len));
    let kernel = format!("{:?}", step.kernel);
    let source = source_hash(step, &kernel);
    Sample::new(kernel, source, sizes, work, seconds)
}

/// Seconds by zone kind, kernel name, sizes, work and coldness.
type Memo = HashMap<(ZoneKind, &'static str, Vec<usize>, u64, bool), f64>;

/// Each generation is handed out once per process.
static GENERATIONS: AtomicU64 = AtomicU64::new(1);

fn fresh() -> u64 {
    GENERATIONS.fetch_add(1, Ordering::Relaxed)
}

/// The table for `name`, `fit`, or a scaled model.
fn model(name: &str, fit: Option<Fit>, platform: &Platform) -> Option<Arc<dyn Chip>> {
    if let Some(table) = known(name) {
        return Some(Arc::from(table));
    }
    if let Some(fit) = fit {
        return Some(Arc::new(fit.pool("fitted")) as Arc<dyn Chip>);
    }
    let gb = platform.limits.group_bytes;
    generic(
        &platform.backend,
        name,
        platform.units,
        platform.bandwidth,
        gb,
    )
    .map(Arc::from)
}

fn dense() -> &'static dyn Layout {
    tessera_codegen::kept::DENSE
}

impl Default for Device {
    fn default() -> Self {
        Self {
            name: String::new(),
            fit: None,
            model: None,
            cluster: Cluster::default(),
            ranks: Vec::new(),
            parts: Vec::new(),
            rank: 0,
            platform: Platform::default(),
            timed: Timings::default(),
            host: Timings::default(),
            favor: None,
            require_favored: false,
            barred: Vec::new(),
            made: Vec::new(),
            pins: Pins::default(),
            compute: Compute::default(),
            knobs: Knobs::default(),
            kept: dense(),
            kept_rows: None,
            config: None,
            generation: 0,
            memo: RefCell::default(),
        }
    }
}

/// The platform of the GPU `settings` picks, if it opens.
pub type Ask = fn(&Settings) -> Option<Platform>;

gather! {
    /// Each linked runtime may answer for its GPU.
    #[gather]
    pub static ASKERS: scattered_collect::ScatteredSlice<Ask>;
}

/// The GPU platform a linked runtime reports, if any.
fn asked(settings: &Settings, backend: Backend) -> Option<Platform> {
    if let Some(asked) = ASKERS.iter().find_map(|ask| ask(settings)) {
        return Some(asked);
    }
    static WARNED: Once = Once::new();
    WARNED.call_once(|| {
        eprintln!(
            "tessera: no {} platform to plan for, so rules that need a matrix unit stay off. \
             Link tessera-run, or use Device::stored(runner.platform(), settings)",
            backend.name()
        )
    });
    None
}

impl Device {
    /// The device `open` picks under `settings`, before any samples.
    pub fn guess(settings: &Settings) -> Self {
        // Native kernels run where the zone is a CPU.
        let cluster = match settings.backend {
            Pick::Cpu => Cluster::cpu("cpu"),
            _ => Cluster::default(),
        };
        let device = |platform| Self {
            platform,
            cluster,
            ..Self::default()
        };
        let Some(b) = settings.gpu() else {
            return device(Platform::host(Limits::host(settings.host_lanes)));
        };
        if let Some(platform) = asked(settings, b) {
            return device(Platform::default()).on(platform);
        }
        // The backend name picks which kernel texts fit.
        device(Platform {
            backend: b.name().into(),
            ..Platform::assumed(Limits::assumed())
        })
    }

    /// Where this machine keeps the samples for `platform`.
    pub fn path(settings: &Settings, platform: &Platform) -> Option<PathBuf> {
        let plain = |c: char| if c.is_ascii_alphanumeric() { c } else { '-' };
        let file: String = platform.id().chars().map(plain).collect();
        Some(settings.costs.as_ref()?.join(format!("{file}.json")))
    }

    /// The samples stored for `platform`, or a guess.
    pub fn stored(platform: Platform, settings: &Settings) -> Result<Self, Error> {
        let loaded = Self::path(settings, &platform).map(|p| Self::load(&p));
        let mut device = match loaded {
            Some(Err(e)) if e.kind() == ErrorKind::NotFound => Self::guess(settings),
            Some(loaded) => loaded?,
            None => Self::guess(settings),
        };
        // A runner runs one zone.
        device.cluster.zones.truncate(1);
        device.cluster.links.clear();
        Ok(device.on(platform))
    }

    /// The same device, choosing as `recipe` says.
    pub fn with(mut self, recipe: &Recipe) -> Self {
        self.pins = recipe.pins.clone();
        self.compute = recipe.compute;
        self.knobs = recipe.shape.clone();
        self.favor = recipe.favor.clone();
        self.require_favored = recipe.require_favored;
        self.barred = recipe.barred.clone();
        self.made = recipe.rules.clone();
        self.kept = recipe.kept;
        self.kept_rows = recipe.kept_rows;
        self.config = recipe.config.clone();
        self.changed()
    }

    /// The same device, passing `config` JSON to runtime rules.
    pub fn configured(mut self, config: impl Into<String>) -> Self {
        self.config = Some(config.into());
        self.changed()
    }

    /// The same device, on this hardware.
    pub fn on(mut self, platform: Platform) -> Self {
        // Without stored samples, the hardware names the device.
        if self.name.is_empty() && !platform.name.is_empty() {
            self.name = format!("{} {}", platform.backend, platform.name);
        }
        self.platform = platform;
        self.model = model(&self.name, self.fit, &self.platform);
        self.changed()
    }

    /// The same device, with a chip model for unmeasured kernels.
    pub fn fitted(mut self, fit: Fit) -> Self {
        self.fit = Some(fit);
        self.model = model(&self.name, self.fit, &self.platform);
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

    /// The same device, with storage forms as `pins` say.
    pub fn pinned(mut self, pins: Pins) -> Self {
        self.pins = pins;
        self.changed()
    }

    /// The same device, using `compute` matrix-unit precision.
    pub fn with_compute(mut self, compute: Compute) -> Self {
        self.compute = compute;
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

    /// The same device, the plan copied into each zone.
    pub fn ranked(mut self, zones: Vec<u32>) -> Self {
        self.ranks = zones;
        self.changed()
    }

    /// Copies of the plan that run together.
    pub fn ranks(&self) -> usize {
        self.ranks.len().max(1)
    }

    /// The same device, ranks weighted unevenly.
    pub fn uneven(mut self, parts: Vec<u32>) -> Self {
        self.parts = parts;
        self.changed()
    }

    /// The same device, lowering for rank `rank`.
    pub fn as_rank(mut self, rank: u32) -> Self {
        self.rank = rank;
        self.changed()
    }

    /// How split axes divide among the ranks.
    pub fn division(&self) -> Parts {
        match self.parts.len() == self.ranks() {
            true => Parts(self.parts.clone()),
            false => Parts::even(self.ranks()),
        }
    }

    /// Seconds the ranks take to sum `bytes`, if linked.
    pub fn all_reduce(&self, bytes: usize) -> Option<f64> {
        let zones: Vec<usize> = self.ranks.iter().map(|z| *z as usize).collect();
        self.cluster.all_reduce(&zones, bytes as f64)
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

    /// Fail lowering unless the favored rule covers a step.
    pub fn requiring_favored(mut self, on: bool) -> Self {
        self.require_favored = on;
        self.changed()
    }

    /// The same device, also asking `rule`.
    pub fn adding(mut self, rule: &'static Rule) -> Self {
        self.made.push(rule);
        self.changed()
    }

    /// Linked and made rules, by name.
    pub fn rules(&self) -> Vec<&'static Rule> {
        rules_with(&self.made)
    }

    pub fn favors(&self, rule: &str) -> bool {
        self.favor.as_deref() == Some(rule)
    }

    /// The favored rule name, if one is set.
    pub fn favored(&self) -> Option<&str> {
        self.favor.as_deref()
    }

    /// The favored rule when lowering requires it to cover steps.
    pub fn required_favor(&self) -> Option<&str> {
        self.require_favored
            .then_some(self.favor.as_deref())
            .flatten()
    }

    /// Seconds for one launch in `zone`. Cold ones stream weights.
    pub fn seconds_in(&self, zone: u32, k: &dyn Kernel, len: usize, cold: bool) -> f64 {
        match (zone, self.cluster.kind(zone as usize)) {
            (0, kind) | (_, kind @ ZoneKind::Cpu) => self.estimate(kind, k, len, cold),
            _ => FIXED + k.guess() * k.work(len),
        }
    }

    /// Estimated seconds for one warm launch writing `len` elements.
    pub fn seconds(&self, k: &dyn Kernel, len: usize) -> f64 {
        self.seconds_in(0, k, len, false)
    }

    /// Bytes the last cache holds between launches, if known.
    pub fn cache(&self) -> Option<f64> {
        let constants = self.model.as_ref()?.constants();
        let cache = constants.iter().find(|(name, _)| *name == "cache");
        cache.map(|(_, c)| c.value)
    }

    /// Whether `weights` bytes are too many for the cache.
    pub fn streams(&self, weights: usize) -> bool {
        self.cache().is_some_and(|cache| weights as f64 > cache)
    }

    /// Seconds from the samples of zones of `kind`.
    fn estimate(&self, kind: ZoneKind, k: &dyn Kernel, len: usize, cold: bool) -> f64 {
        if self.barred.iter().any(|b| b == k.name()) {
            return f64::INFINITY;
        }
        let work = k.work(len);
        let key = (kind, k.name(), k.sizes(), work.to_bits(), cold);
        if let Some(seconds) = self.memo.borrow().get(&key) {
            return *seconds;
        }
        // A host zone beside a GPU has its own samples.
        let own = kind == self.cluster.kind(0);
        let timed = if own { &self.timed } else { &self.host };
        let near = timed.estimate(k.name(), &key.2, work, cold);
        // With no sample of this kernel, the chip model speaks.
        let chip = self.model.as_ref().filter(|_| own);
        let modelled = || Some(chip?.estimate(&k.load(len)?).seconds);
        let seconds = near.or_else(modelled).unwrap_or(FIXED + k.guess() * work);
        self.memo.borrow_mut().insert(key, seconds);
        seconds
    }

    /// The least seconds the chip needs for this launch's work.
    pub fn bound(&self, k: &dyn Kernel, len: usize) -> Option<f64> {
        Some(self.model.as_ref()?.bound(&k.load(len)?))
    }

    /// Seconds for one step of `plan`, if its zones link.
    pub fn step_seconds(&self, plan: &Plan, step: &Step) -> Option<f64> {
        // A read in another zone makes the step a move.
        let from = step.reads.first().map(|r| plan.zone(r.buffer));
        let cold = plan.reads_weight(step) && self.streams(plan.weight_bytes());
        match from.filter(|z| *z != step.zone) {
            Some(from) => self.transfer(from, step.zone, step.write.bytes()),
            None if crosses(step.kernel.name()) => {
                self.all_reduce(moved(&*step.kernel, step.write.len, step.write.bytes()))
            }
            None => Some(self.seconds_in(step.zone, &*step.kernel, step.write.len, cold)),
        }
    }

    /// The plan's estimated use of each resource.
    pub fn cost(&self, plan: &Plan) -> Cost {
        let mut total = Cost::default();
        for step in &plan.steps {
            let moved: usize = step.reads.iter().map(|r| r.bytes()).sum();
            let written: usize =
                step.write.bytes() + step.also.iter().map(|w| w.bytes()).sum::<usize>();
            total.seconds += self.step_seconds(plan, step).unwrap_or(f64::INFINITY);
            total.bytes += (moved + written) as u64;
            total.launches += step.launches().len();
        }
        total
    }

    /// Whether this launch was timed with its current source.
    pub fn known(&self, step: &Step) -> bool {
        self.timed.known(step.kernel.name(), &sample(step, 0.0))
    }

    /// Whether this launch has a cold time from its source.
    pub fn known_cold(&self, step: &Step) -> bool {
        self.timed
            .known_cold(step.kernel.name(), &sample(step, 0.0))
    }

    pub fn samples(&self) -> usize {
        self.timed.len()
    }

    /// The same device, with `host`'s samples for its host zone.
    pub fn hosted(mut self, host: &Device) -> Self {
        self.host = host.timed.clone();
        self.changed()
    }

    pub fn host_samples(&self) -> usize {
        self.host.len()
    }

    /// Adds timed launches. A launch timed before is replaced.
    pub fn fit<'a>(&mut self, samples: impl IntoIterator<Item = (&'a Step, f64)>) {
        self.generation = fresh();
        self.memo.get_mut().clear();
        let timed = samples
            .into_iter()
            .map(|(s, seconds)| (s.kernel.name(), sample(s, seconds)));
        self.timed.record(timed);
    }

    /// Adds cold times for launches already sampled warm.
    pub fn fit_cold<'a>(&mut self, samples: impl IntoIterator<Item = (&'a Step, f64)>) {
        self.generation = fresh();
        self.memo.get_mut().clear();
        let timed = samples
            .into_iter()
            .map(|(s, seconds)| (s.kernel.name(), sample(s, seconds)));
        self.timed.record_cold(timed);
    }

    /// Per kernel: cold samples, and the middle cold over warm.
    pub fn colder(&self) -> Vec<(&str, usize, f64)> {
        self.timed.colder()
    }

    /// The share of sampled seconds whose spread passes `above`.
    pub fn noisy(&self, above: f64) -> f64 {
        self.timed.noisy(above)
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
        device.timed.restore();
        device.host.restore();
        // Files from before residency take the table's constants.
        device.fit = device.fit.map(|fit| fit.filled(&device.name));
        device.model = model(&device.name, device.fit, &device.platform);
        device.generation = fresh();
        Ok(device)
    }
}
