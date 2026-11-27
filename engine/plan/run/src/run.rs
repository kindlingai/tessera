//! Work every backend's plan runner does.

use crate::Error;
use crate::held::Held;
use crate::profile::Profile;
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::OnceLock;
use tessera_codegen::Precision;
use tessera_engine_settings::{Memory, Reads};
use tessera_gpukit::{Limits, Platform};
use tessera_graph::DimName;
use tessera_lower::form::{F32_FORM, Load};
use tessera_lower::{
    Arg, Extent, Filed, Launch, Launching, Loaded, Plan, Shape, Share, Slot, Weight,
};
use tessera_mem::{Tensor, View, copy};

/// State every runner keeps: a plan, its buffers and sizes.
#[derive(Default)]
pub struct Base {
    pub(crate) plan: Plan,
    pub(crate) held: Held,
    /// Runtime dimensions, as last bound.
    pub(crate) bound: HashMap<DimName, u32>,
    /// Each step's launches before binding, built once.
    launches: Vec<OnceLock<Vec<Launch>>>,
    /// The most bytes a plan may use. `None` checks nothing.
    pub(crate) budget: Option<u64>,
    /// Whether weights past the budget may page from the host.
    pub(crate) streams: bool,
    /// Cold experts copy to VRAM ahead of their reader.
    pub(crate) stages: bool,
    /// Seconds per step kernel, while profiling.
    pub(crate) profile: Option<Profile>,
    pub(crate) reads: Reads,
}

impl Base {
    /// Empty, with the plan budget `memory` gives on `platform`.
    pub(crate) fn new(memory: Memory, platform: &Platform) -> Self {
        // A tenth of the device stays spare.
        let budget = match memory {
            Memory::Off => None,
            Memory::Bytes(bytes) => Some(bytes),
            Memory::Device if platform.memory == 0 => None,
            Memory::Device => Some(platform.memory - platform.memory / 10),
        };
        Self {
            budget,
            ..Self::default()
        }
    }

    /// Extends a device-derived budget with external carveout bytes.
    pub(crate) fn extend_budget(&mut self, extra: u64) {
        if let Some(budget) = &mut self.budget {
            *budget = budget.saturating_add(extra);
        }
    }

    /// Takes `plan` as the one to run.
    pub(crate) fn adopt(&mut self, plan: &Plan) {
        self.plan = plan.clone();
        self.launches = (0..plan.steps.len()).map(|_| OnceLock::new()).collect();
    }

    /// Swaps in a saved plan and its cached launches.
    pub(crate) fn swap(&mut self, plan: &mut Plan, launches: &mut Vec<OnceLock<Vec<Launch>>>) {
        std::mem::swap(&mut self.plan, plan);
        std::mem::swap(&mut self.launches, launches);
    }

    /// Whether `plan` is new here. Plans over several zones fail.
    pub(crate) fn fresh(&self, plan: &Plan) -> Result<bool, Error> {
        self.fresh_with(plan, 0)
    }

    /// Checks `plan`'s zone and memory budget with `extra` bytes.
    pub(crate) fn validate(&self, plan: &Plan, extra: u64) -> Result<(), Error> {
        if plan.zoned() {
            return Err(Error::Zones);
        }
        let planned: u64 = plan.buffers.iter().map(|b| *b as u64).sum::<u64>() + extra;
        if let Some(budget) = self.budget
            && planned > budget
        {
            return Err(Error::OverBudget { planned, budget });
        }
        Ok(())
    }

    /// As `fresh`, counting `extra` bytes held beside `plan`'s buffers.
    pub(crate) fn fresh_with(&self, plan: &Plan, extra: u64) -> Result<bool, Error> {
        if plan.id != 0 && plan.id == self.plan.id {
            return Ok(false);
        }
        self.validate(plan, extra)?;
        Ok(true)
    }

    /// Each launch of a step, its runtime sizes filled in.
    pub(crate) fn launches(&self, step: usize) -> Result<Vec<Launch>, Error> {
        let size = |name| self.bound.get(&name).copied();
        let bound = |l: &Launch| {
            l.clone()
                .bound(&size)
                .ok_or(Error::from(tessera_lower::Error::Unbound))
        };
        let unbound = self.launches[step].get_or_init(|| self.plan.steps[step].launches());
        unbound.iter().map(bound).collect()
    }

    /// Runtime dimensions referenced by `steps`, sorted and unique.
    pub(crate) fn step_dims(&self, steps: Range<usize>) -> Vec<DimName> {
        let mut dims = Vec::new();
        for step in steps {
            let unbound = self.launches[step].get_or_init(|| self.plan.steps[step].launches());
            for launch in unbound {
                for arg in &launch.args {
                    if let Arg::Size(Extent::Dim { name, .. })
                    | Arg::Buffer(_, Extent::Dim { name, .. }) = *arg
                    {
                        dims.push(name);
                    }
                }
                if let Shape::Runtime(Extent::Dim { name, .. }) = launch.shape {
                    dims.push(name);
                }
            }
        }
        dims.sort_unstable();
        dims.dedup();
        dims
    }
}

/// Groups and threads for a line of `count` threads.
pub(crate) fn line(count: usize) -> (usize, usize) {
    let threads = count.clamp(1, 256);
    (count.div_ceil(threads), threads)
}

/// The effect of a kernel file's last change.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// Its kernel runs in place of the planned one.
    Loaded,
    /// The planned kernel runs again.
    Planned,
    /// It did not compile. The kernel before it still runs.
    Failed(String),
}

/// A kernel file whose change altered what runs.
#[derive(Clone, Debug, PartialEq)]
pub struct Reload {
    pub path: PathBuf,
    pub outcome: Outcome,
}

impl fmt::Display for Reload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let path = self.path.display();
        match &self.outcome {
            Outcome::Loaded => write!(f, "{path}: loaded"),
            Outcome::Planned => write!(f, "{path}: planned kernel"),
            Outcome::Failed(why) => write!(f, "{path}: {why}"),
        }
    }
}

/// A plan with its buffers allocated and kernels compiled.
pub trait Run {
    fn base(&self) -> &Base;
    fn base_mut(&mut self) -> &mut Base;
    fn plan(&self) -> &Plan {
        &self.base().plan
    }
    fn held(&self) -> &Held {
        &self.base().held
    }
    fn held_mut(&mut self) -> &mut Held {
        &mut self.base_mut().held
    }

    /// Paged kept-slot metadata for `page` on this runner's plan.
    fn kept_page_meta(&self, page: usize) -> Option<Vec<(u32, u32, usize, &'static str)>> {
        Held::kept_page_meta(self.plan(), page)
    }

    /// Downloads `phys` physical KV pages packed into `dst`.
    fn download_kept_pages(
        &mut self,
        phys: &[u32],
        page: usize,
        dst: &mut [u8],
    ) -> Result<bool, Error> {
        self.check_ready()?;
        if phys.is_empty() {
            return Ok(true);
        }
        let page_bytes = dst.len() / phys.len();
        if page_bytes * phys.len() != dst.len() {
            return Ok(false);
        }
        let plan = self.plan().clone();
        for (i, &p) in phys.iter().enumerate() {
            let slice = &mut dst[i * page_bytes..(i + 1) * page_bytes];
            if !self.held().download_page(&plan, p, page, slice)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Uploads `phys` physical KV pages packed in `src`.
    fn upload_kept_pages(&mut self, phys: &[u32], page: usize, src: &[u8]) -> Result<bool, Error> {
        self.check_ready()?;
        if phys.is_empty() {
            return Ok(true);
        }
        let page_bytes = src.len() / phys.len();
        if page_bytes * phys.len() != src.len() {
            return Ok(false);
        }
        let plan = self.plan().clone();
        for (i, &p) in phys.iter().enumerate() {
            let slice = &src[i * page_bytes..(i + 1) * page_bytes];
            if !self.held_mut().upload_page(&plan, p, page, slice)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Opens a pinned device-mapped host KV pool of `pages`.
    fn open_host_kv(&mut self, _pages: usize, _page: usize) -> Result<bool, Error> {
        Ok(false)
    }

    /// Promotes or demotes `(vram_phys, host_phys)` sparse KV page pairs.
    fn move_kv_pages(&mut self, _promote: bool, _pairs: &[(u32, u32)]) -> Result<bool, Error> {
        Ok(false)
    }

    /// Takes up another plan. Weights and kept rows stay loaded.
    fn replan(&mut self, plan: &Plan) -> Result<(), Error>;
    /// Gives a runtime dimension its value for the next launches.
    fn bind(&mut self, name: DimName, value: u32) {
        self.base_mut().bound.insert(name, value);
    }
    /// Runs these steps in order and waits for them.
    fn submit(&mut self, steps: Range<usize>) -> Result<(), Error>;
    /// Queues these steps in order without waiting when supported.
    fn queue(&mut self, steps: Range<usize>) -> Result<(), Error> {
        self.submit(steps)
    }
    /// Rereads kernel files that changed. Returns what changed.
    fn refresh(&mut self) -> Result<Vec<Reload>, Error> {
        Ok(Vec::new())
    }
    /// Writes each planned kernel's text where no file holds it.
    fn dump(&self) -> Result<Vec<PathBuf>, Error> {
        Ok(Vec::new())
    }
    /// Times `step`'s `TS_MARK()` gaps. `None` on Metal and hosts.
    fn marks(&mut self, _step: usize) -> Result<Option<crate::Marks>, Error> {
        Ok(None)
    }
    /// Compiled resource counts and occupancy for `step`, if known.
    fn step_attrs(&self, _step: usize) -> Option<tessera_gpukit::device::KernelAttrs> {
        None
    }
    /// Runs these steps as `submit` does, timed by the device.
    fn clocked(&mut self, steps: Range<usize>) -> Result<Option<f64>, Error> {
        self.submit(steps).map(|_| None)
    }
    /// The slot's device address, where buffers have one.
    fn address(&self, _slot: Slot) -> Option<u64> {
        None
    }
    /// Copies `bytes` from device address `src_addr` into `dst`.
    fn copy_from(
        &mut self,
        _dst: Slot,
        _dst_byte_off: usize,
        _src_addr: u64,
        _bytes: usize,
    ) -> Result<(), Error> {
        Err(Error::Unaddressed(0))
    }
    /// How to launch from now on, where the device can.
    fn launching(&mut self, _: Launching) {}
    /// Runs resident steps through `grid` from now on.
    fn persist(&mut self, grid: Box<dyn crate::Grid>) -> Result<(), Error> {
        drop(grid);
        Err(Error::NoGrids)
    }
    /// Starts or stops timing each step.
    fn profiling(&mut self, on: bool) {
        self.base_mut().profile = on.then(Profile::default);
    }
    /// The seconds gathered since the last take.
    fn take_profile(&mut self) -> Option<Profile> {
        self.base_mut().profile.as_mut().map(std::mem::take)
    }
    /// Adds `seconds` to `name`, while profiling.
    fn note(&mut self, name: &str, seconds: f64) {
        if let Some(p) = self.base_mut().profile.as_mut() {
            p.add(name, seconds);
        }
    }
    /// Adds `seconds` to step `i`'s row, while profiling.
    fn note_step(&mut self, i: usize, seconds: f64) {
        let base = self.base_mut();
        if let Some(p) = base.profile.as_mut() {
            p.step(&base.plan.steps[i], seconds);
        }
    }
    /// Runs one step `times` over. Returns seconds per run.
    fn repeat(&mut self, step: usize, times: usize) -> Result<f64, Error>;
    /// The backend and the hardware it drives.
    fn device(&self) -> String;
    /// Steps running compiled host code, out of steps with trees.
    fn compiled(&self) -> (usize, usize) {
        (0, 0)
    }
    /// The hardware's limits for one group of threads.
    fn limits(&self) -> Limits {
        self.platform().limits
    }
    /// The hardware as asked, with its table row.
    fn platform(&self) -> Platform;
    /// How the hardware's tiles round.
    fn precision(&self) -> Precision;

    /// This rank's part of weight `weight`, if it is split.
    fn share<'a>(&self, weight: u32, data: View<'a>) -> Result<View<'a>, Error> {
        let loaded = self.plan().weights.iter().find(|l| l.weight == weight);
        let Some(share) = loaded.and_then(|l| l.share) else {
            return Ok(data);
        };
        Ok(data.slice(share.axis as usize, share.first, share.count))
    }

    /// Uploads packed bytes for `weight` into `slot`.
    fn upload_weight(&mut self, _weight: u32, slot: Slot, bytes: &[u8]) -> Result<(), Error> {
        self.check_ready()?;
        self.held_mut().upload(slot, bytes)
    }

    /// Uploads a weight as stored, widened where a slot differs.
    fn load_weight(&mut self, weight: u32, w: &Weight) -> Result<(), Error> {
        self.check_ready()?;
        let (form, bytes) = match w {
            Weight::Floats(t) => return self.load(weight, t.view()),
            Weight::Packed(form, bytes) => (form, bytes),
            Weight::Filed(form, file) => {
                if filed(self, weight, *form, file.as_ref())? {
                    return Ok(());
                }
                let reads = self.base().reads;
                let read = w.held(reads).map_err(|e| Error::WeightRead(weight, e))?;
                return self.load_weight(weight, &read.expect("a filed weight"));
            }
        };
        let loaded: Vec<Loaded> = (self.plan().weights.iter())
            .filter(|l| l.weight == weight)
            .copied()
            .collect();
        if loaded.iter().any(|l| l.part.is_some()) {
            return parts(self, weight, *form, bytes, &loaded);
        }
        // A file stored whole loads into other forms decoded.
        let whole = form.split(bytes).is_some();
        // Bytes a reader already cut fill the slot.
        let cut = loaded
            .first()
            .is_some_and(|l| form.bytes(l.slot.len) == bytes.len());
        let part = match loaded.first().and_then(|l| l.share).filter(|_| !cut) {
            None => None,
            Some(share) => match share.of(*form, bytes) {
                Some(part) => Some(part),
                // Decoding into the same form would round it twice.
                None if loaded.iter().any(|l| l.slot.form.name() == form.name()) => {
                    let (form, why) = (form.name(), "a share cuts its blocks");
                    return Err(Error::WeightParts { weight, form, why });
                }
                None => return unpacked(self, weight, *form, bytes, share),
            },
        };
        let bytes = part.as_deref().unwrap_or(bytes);
        for l in loaded {
            let slot = l.slot;
            let same = slot.form.name() == form.name();
            if !same && !slot.form.packs() {
                let (form, into) = (form.name(), slot.form.name());
                return Err(Error::WeightInto { weight, form, into });
            }
            if !whole && form.bytes(slot.len) != bytes.len() {
                let form = form.name();
                return Err(Error::WeightSize { weight, form });
            }
            if same {
                self.upload_weight(weight, slot, bytes)?;
                continue;
            }
            let floats = form.unpack_all(bytes);
            self.write(slot, Tensor::f32(vec![slot.len], floats).view())?;
        }
        Ok(())
    }

    /// Uploads one weight already in its form's bytes.
    fn load_packed(&mut self, weight: u32, bytes: &[u8]) -> Result<(), Error> {
        self.check_ready()?;
        let slots: Vec<Slot> = (self.plan().weights.iter())
            .filter(|l| l.weight == weight)
            .map(|l| l.slot)
            .collect();
        for slot in slots {
            if slot.form.bytes(slot.len) != bytes.len() {
                return Err(Error::WeightSize {
                    weight,
                    form: slot.form.name(),
                });
            }
            self.upload_weight(weight, slot, bytes)?;
        }
        Ok(())
    }

    /// Fails when an incomplete replan left this runner inconsistent.
    fn check_ready(&self) -> Result<(), Error> {
        Ok(())
    }

    /// A plain slot copies. Any other packs through its form.
    fn write(&mut self, slot: Slot, src: View) -> Result<(), Error> {
        self.check_ready()?;
        let plain = slot.form.block() == 1
            && src.dtype() == slot.form.dtype()
            && slot.form.bytes(1) == src.dtype().bytes();
        if plain {
            return Ok(copy(src, &mut self.held_mut().view_mut(slot))?);
        }
        let packed = slot.form.pack(&src.to_f32()?);
        self.held_mut().upload(slot, &packed)
    }

    /// The slot's values on the host, as one row.
    fn read(&self, slot: Slot) -> Result<Tensor, Error> {
        self.check_ready()?;
        if !slot.form.narrow() {
            return Ok(self.held().view(slot).to_tensor()?);
        }
        let bytes = self.held().bytes(slot)?;
        Ok(Tensor::f32(vec![slot.len], slot.form.unpack_all(&bytes)))
    }

    fn ids(&self, slot: Slot) -> Result<Vec<u32>, Error> {
        self.check_ready()?;
        Ok(self.held().view(slot).to_u32()?)
    }

    fn bind_all(&mut self, sizes: &[(DimName, u32)]) {
        for (name, value) in sizes {
            self.bind(*name, *value);
        }
    }

    /// Uploads one graph weight, or this rank's slice.
    fn load(&mut self, weight: u32, data: View) -> Result<(), Error> {
        self.check_ready()?;
        let split = |l: &Loaded| l.weight == weight && l.part.is_some();
        if self.plan().weights.iter().any(split) {
            let (form, why) = ("floats", "a split weight takes its file");
            return Err(Error::WeightParts { weight, form, why });
        }
        let data = self.share(weight, data)?;
        let slots: Vec<Slot> = (self.plan().weights.iter())
            .filter(|l| l.weight == weight)
            .map(|l| l.slot)
            .collect();
        for slot in slots {
            if slot.len != data.len() {
                return Err(Error::WeightLen(weight));
            }
            self.write(slot, data.clone())?;
        }
        Ok(())
    }

    fn feed(&mut self, inputs: &[View]) -> Result<(), Error> {
        self.check_ready()?;
        let slots = self.plan().inputs.clone();
        if inputs.len() != slots.len() {
            return Err(Error::InputCount);
        }
        for (at, (slot, input)) in slots.into_iter().zip(inputs).enumerate() {
            if input.len() != slot.len || input.dtype() != slot.form.dtype() {
                return Err(Error::InputShape(at));
            }
            self.write(slot, input.clone())?;
        }
        Ok(())
    }

    /// Runs every step as one submission. Outputs come back shaped.
    fn run(&mut self, inputs: &[View]) -> Result<Vec<Tensor>, Error> {
        self.feed(inputs)?;
        self.submit(0..self.plan().steps.len())?;
        let outputs = self.plan().outputs.iter();
        let shaped = |o: &tessera_lower::Output| Ok(self.read(o.slot)?.reshaped(o.shape.clone()));
        outputs.map(shaped).collect()
    }

    /// Runs step by step. Returns each step's seconds.
    fn run_timed(&mut self, inputs: &[View]) -> Result<Vec<f64>, Error> {
        self.feed(inputs)?;
        let steps = 0..self.plan().steps.len();
        steps.map(|i| self.repeat(i, 1)).collect()
    }
}

/// Reads a file's bytes into slots of its form.
fn filed<R: Run + ?Sized>(
    r: &mut R,
    weight: u32,
    form: &dyn Load,
    file: &dyn Filed,
) -> Result<bool, Error> {
    let loaded: Vec<Loaded> = (r.plan().weights.iter())
        .filter(|l| l.weight == weight)
        .copied()
        .collect();
    let plain = |l: &Loaded| {
        let slot = l.slot;
        let whole = l.part.is_none() && l.share.is_none();
        whole && slot.form.name() == form.name() && slot.bytes() == file.bytes()
    };
    let Some((first, rest)) = loaded.split_first() else {
        return Ok(true);
    };
    if !loaded.iter().all(plain) {
        return Ok(false);
    }
    let reads = r.base().reads;
    let Some(into) = r.held_mut().mapped_slot(first.slot) else {
        return Ok(false);
    };
    file.read(into, reads)
        .map_err(|e| Error::WeightRead(weight, e))?;
    if rest.is_empty() {
        return Ok(true);
    }
    let bytes = r.held().bytes(first.slot)?;
    for l in rest {
        r.held_mut().upload(l.slot, &bytes)?;
    }
    Ok(true)
}

/// A split the form cannot cut, decoded before the cut.
fn unpacked<R: Run + ?Sized>(
    r: &mut R,
    weight: u32,
    form: &dyn Load,
    bytes: &[u8],
    share: Share,
) -> Result<(), Error> {
    let all = form.unpack_all(bytes);
    let outer = all.len() / (share.whole * share.inner).max(1);
    let floats = Tensor::f32(vec![outer, share.whole, share.inner], all);
    let part = floats.view().slice(1, share.first, share.count);
    let slots: Vec<Slot> = (r.plan().weights.iter())
        .filter(|l| l.weight == weight)
        .map(|l| l.slot)
        .collect();
    for slot in slots {
        if !slot.form.packs() {
            let (form, axis) = (form.name(), share.axis);
            return Err(Error::WeightSplit { weight, form, axis });
        }
        r.write(slot, part.clone())?;
    }
    Ok(())
}

/// A file stored whole, split into its parts' slots.
fn parts<R: Run + ?Sized>(
    r: &mut R,
    weight: u32,
    form: &'static dyn Load,
    bytes: &[u8],
    loaded: &[Loaded],
) -> Result<(), Error> {
    let what = |why: &'static str| Error::WeightParts {
        weight,
        form: form.name(),
        why,
    };
    let split = form.split(bytes).ok_or_else(|| what("no parts"))?;
    let codes = form.codes().ok_or_else(|| what("no codes"))?;
    for l in loaded {
        let at = l.part.ok_or_else(|| what("a whole slot beside parts"))? as usize;
        let into: &dyn Load = if at == 0 { codes } else { F32_FORM };
        let data = match l.share {
            Some(share) => share
                .of(into, &split[at])
                .ok_or_else(|| what("an uneven cut"))?,
            None => split[at].clone(),
        };
        if into.bytes(l.slot.len) != data.len() || l.slot.form.name() != into.name() {
            let form = into.name();
            return Err(Error::WeightSize { weight, form });
        }
        r.held_mut().upload(l.slot, &data)?;
    }
    Ok(())
}
