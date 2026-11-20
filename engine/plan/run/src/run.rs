//! What every backend's plan runner does.

use crate::Error;
use crate::held::Held;
use crate::profile::Profile;
use sensai_codegen::Precision;
use sensai_engine_settings::Memory;
use sensai_gpukit::{Limits, Platform};
use sensai_graph::DimName;
use sensai_lower::form::{F32_FORM, Load};
use sensai_lower::{Arg, Launch, Launching, Loaded, Plan, Shape, Share, Slot, Weight};
use sensai_mem::{Tensor, View, copy};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::OnceLock;

/// What every runner keeps: a plan, its buffers and sizes.
#[derive(Default)]
pub struct Base {
    pub(crate) plan: Plan,
    pub(crate) held: Held,
    /// Runtime dimensions, as last bound.
    pub(crate) bound: HashMap<DimName, u32>,
    /// Each step's launches before binding, built once.
    launches: Vec<OnceLock<Vec<Launch>>>,
    /// Bytes a plan may hold. `None` checks nothing.
    pub(crate) budget: Option<u64>,
    /// Seconds per step kernel, while profiling.
    pub(crate) profile: Option<Profile>,
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

    /// Takes `plan` as the one to run.
    pub(crate) fn adopt(&mut self, plan: &Plan) {
        self.plan = plan.clone();
        self.launches = (0..plan.steps.len()).map(|_| OnceLock::new()).collect();
    }

    /// Whether `plan` is new here. Plans over several zones fail.
    pub(crate) fn fresh(&self, plan: &Plan) -> Result<bool, Error> {
        if plan.id != 0 && plan.id == self.plan.id {
            return Ok(false);
        }
        if plan.zoned() {
            return Err(Error::Zones);
        }
        let planned: u64 = plan.buffers.iter().map(|b| *b as u64).sum();
        if let Some(budget) = self.budget
            && planned > budget
        {
            return Err(Error::OverBudget { planned, budget });
        }
        Ok(true)
    }

    /// Each launch of a step, its runtime sizes filled in.
    pub(crate) fn launches(&self, step: usize) -> Result<Vec<Launch>, Error> {
        let size = |name| self.bound.get(&name).copied();
        let bound = |l: &Launch| {
            l.clone()
                .bound(&size)
                .ok_or(Error::from(sensai_lower::Error::Unbound))
        };
        let unbound = self.launches[step].get_or_init(|| self.plan.steps[step].launches());
        unbound.iter().map(bound).collect()
    }
}

/// Groups and threads for a line of `count` threads.
pub(crate) fn line(count: usize) -> (usize, usize) {
    let threads = count.clamp(1, 256);
    (count.div_ceil(threads), threads)
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
    /// Takes up another plan. Weights and kept rows stay loaded.
    fn replan(&mut self, plan: &Plan) -> Result<(), Error>;
    /// Gives a runtime dimension its value for the next launches.
    fn bind(&mut self, name: DimName, value: u32) {
        self.base_mut().bound.insert(name, value);
    }
    /// Runs these steps in order and waits for them.
    fn submit(&mut self, steps: Range<usize>) -> Result<(), Error>;
    /// Runs these steps as `submit` does, timed by the device.
    fn clocked(&mut self, steps: Range<usize>) -> Result<Option<f64>, Error> {
        self.submit(steps).map(|_| None)
    }
    /// Whether `launch_fixed` runs here.
    fn fixed(&self) -> bool {
        false
    }
    /// Runs a hand-written kernel once and waits for it.
    fn launch_fixed(
        &self,
        _source: &'static str,
        _entry: &'static str,
        _args: &[Arg],
        _shape: Shape,
    ) -> Result<(), Error> {
        Err(Error::FixedKernel)
    }
    /// How to launch from now on, where the device can.
    fn launching(&mut self, _: Launching) {}
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
    /// Runs one step `times` over. Returns seconds per run.
    fn repeat(&mut self, step: usize, times: usize) -> Result<f64, Error>;
    /// The backend and the hardware it drives.
    fn device(&self) -> String;
    /// Steps running compiled host code, out of steps with trees.
    fn compiled(&self) -> (usize, usize) {
        (0, 0)
    }
    /// What the hardware gives one group of threads.
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

    /// Uploads a weight as stored, widened where a slot differs.
    fn load_weight(&mut self, weight: u32, w: &Weight) -> Result<(), Error> {
        let Weight::Packed(form, bytes) = w else {
            let Weight::Floats(t) = w else { unreachable!() };
            return self.load(weight, t.view());
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
                self.held_mut().upload(slot, bytes)?;
                continue;
            }
            let floats = form.unpack_all(bytes);
            self.write(slot, Tensor::f32(vec![slot.len], floats).view())?;
        }
        Ok(())
    }

    /// Uploads one weight already in its form's bytes.
    fn load_packed(&mut self, weight: u32, bytes: &[u8]) -> Result<(), Error> {
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
            self.held_mut().upload(slot, bytes)?;
        }
        Ok(())
    }

    /// A plain slot copies. Any other packs through its form.
    fn write(&mut self, slot: Slot, src: View) -> Result<(), Error> {
        let plain = slot.form.block() == 1 && src.dtype() == slot.form.dtype();
        if plain {
            return Ok(copy(src, &mut self.held_mut().view_mut(slot))?);
        }
        let packed = slot.form.pack(&src.to_f32()?);
        self.held_mut().upload(slot, &packed)
    }

    /// The slot's values on the host, as one row.
    fn read(&self, slot: Slot) -> Result<Tensor, Error> {
        Ok(self.held().view(slot).to_tensor()?)
    }

    fn ids(&self, slot: Slot) -> Result<Vec<u32>, Error> {
        Ok(self.held().view(slot).to_u32()?)
    }

    fn bind_all(&mut self, sizes: &[(DimName, u32)]) {
        for (name, value) in sizes {
            self.bind(*name, *value);
        }
    }

    /// Uploads one graph weight, or this rank's slice of it.
    fn load(&mut self, weight: u32, data: View) -> Result<(), Error> {
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
        let shaped = |o: &sensai_lower::Output| Ok(self.read(o.slot)?.reshaped(o.shape.clone()));
        outputs.map(shaped).collect()
    }

    /// Runs step by step. Returns each step's seconds.
    fn run_timed(&mut self, inputs: &[View]) -> Result<Vec<f64>, Error> {
        self.feed(inputs)?;
        let steps = 0..self.plan().steps.len();
        steps.map(|i| self.repeat(i, 1)).collect()
    }
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
