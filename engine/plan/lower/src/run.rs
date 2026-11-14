//! What every backend's plan runner does.

use crate::Error;
use crate::form::Form;
use crate::plan::{Plan, Slot};
use crate::recipe::Launching;
use codegen::Precision;
use gpukit::backend::Backend;
use gpukit::{Limits, Platform, Settings};
use graph::DimName;
use mem::{Buffer, Tensor, View, ViewMut, copy};
use std::collections::HashMap;
use std::ops::Range;

/// Buffers that outlive one plan.
#[derive(Default)]
pub struct Held {
    /// Every live buffer.
    all: Vec<Buffer>,
    weights: HashMap<u32, usize>,
    kept: HashMap<u32, usize>,
    /// Where each of the plan's buffers lies in `all`.
    bound: Vec<usize>,
}

impl Held {
    pub fn get(&self, slot: Slot) -> &Buffer {
        &self.all[self.bound[slot.buffer as usize]]
    }

    /// These buffers' bytes, each named once, for the host.
    pub fn mapped_mut(&mut self, buffers: &[u32]) -> Vec<&mut [u8]> {
        let wanted: Vec<usize> = buffers.iter().map(|b| self.bound[*b as usize]).collect();
        let mut out: Vec<Option<&mut [u8]>> = (0..wanted.len()).map(|_| None).collect();
        for (at, buffer) in self.all.iter_mut().enumerate() {
            if let Some(i) = wanted.iter().position(|w| *w == at) {
                out[i] = buffer.backing_mut().mapped_mut();
            }
        }
        out.into_iter().map(|o| o.expect("host memory")).collect()
    }

    pub fn view(&self, slot: Slot) -> View<'_> {
        self.get(slot)
            .view(slot.start(), slot.form.dtype(), vec![slot.len])
    }

    /// `bytes` into the slot, as the form packed them.
    pub fn upload(&mut self, slot: Slot, bytes: &[u8]) -> Result<(), Error> {
        let at = self.bound[slot.buffer as usize];
        Ok(self.all[at].backing_mut().upload(slot.start(), bytes)?)
    }

    pub fn view_mut(&mut self, slot: Slot) -> ViewMut<'_> {
        let at = self.bound[slot.buffer as usize];
        self.all[at].view_mut(slot.start(), slot.form.dtype(), vec![slot.len])
    }

    /// Binds `plan`'s buffers. Weights and kept rows stay as loaded.
    pub fn bind(
        &mut self,
        plan: &Plan,
        mut alloc: impl FnMut(u32, usize) -> Result<Buffer, Error>,
    ) -> Result<(), Error> {
        let old = std::mem::take(&mut self.all).into_iter().map(Some);
        let mut old: Vec<Option<Buffer>> = old.collect();
        for at in self.weights.values_mut().chain(self.kept.values_mut()) {
            self.all.push(old[*at].take().expect("each is held once"));
            *at = self.all.len() - 1;
        }
        // The last plan's scratch buffers.
        let mut spare: Vec<Buffer> = old.into_iter().flatten().collect();
        let mut owner = vec![None; plan.buffers.len()];
        for l in &plan.weights {
            owner[l.slot.buffer as usize] = Some((false, l.weight));
        }
        for (index, slot) in &plan.kept {
            owner[slot.buffer as usize] = Some((true, *index));
        }
        self.bound.clear();
        for (b, (&bytes, owner)) in plan.buffers.iter().zip(owner).enumerate() {
            let zone = plan.zone(b as u32);
            // Named buffers fit exactly. Scratch may be larger.
            let fits = |buffer: &Buffer| buffer.zone == zone && buffer.bytes() >= bytes;
            let held = match owner {
                Some((true, index)) => self.kept.get(&index).copied(),
                Some((false, index)) => self.weights.get(&index).copied(),
                None => spare.iter().position(fits).map(|at| {
                    self.all.push(spare.swap_remove(at));
                    self.all.len() - 1
                }),
            };
            let same = |buffer: &Buffer| buffer.zone == zone && buffer.bytes() == bytes;
            let at = match held.filter(|at| owner.is_none() || same(&self.all[*at])) {
                Some(at) => at,
                None => {
                    // Scratch grows in steps, so a growing context reuses it.
                    let want = match owner {
                        Some(_) => bytes,
                        None => bytes.next_power_of_two(),
                    };
                    self.all.push(alloc(zone, want)?);
                    self.all.len() - 1
                }
            };
            match owner {
                Some((true, index)) => self.kept.insert(index, at),
                Some((false, index)) => self.weights.insert(index, at),
                None => None,
            };
            self.bound.push(at);
        }
        Ok(())
    }
}

/// A weight as a file holds it.
pub enum Weight {
    Floats(Tensor),
    /// One form's bytes, as `Form::pack` lays them out.
    Packed(&'static dyn Form, Vec<u8>),
}

/// A plan with its buffers allocated and kernels compiled.
pub trait Run {
    fn plan(&self) -> &Plan;
    fn held(&self) -> &Held;
    fn held_mut(&mut self) -> &mut Held;
    /// Takes up another plan. Weights and kept rows stay loaded.
    fn replan(&mut self, plan: &Plan) -> Result<(), Error>;
    /// Gives a runtime dimension its value for the next launches.
    fn bind(&mut self, name: DimName, value: u32);
    /// Runs these steps in order and waits for them.
    fn submit(&mut self, steps: Range<usize>) -> Result<(), Error>;
    /// How to launch from now on, where the device can.
    fn launching(&mut self, _: Launching) {}
    /// Runs one step `times` over. Returns seconds per run.
    fn repeat(&mut self, step: usize, times: usize) -> Result<f64, Error>;
    /// The backend and the hardware it drives.
    fn device(&self) -> String;
    /// What the hardware gives one group of threads.
    fn limits(&self) -> Limits {
        self.platform().limits
    }
    /// The hardware as asked, with its table row.
    fn platform(&self) -> Platform;
    /// How the hardware's tiles round.
    fn precision(&self) -> Precision;

    /// Uploads a weight as stored, widened where a slot differs.
    fn load_weight(&mut self, weight: u32, w: &Weight) -> Result<(), Error> {
        let Weight::Packed(form, bytes) = w else {
            let Weight::Floats(t) = w else { unreachable!() };
            return self.load(weight, t.view());
        };
        let slots: Vec<Slot> = (self.plan().weights.iter())
            .filter(|l| l.weight == weight)
            .map(|l| l.slot)
            .collect();
        for slot in slots {
            if form.bytes(slot.len) != bytes.len() {
                return Err(Error::Size(format!("weight {weight} as {}", form.name())));
            }
            if slot.form.name() == form.name() {
                self.held_mut().upload(slot, bytes)?;
                continue;
            }
            let floats = (0..slot.len).map(|i| form.unpack(bytes, i)).collect();
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
                return Err(Error::Size(format!(
                    "weight {weight} as {}",
                    slot.form.name()
                )));
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

    /// Uploads one graph weight. Unused weights are ignored.
    fn load(&mut self, weight: u32, data: View) -> Result<(), Error> {
        let slots: Vec<Slot> = (self.plan().weights.iter())
            .filter(|l| l.weight == weight)
            .map(|l| l.slot)
            .collect();
        for slot in slots {
            if slot.len != data.len() {
                return Err(Error::Size(format!("weight {weight}")));
            }
            self.write(slot, data.clone())?;
        }
        Ok(())
    }

    fn feed(&mut self, inputs: &[View]) -> Result<(), Error> {
        let slots = self.plan().inputs.clone();
        if inputs.len() != slots.len() {
            return Err(Error::Size("the input list".into()));
        }
        for (at, (slot, input)) in slots.into_iter().zip(inputs).enumerate() {
            if input.len() != slot.len || input.dtype() != slot.form.dtype() {
                return Err(Error::Size(format!("input {at}")));
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
        let shaped = |o: &crate::plan::Output| Ok(self.read(o.slot)?.reshaped(o.shape.clone()));
        outputs.map(shaped).collect()
    }

    /// Runs step by step. Returns each step's seconds.
    fn run_timed(&mut self, inputs: &[View]) -> Result<Vec<f64>, Error> {
        self.feed(inputs)?;
        let steps = 0..self.plan().steps.len();
        steps.map(|i| self.repeat(i, 1)).collect()
    }
}

/// How `backend`'s tiles round.
pub fn precision(backend: Backend) -> Precision {
    match backend {
        Backend::Metal => Precision::METAL,
        Backend::Cuda => Precision::CUDA,
        Backend::Rocm => Precision::HIP,
    }
}

/// A runner on the GPU `settings` picks, or the host.
pub fn open(plan: Plan, settings: &Settings) -> Result<Box<dyn Run>, Error> {
    #[cfg(any(target_os = "macos", feature = "cuda", feature = "rocm"))]
    use gpukit::backend::Backend;
    match settings.gpu() {
        #[cfg(target_os = "macos")]
        Some(Backend::Metal) => Ok(Box::new(crate::metal::Runner::new(plan)?)),
        #[cfg(feature = "cuda")]
        Some(Backend::Cuda) => Ok(Box::new(crate::device::cuda::Runner::new(plan)?)),
        #[cfg(feature = "rocm")]
        Some(Backend::Rocm) => Ok(Box::new(crate::device::rocm::Runner::new(plan)?)),
        _ => Ok(Box::new(crate::cpu::Runner::new(
            plan,
            settings.host_lanes,
            settings.emulate.map_or(Precision::EXACT, precision),
        )?)),
    }
}
