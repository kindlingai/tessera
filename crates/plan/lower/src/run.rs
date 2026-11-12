//! What every backend's plan runner does.

use crate::Error;
use crate::plan::{Plan, Slot};
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

    /// The slot as a flat view.
    pub fn view(&self, slot: Slot) -> View<'_> {
        self.get(slot)
            .view(slot.start(), slot.dtype, vec![slot.len])
    }

    pub fn view_mut(&mut self, slot: Slot) -> ViewMut<'_> {
        let at = self.bound[slot.buffer as usize];
        self.all[at].view_mut(slot.start(), slot.dtype, vec![slot.len])
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
            let same = |buffer: &Buffer| buffer.bytes() == bytes && buffer.zone == zone;
            let held = match owner {
                Some((true, index)) => self.kept.get(&index).copied(),
                Some((false, index)) => self.weights.get(&index).copied(),
                None => spare.iter().position(same).map(|at| {
                    self.all.push(spare.swap_remove(at));
                    self.all.len() - 1
                }),
            };
            let at = match held.filter(|at| same(&self.all[*at])) {
                Some(at) => at,
                None => {
                    self.all.push(alloc(zone, bytes)?);
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

/// A plan with its buffers allocated and kernels compiled.
pub trait Run {
    fn plan(&self) -> &Plan;
    fn held(&self) -> &Held;
    fn held_mut(&mut self) -> &mut Held;
    /// Takes up another plan. Weights and kept rows stay loaded.
    fn replan(&mut self, plan: Plan) -> Result<(), Error>;
    /// Runs these steps in order and waits for them.
    fn submit(&self, steps: Range<usize>) -> Result<(), Error>;
    /// Runs one step `times` over. Returns seconds per run.
    fn repeat(&self, step: usize, times: usize) -> Result<f64, Error>;
    /// The backend and the hardware it drives.
    fn device(&self) -> String;

    /// Copies `src` into the slot, converting its form if needed.
    fn write(&mut self, slot: Slot, src: View) -> Result<(), Error> {
        Ok(copy(src, &mut self.held_mut().view_mut(slot))?)
    }

    /// The slot's values on the host, as one row.
    fn read(&self, slot: Slot) -> Result<Tensor, Error> {
        Ok(self.held().view(slot).to_tensor()?)
    }

    fn ids(&self, slot: Slot) -> Result<Vec<u32>, Error> {
        Ok(self.held().view(slot).to_u32()?)
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
            if input.len() != slot.len || input.dtype() != slot.dtype {
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

/// A runner on the backend this build dispatches to.
pub fn open(plan: Plan) -> Result<Box<dyn Run>, Error> {
    #[cfg(any(target_os = "macos", feature = "cuda", feature = "rocm"))]
    use gpukit::backend::Backend;
    use gpukit::backend::available;
    match available() {
        #[cfg(target_os = "macos")]
        Some(Backend::Metal) => Ok(Box::new(crate::metal::Runner::new(plan)?)),
        #[cfg(feature = "cuda")]
        Some(Backend::Cuda) => Ok(Box::new(crate::device::cuda::Runner::new(plan)?)),
        #[cfg(feature = "rocm")]
        Some(Backend::Rocm) => Ok(Box::new(crate::device::rocm::Runner::new(plan)?)),
        _ => {
            drop(plan);
            Err(Error::Unsupported("a build with no GPU backend".into()))
        }
    }
}
