//! What every backend's plan runner does.

use crate::Error;
use crate::plan::{Plan, Slot};
use std::collections::HashMap;
use std::ops::Range;

/// One entry input's values.
pub enum Feed<'a> {
    Floats(&'a [f32]),
    Ids(&'a [u32]),
}

/// Device buffers that outlive one plan.
pub struct Held<B> {
    /// Every live buffer, with its element count.
    all: Vec<(usize, B)>,
    weights: HashMap<u32, usize>,
    kept: HashMap<u32, usize>,
    /// Where each of the plan's buffers lies in `all`.
    bound: Vec<usize>,
}

impl<B> Default for Held<B> {
    fn default() -> Self {
        let (weights, kept) = Default::default();
        Self {
            all: Vec::new(),
            weights,
            kept,
            bound: Vec::new(),
        }
    }
}

impl<B> Held<B> {
    pub fn get(&self, slot: Slot) -> &B {
        &self.all[self.bound[slot.buffer as usize]].1
    }

    /// Binds `plan`'s buffers. Weights and kept rows stay as loaded.
    pub fn bind(
        &mut self,
        plan: &Plan,
        mut alloc: impl FnMut(usize) -> Result<B, Error>,
    ) -> Result<(), Error> {
        let old = std::mem::take(&mut self.all).into_iter().map(Some);
        let mut old: Vec<Option<(usize, B)>> = old.collect();
        for at in self.weights.values_mut().chain(self.kept.values_mut()) {
            self.all.push(old[*at].take().expect("each is held once"));
            *at = self.all.len() - 1;
        }
        // The last plan's scratch buffers.
        let mut spare: Vec<(usize, B)> = old.into_iter().flatten().collect();
        let mut owner = vec![None; plan.buffers.len()];
        for l in &plan.weights {
            owner[l.slot.buffer as usize] = Some((false, l.weight));
        }
        for (index, slot) in &plan.kept {
            owner[slot.buffer as usize] = Some((true, *index));
        }
        self.bound.clear();
        for (&len, owner) in plan.buffers.iter().zip(owner) {
            let held = match owner {
                Some((true, index)) => self.kept.get(&index).copied(),
                Some((false, index)) => self.weights.get(&index).copied(),
                None => {
                    let fits = spare.iter().position(|(n, _)| *n == len);
                    fits.map(|at| {
                        self.all.push(spare.swap_remove(at));
                        self.all.len() - 1
                    })
                }
            };
            let at = match held.filter(|at| self.all[*at].0 == len) {
                Some(at) => at,
                None => {
                    self.all.push((len, alloc(len)?));
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
    /// Takes up another plan. Weights and kept rows stay loaded.
    fn replan(&mut self, plan: Plan) -> Result<(), Error>;
    /// Copies host bytes to the start of a slot.
    fn write(&self, slot: Slot, bytes: &[u8]) -> Result<(), Error>;
    fn read(&self, slot: Slot) -> Result<Vec<f32>, Error>;

    /// Reads a slot of indices, stored as float bits.
    fn ids(&self, slot: Slot) -> Result<Vec<u32>, Error> {
        Ok(self.read(slot)?.into_iter().map(f32::to_bits).collect())
    }

    /// Runs these steps in order and waits for them.
    fn submit(&self, steps: Range<usize>) -> Result<(), Error>;
    /// Runs one step `times` over. Returns seconds per run.
    fn repeat(&self, step: usize, times: usize) -> Result<f64, Error>;
    /// The backend and the hardware it drives.
    fn device(&self) -> String;

    /// Uploads one graph weight as planned. Unused weights are ignored.
    fn load(&self, weight: u32, data: &[f32]) -> Result<(), Error> {
        for l in self.plan().weights.iter().filter(|l| l.weight == weight) {
            if l.slot.len != l.form.words(data.len()) {
                return Err(Error::Size(format!("weight {weight}")));
            }
            self.write(l.slot, &l.form.narrow(data))?;
        }
        Ok(())
    }

    fn feed(&self, inputs: &[Feed]) -> Result<(), Error> {
        if inputs.len() != self.plan().inputs.len() {
            return Err(Error::Size("the input list".into()));
        }
        for (at, (slot, feed)) in self.plan().inputs.iter().zip(inputs).enumerate() {
            let bytes: Vec<u8> = match feed {
                Feed::Floats(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
                Feed::Ids(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            };
            if bytes.len() != slot.len * 4 {
                return Err(Error::Size(format!("input {at}")));
            }
            self.write(*slot, &bytes)?;
        }
        Ok(())
    }

    /// Runs every step as one submission.
    fn run(&self, inputs: &[Feed]) -> Result<Vec<Vec<f32>>, Error> {
        self.feed(inputs)?;
        self.submit(0..self.plan().steps.len())?;
        let outputs = self.plan().outputs.iter();
        outputs.map(|o| self.read(o.slot)).collect()
    }

    /// Runs step by step. Returns each step's seconds.
    fn run_timed(&self, inputs: &[Feed]) -> Result<Vec<f64>, Error> {
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
