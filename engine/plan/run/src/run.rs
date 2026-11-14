//! What every backend's plan runner does.

use crate::held::Held;
use codegen::{Dialect, Precision};
use gpukit::{Limits, Platform};
use graph::DimName;
use lower::{Error, Launch, Launching, Plan, Slot, Source, Step, Weight};
use mem::{Tensor, View, copy};
use std::collections::HashMap;
use std::ops::Range;

/// What every runner keeps: a plan, its buffers and sizes.
#[derive(Default)]
pub struct Base {
    pub(crate) plan: Plan,
    pub(crate) held: Held,
    /// Runtime dimensions, as last bound.
    pub(crate) bound: HashMap<DimName, u32>,
}

impl Base {
    /// Whether `plan` is new here. Plans over several zones fail.
    pub(crate) fn fresh(&self, plan: &Plan) -> Result<bool, Error> {
        if plan.id != 0 && plan.id == self.plan.id {
            return Ok(false);
        }
        if plan.zoned() {
            return Err(Error::Unsupported("a plan over several zones".into()));
        }
        Ok(true)
    }

    /// Each launch of a step, its runtime sizes filled in.
    pub(crate) fn launches(&self, step: usize) -> Result<Vec<Launch>, Error> {
        let size = |name| self.bound.get(&name).copied();
        let bound = |l: Launch| l.bound(&size).ok_or(Error::Unbound);
        self.plan.steps[step]
            .launches()
            .into_iter()
            .map(bound)
            .collect()
    }
}

/// Groups and threads for a line of `count` threads.
pub(crate) fn line(count: usize) -> (usize, usize) {
    let threads = count.clamp(1, 256);
    (count.div_ceil(threads), threads)
}

/// Compiled kernels, by how a kernel prints and by entry.
pub(crate) struct Kernels<T> {
    made: HashMap<String, T>,
    /// With whether each was generated.
    known: HashMap<String, (T, bool)>,
}

impl<T> Default for Kernels<T> {
    fn default() -> Self {
        Self {
            made: HashMap::new(),
            known: HashMap::new(),
        }
    }
}

impl<T: Clone> Kernels<T> {
    /// `step`'s kernel, compiled once. True when generated.
    pub(crate) fn get(
        &mut self,
        step: &Step,
        dialect: Dialect,
        fixed: impl FnOnce(&'static str, &'static str) -> Result<T, Error>,
        made: impl FnOnce(&str, &str) -> Result<T, Error>,
    ) -> Result<(T, bool), Error> {
        // Printing a kernel is cheaper than writing its source.
        let key = format!("{:?}", step.kernel);
        if let Some(found) = self.known.get(&key) {
            return Ok(found.clone());
        }
        let found = match step.kernel.source() {
            Source::Fixed { metal, cuda, entry } => {
                let text = if dialect == Dialect::Metal {
                    metal
                } else {
                    cuda
                };
                (fixed(text, entry)?, false)
            }
            Source::Made(body) => match self.made.get(&body.entry) {
                Some(compiled) => (compiled.clone(), true),
                None => {
                    let compiled = made(&body.text().render(dialect), &body.entry)?;
                    self.made.insert(body.entry.clone(), compiled.clone());
                    (compiled, true)
                }
            },
        };
        self.known.insert(key, found.clone());
        Ok(found)
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
    /// Takes up another plan. Weights and kept rows stay loaded.
    fn replan(&mut self, plan: &Plan) -> Result<(), Error>;
    /// Gives a runtime dimension its value for the next launches.
    fn bind(&mut self, name: DimName, value: u32) {
        self.base_mut().bound.insert(name, value);
    }
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
        let shaped = |o: &lower::Output| Ok(self.read(o.slot)?.reshaped(o.shape.clone()));
        outputs.map(shaped).collect()
    }

    /// Runs step by step. Returns each step's seconds.
    fn run_timed(&mut self, inputs: &[View]) -> Result<Vec<f64>, Error> {
        self.feed(inputs)?;
        let steps = 0..self.plan().steps.len();
        steps.map(|i| self.repeat(i, 1)).collect()
    }
}
