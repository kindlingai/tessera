//! What every backend's plan runner does.

use crate::Error;
use crate::plan::{Plan, Slot};
use std::ops::Range;
use std::time::Instant;

/// One entry input's values.
pub enum Feed<'a> {
    Floats(&'a [f32]),
    Ids(&'a [u32]),
}

/// A plan with its buffers allocated and kernels compiled.
pub trait Run {
    fn plan(&self) -> &Plan;
    /// Copies host bytes to the start of a slot.
    fn write(&self, slot: Slot, bytes: &[u8]) -> Result<(), Error>;
    fn read(&self, slot: Slot) -> Result<Vec<f32>, Error>;
    /// Runs these steps in order and waits for them.
    fn submit(&self, steps: Range<usize>) -> Result<(), Error>;

    /// Uploads one graph weight. Unused weights are ignored.
    fn load(&self, weight: u32, data: &[f32]) -> Result<(), Error> {
        for (_, slot) in self.plan().weights.iter().filter(|(w, _)| *w == weight) {
            if slot.len != data.len() {
                return Err(Error::Size(format!("weight {weight}")));
            }
            let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_ne_bytes()).collect();
            self.write(*slot, &bytes)?;
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

    /// Runs and waits step by step; returns each step's seconds.
    fn run_timed(&self, inputs: &[Feed]) -> Result<Vec<f64>, Error> {
        self.feed(inputs)?;
        let mut seconds = Vec::with_capacity(self.plan().steps.len());
        for i in 0..self.plan().steps.len() {
            let start = Instant::now();
            self.submit(i..i + 1)?;
            seconds.push(start.elapsed().as_secs_f64());
        }
        Ok(seconds)
    }
}

/// A runner on the backend this build dispatches to.
pub fn open(plan: Plan) -> Result<Box<dyn Run>, Error> {
    use gpukit::backend::{Backend, available};
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
