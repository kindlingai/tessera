//! One process's rank of a plan, over a transport.

use crate::open;
use crate::{Base, Error, Run};
use std::ops::Range;
use std::time::Instant;
use tessera_codegen::Precision;
use tessera_engine_comm::reduce::{CUDA, ENTRY};
use tessera_engine_comm::{Reduce, Transport, summed};
use tessera_engine_settings::Settings;
use tessera_gpukit::Platform;
use tessera_lower::{Arg, Launching, Plan, Shape, Step};
use tessera_mem::{DType, Tensor};

/// A sum across ranks between two device addresses.
pub trait Sum {
    /// Sums `len` floats at `from` into `to`. Waits for it.
    fn sum(&mut self, from: u64, to: u64, len: usize) -> Result<(), Error>;
}

/// One rank: its runner and its way to the others.
pub struct Rank<T: Transport> {
    runner: Box<dyn Run>,
    transport: T,
    /// The device's rounds, when it runs them itself.
    reduce: Option<Reduce>,
    /// A persistent grid that runs the sums, if given.
    grid: Option<Box<dyn Sum>>,
}

impl<T: Transport> Rank<T> {
    /// A runner for the transport's rank of `plan`.
    pub fn open(plan: Plan, settings: &Settings, transport: T) -> Result<Self, Error> {
        let fits = plan.ranks.max(1) == transport.ranks() && plan.rank == transport.rank();
        if !fits {
            return Err(Error::RankMismatch {
                plan: plan.rank,
                plans: plan.ranks,
                rank: transport.rank(),
                ranks: transport.ranks(),
            });
        }
        let runner = open(plan, settings)?;
        let mut transport = transport;
        let reduce = match runner.fixed() {
            true => transport.device(),
            false => None,
        };
        Ok(Self {
            runner,
            transport,
            reduce,
            grid: None,
        })
    }

    /// Runs the sums through `grid` from now on.
    pub fn with_grid(mut self, grid: Option<Box<dyn Sum>>) -> Self {
        self.grid = grid;
        self
    }

    /// Whether sums run on the device.
    pub fn on_device(&self) -> bool {
        self.reduce.is_some() || self.grid.is_some()
    }

    /// One sum through the persistent grid.
    fn grid_sum(&mut self, i: usize, step: &Step) -> Result<(), Error> {
        let at = |slot| self.runner.address(slot).ok_or(Error::Unaddressed(i));
        let (from, to) = (at(step.reads[0])?, at(step.write)?);
        let grid = self.grid.as_mut().expect("a grid");
        grid.sum(from, to, step.write.len)
    }

    /// One sum on the device: copy, publish, wait and add.
    fn reduce(&mut self, step: &Step) -> Result<(), Error> {
        let r = self.reduce.as_mut().expect("a device");
        let n = step.write.len;
        if 16 + 4 * n > r.capacity {
            return Err(Error::SumTooLong(n));
        }
        let seq = r.round();
        let args = [
            Arg::value(&r.send),
            Arg::value(&r.recv),
            Arg::value(&r.flags),
            Arg::value(&r.ctl),
            step.read(0),
            Arg::at(step.write, 0),
            Arg::value(&(n as u32)),
            Arg::value(&r.rank),
            Arg::value(&r.ranks),
            Arg::value(&r.lanes),
            Arg::value(&seq),
            Arg::value(&(r.capacity as u64)),
        ];
        let blocks = (n / 1024).clamp(1, 8);
        let launched = self
            .runner
            .launch_fixed(CUDA, ENTRY, &args, Shape::Line(blocks * 256));
        if let Some(why) = self.reduce.as_ref().and_then(Reduce::failure) {
            return Err(tessera_engine_comm::Error::Link(why).into());
        }
        launched
    }

    /// One sum on the host, over the transport.
    fn exchange(&mut self, step: &Step) -> Result<(), Error> {
        let part = self.runner.read(step.reads[0])?;
        let parts = self.transport.exchange(part.bytes())?;
        let floats = |p: &Vec<u8>| Tensor::from_bytes(vec![p.len() / 4], DType::F32, p);
        let parts: Vec<Tensor> = parts.iter().map(floats).collect();
        let sum = summed(parts.iter().map(|p| p.floats().expect("floats")));
        let sum = Tensor::f32(vec![step.write.len], sum);
        self.runner.write(step.write, sum.view())
    }
}

impl<T: Transport> Run for Rank<T> {
    fn base(&self) -> &Base {
        self.runner.base()
    }

    fn base_mut(&mut self) -> &mut Base {
        self.runner.base_mut()
    }

    fn replan(&mut self, plan: &Plan) -> Result<(), Error> {
        self.runner.replan(plan)
    }

    /// Runs these steps. Sums wait for every rank.
    fn submit(&mut self, steps: Range<usize>) -> Result<(), Error> {
        let mut from = steps.start;
        for i in steps.clone() {
            let step = &self.runner.plan().steps[i];
            if step.kernel.name() != "all_reduce" {
                continue;
            }
            let step = step.clone();
            self.runner.submit(from..i)?;
            from = i + 1;
            let start = Instant::now();
            match (&self.grid, &self.reduce) {
                (Some(_), _) => self.grid_sum(i, &step)?,
                (None, Some(_)) => self.reduce(&step)?,
                (None, None) => self.exchange(&step)?,
            }
            self.note("all_reduce", start.elapsed().as_secs_f64());
        }
        self.runner.submit(from..steps.end)
    }

    fn fixed(&self) -> bool {
        self.runner.fixed()
    }

    fn address(&self, slot: tessera_lower::Slot) -> Option<u64> {
        self.runner.address(slot)
    }

    fn launch_fixed(
        &self,
        source: &'static str,
        entry: &'static str,
        args: &[Arg],
        shape: Shape,
    ) -> Result<(), Error> {
        self.runner.launch_fixed(source, entry, args, shape)
    }

    fn launching(&mut self, how: Launching) {
        self.runner.launching(how)
    }

    fn repeat(&mut self, step: usize, times: usize) -> Result<f64, Error> {
        self.runner.repeat(step, times)
    }

    fn device(&self) -> String {
        self.runner.device()
    }

    fn compiled(&self) -> (usize, usize) {
        self.runner.compiled()
    }

    fn platform(&self) -> Platform {
        self.runner.platform()
    }

    fn precision(&self) -> Precision {
        self.runner.precision()
    }
}
