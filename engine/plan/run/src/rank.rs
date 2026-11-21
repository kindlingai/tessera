//! One process's rank of a plan, over a transport.

use crate::open;
use crate::{Base, Error, Run};
use std::ops::Range;
use std::time::Instant;
use tessera_codegen::Precision;
use tessera_engine_comm::{Transport, summed};
use tessera_engine_settings::Settings;
use tessera_gpukit::Platform;
use tessera_lower::{Launching, Plan, Step};
use tessera_mem::{DType, Tensor};

/// A sum across ranks between two device addresses.
pub trait Sum {
    /// Queues a sum of `len` floats at `from` into `to`.
    fn sum(&mut self, from: u64, to: u64, len: usize) -> Result<(), Error>;
    /// Fails if a queued sum failed. May wait.
    fn check(&mut self) -> Result<(), Error> {
        Ok(())
    }
    /// Gives later sums a barrier round, and times them.
    fn timing(&mut self, _on: bool) {}
    /// Seconds waiting and moving since the last call, if timed.
    fn spent(&mut self) -> Result<Option<[f64; 2]>, Error> {
        Ok(None)
    }
}

/// One rank: its runner and its way to the others.
pub struct Rank<T: Transport> {
    runner: Box<dyn Run>,
    transport: T,
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
        Ok(Self {
            runner,
            transport,
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
        self.grid.is_some()
    }

    /// One sum through the persistent grid.
    fn grid_sum(&mut self, i: usize, step: &Step) -> Result<(), Error> {
        let at = |slot| self.runner.address(slot).ok_or(Error::Unaddressed(i));
        let (from, to) = (at(step.reads[0])?, at(step.write)?);
        let grid = self.grid.as_mut().expect("a grid");
        grid.sum(from, to, step.write.len)
    }

    /// A host sum. Profiling first waits for all ranks.
    fn exchange(&mut self, step: &Step) -> Result<(), Error> {
        let start = Instant::now();
        let part = self.runner.read(step.reads[0])?;
        let mut at = self.lap("all_reduce: read", start);
        if self.base().profile.is_some() {
            self.transport.exchange(&[])?;
            at = self.lap("all_reduce: wait", at);
        }
        let parts = self.transport.exchange(part.bytes())?;
        at = self.lap("all_reduce: transfer", at);
        let floats = |p: &Vec<u8>| Tensor::from_bytes(vec![p.len() / 4], DType::F32, p);
        let parts: Vec<Tensor> = parts.iter().map(floats).collect();
        let sum = summed(parts.iter().map(|p| p.floats().expect("floats")));
        let sum = Tensor::f32(vec![step.write.len], sum);
        self.runner.write(step.write, sum.view())?;
        self.lap("all_reduce: sum", at);
        self.lap("all_reduce", start);
        Ok(())
    }

    /// Notes the seconds since `from` as `name`. Returns now.
    fn lap(&mut self, name: &str, from: Instant) -> Instant {
        let now = Instant::now();
        self.note(name, (now - from).as_secs_f64());
        now
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
        let timed = self.base().profile.is_some();
        if let Some(grid) = &mut self.grid {
            grid.timing(timed);
        }
        let mut from = steps.start;
        for i in steps.clone() {
            let step = &self.runner.plan().steps[i];
            if step.kernel.name() != "all_reduce" {
                continue;
            }
            let step = step.clone();
            self.runner.submit(from..i)?;
            from = i + 1;
            match self.grid.is_some() {
                true => self.grid_sum(i, &step)?,
                false => self.exchange(&step)?,
            }
        }
        self.runner.submit(from..steps.end)?;
        let Some(grid) = self.grid.as_mut().filter(|_| from > steps.start) else {
            return Ok(());
        };
        grid.check()?;
        if let Some([wait, moved]) = grid.spent()? {
            self.note("all_reduce: wait", wait);
            self.note("all_reduce: transfer", moved);
            self.note("all_reduce", wait + moved);
        }
        Ok(())
    }

    fn address(&self, slot: tessera_lower::Slot) -> Option<u64> {
        self.runner.address(slot)
    }

    fn launching(&mut self, how: Launching) {
        self.runner.launching(how)
    }

    fn persist(&mut self, grid: Box<dyn crate::Grid>) -> Result<(), Error> {
        self.runner.persist(grid)
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
