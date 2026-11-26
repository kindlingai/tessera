//! One process's rank of a plan, over a transport.

use crate::open;
use crate::{Base, Error, Run};
use std::ops::Range;
use std::time::Instant;
use tessera_codegen::Precision;
use tessera_engine_comm::{Transport, gathered, summed};
use tessera_engine_settings::Settings;
use tessera_gpukit::Platform;
use tessera_gpukit::backend::OnGpu;
use tessera_gpukit::device::Device;
use tessera_gpukit::launch::Launch;
use tessera_lower::kernels::{AllGather, crosses};
use tessera_lower::{Launching, Plan, Step};
use tessera_mem::{DType, Tensor};

/// A sum or gather across ranks between two device addresses.
pub trait Sum {
    /// Queues `len` floats as `wire` parts. Zero starts the grid.
    fn sum(&mut self, from: u64, to: u64, len: usize, wire: DType) -> Result<(), Error>;
    /// The wire dtype for sums the model leaves unpinned.
    fn wire(&self) -> DType {
        DType::F32
    }
    /// Whether `sum` takes parts as `wire`.
    fn carries(&self, wire: DType) -> bool {
        matches!(wire, DType::F32 | DType::Bf16)
    }
    /// Queues a gather of `part` floats over `rows` rows.
    fn gather(
        &mut self,
        _from: u64,
        _to: u64,
        _part: usize,
        _total: usize,
        _rows: usize,
    ) -> Result<bool, Error> {
        Ok(false)
    }
    /// Whether launches capture into a device graph.
    fn capturable(&self) -> bool {
        false
    }
    /// Stops a running grid before context-syncing device calls.
    fn pause(&mut self) -> Result<(), Error> {
        Ok(())
    }
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

/// Aborts a capture cleanly on error or host-fallback gather.
pub(crate) enum Capture {
    Err(Error),
    Split(usize),
}

impl Capture {
    pub(crate) fn into_error(self) -> Error {
        match self {
            Self::Err(e) => e,
            Self::Split(_) => unreachable!("plain launch never splits"),
        }
    }
}

impl From<tessera_gpukit::Error> for Capture {
    fn from(e: tessera_gpukit::Error) -> Self {
        Self::Err(e.into())
    }
}

impl From<Error> for Capture {
    fn from(e: Error) -> Self {
        Self::Err(e)
    }
}

/// A device runner that captures crossing steps in one graph.
pub(crate) trait RankRun: Run {
    fn can_graph(&self, steps: Range<usize>) -> bool;
    fn replan_with(
        &mut self,
        plan: &Plan,
        sum: Option<&mut (dyn Sum + 'static)>,
    ) -> Result<(), Error>;
    fn run_with(
        &mut self,
        steps: Range<usize>,
        wait: bool,
        cross: bool,
        sum: &mut (dyn Sum + 'static),
    ) -> Result<Option<usize>, Error>;
    fn finish_device(&self) -> Result<(), Error>;
}

impl<D: Device + Launch<Kernel = <D as Device>::Kernel>> RankRun for crate::device::Runner<D> {
    fn can_graph(&self, steps: Range<usize>) -> bool {
        self.can_record(&steps)
    }

    fn replan_with(
        &mut self,
        plan: &Plan,
        sum: Option<&mut (dyn Sum + 'static)>,
    ) -> Result<(), Error> {
        self.replan_with(plan, sum)
    }

    fn run_with(
        &mut self,
        steps: Range<usize>,
        wait: bool,
        cross: bool,
        sum: &mut (dyn Sum + 'static),
    ) -> Result<Option<usize>, Error> {
        self.run(steps, false, wait, cross, Some(sum))
            .map(|(_, split)| split)
    }

    fn finish_device(&self) -> Result<(), Error> {
        self.finish_ctx()
    }
}

/// Unpinned sums' wire: `setting`, else `grid`'s, else F32 for host.
pub fn sum_wire(setting: Option<DType>, grid: Option<&dyn Sum>) -> DType {
    setting.unwrap_or_else(|| grid.map_or(DType::F32, |g| g.wire()))
}

/// The wire dtype of all-reduce `step`.
pub(crate) fn wire(step: &Step) -> DType {
    step.kernel
        .wire()
        .expect("an all-reduce step has a wire dtype")
}

/// Fails if `grid` cannot carry a wire of `plan`.
fn carried(plan: &Plan, grid: Option<&dyn Sum>) -> Result<(), Error> {
    let Some(grid) = grid else {
        return Ok(());
    };
    for step in &plan.steps {
        if let Some(wire) = step.kernel.wire()
            && !grid.carries(wire)
        {
            return Err(Error::Wire(wire));
        }
    }
    Ok(())
}

/// Queues one crossing step on `grid`. False means host fallback.
pub(crate) fn grid_cross(
    step: &Step,
    from: u64,
    to: u64,
    grid: &mut dyn Sum,
) -> Result<bool, Error> {
    if step.kernel.name() != AllGather::NAME {
        grid.sum(from, to, step.write.len, wire(step))?;
        return Ok(true);
    }
    let (src, dst) = (step.reads[0], step.write);
    if src.form.name() != "f32" || dst.form.name() != "f32" {
        return Ok(false);
    }
    let rows = step.kernel.sizes()[0];
    grid.gather(from, to, src.len, dst.len, rows)
}

enum Inner {
    Plain(Box<dyn Run>),
    Device(Box<dyn RankRun>),
}

impl Inner {
    fn run(&self) -> &dyn Run {
        match self {
            Self::Plain(r) => r.as_ref(),
            Self::Device(r) => r.as_ref(),
        }
    }

    fn run_mut(&mut self) -> &mut dyn Run {
        match self {
            Self::Plain(r) => r.as_mut(),
            Self::Device(r) => r.as_mut(),
        }
    }
}

struct OpenRank<'a>(Plan, &'a Settings);

impl OnGpu for OpenRank<'_> {
    type Out = Result<Box<dyn RankRun>, Error>;

    fn on<D: Device + Launch<Kernel = <D as Device>::Kernel>>(self) -> Self::Out {
        Ok(Box::new(crate::device::Runner::<D>::new(self.0, self.1)?))
    }
}

fn check_rank<T: Transport>(plan: &Plan, transport: &T) -> Result<(), Error> {
    let fits = plan.ranks.max(1) == transport.ranks() && plan.rank == transport.rank();
    if !fits {
        return Err(Error::RankMismatch {
            plan: plan.rank,
            plans: plan.ranks,
            rank: transport.rank(),
            ranks: transport.ranks(),
        });
    }
    Ok(())
}

/// One rank: its runner and its way to the others.
pub struct Rank<T: Transport> {
    runner: Inner,
    transport: T,
    /// A persistent grid that runs the sums, if given.
    grid: Option<Box<dyn Sum>>,
    /// Host-fallback gather steps cached for one `plan.id`.
    host_gathers: (u64, Vec<usize>),
    /// Whether an async `queue` launched grid sums not yet checked.
    queued_sums: bool,
}

impl<T: Transport> Drop for Rank<T> {
    fn drop(&mut self) {
        // Stop any live persistent grid before dropping device buffers.
        if let Some(grid) = &mut self.grid {
            let _ = grid.pause();
        }
    }
}

impl<T: Transport> Rank<T> {
    /// A runner for the transport's rank of `plan`.
    pub fn open(plan: Plan, settings: &Settings, transport: T) -> Result<Self, Error> {
        check_rank(&plan, &transport)?;
        crate::configure(settings);
        let runner =
            match tessera_gpukit::backend::dispatch(settings.gpu(), OpenRank(plan, settings)) {
                Ok(run) => Inner::Device(run?),
                Err(OpenRank(plan, _)) => Inner::Plain(open(plan, settings)?),
            };
        Ok(Self {
            runner,
            transport,
            grid: None,
            host_gathers: (0, Vec::new()),
            queued_sums: false,
        })
    }

    /// A ranked runner on any `Device`, for tests.
    pub fn open_device<D>(plan: Plan, settings: &Settings, transport: T) -> Result<Self, Error>
    where
        D: Device + Launch<Kernel = <D as Device>::Kernel>,
    {
        check_rank(&plan, &transport)?;
        let runner = Inner::Device(Box::new(crate::device::Runner::<D>::new(plan, settings)?));
        Ok(Self {
            runner,
            transport,
            grid: None,
            host_gathers: (0, Vec::new()),
            queued_sums: false,
        })
    }

    /// Runs the sums through `grid` from now on.
    pub fn with_grid(mut self, grid: Option<Box<dyn Sum>>) -> Result<Self, Error> {
        carried(self.runner.run().plan(), grid.as_deref())?;
        self.grid = grid;
        Ok(self)
    }

    /// Whether sums run on the device.
    pub fn on_device(&self) -> bool {
        self.grid.is_some()
    }

    /// One crossing step through the persistent grid.
    fn grid_step(&mut self, i: usize, step: &Step) -> Result<bool, Error> {
        let at = |slot| self.runner.run().address(slot).ok_or(Error::Unaddressed(i));
        let (from, to) = (at(step.reads[0])?, at(step.write)?);
        let grid = self.grid.as_mut().expect("a grid");
        grid_cross(step, from, to, grid.as_mut())
    }

    /// Runs `steps` with crossing steps inside device graphs.
    fn submit_graphed(&mut self, steps: Range<usize>, wait: bool) -> Result<bool, Error> {
        let Inner::Device(dev) = &mut self.runner else {
            return Ok(false);
        };
        if !dev.can_graph(steps.clone()) {
            return Ok(false);
        }
        let plan_id = dev.plan().id;
        if self.host_gathers.0 != plan_id {
            self.host_gathers = (plan_id, Vec::new());
        }
        let mut from = steps.start;
        let mut used_grid = false;
        while from < steps.end {
            let next_host = self
                .host_gathers
                .1
                .iter()
                .copied()
                .find(|&at| at >= from && at < steps.end);
            let end = next_host.unwrap_or(steps.end);
            if from < end {
                let span = from..end;
                let has_cross = span
                    .clone()
                    .any(|i| crosses(self.runner.run().plan().steps[i].kernel.name()));
                let Inner::Device(dev) = &mut self.runner else {
                    unreachable!()
                };
                let grid = self.grid.as_mut().expect("a grid");
                let sync = wait || next_host.is_some();
                let split = match dev.run_with(span, sync, has_cross, grid.as_mut()) {
                    Ok(s) => s,
                    Err(e) => {
                        if used_grid || has_cross || std::mem::take(&mut self.queued_sums) {
                            let _ = grid.check();
                        }
                        return Err(e);
                    }
                };
                if let Some(at) = split {
                    self.host_gathers.1.push(at);
                    self.host_gathers.1.sort_unstable();
                    self.host_gathers.1.dedup();
                    continue;
                }
                used_grid |= has_cross;
            }
            if let Some(at) = next_host {
                let step = self.runner.run().plan().steps[at].clone();
                if let Err(e) = self.gather(&step) {
                    if used_grid || std::mem::take(&mut self.queued_sums) {
                        let _ = self.grid.as_mut().expect("a grid").check();
                    }
                    return Err(e);
                }
                from = at + 1;
            } else {
                from = end;
            }
        }
        if !wait {
            self.queued_sums |= used_grid;
        } else if used_grid || std::mem::take(&mut self.queued_sums) {
            self.grid.as_mut().expect("a grid").check()?;
        }
        Ok(true)
    }

    /// Runs a plain span, pausing the grid on cold capture.
    fn run_span(&mut self, span: Range<usize>, wait: bool) -> Result<(), Error> {
        if self.base().profile.is_none()
            && let (Inner::Device(dev), Some(grid)) = (&mut self.runner, self.grid.as_deref_mut())
        {
            return dev.run_with(span, wait, false, grid).map(|_| ());
        }
        match wait {
            true => self.runner.run_mut().submit(span),
            false => self.runner.run_mut().queue(span),
        }
    }

    /// A host sum. Profiling first waits for all ranks.
    fn exchange(&mut self, step: &Step) -> Result<(), Error> {
        let start = Instant::now();
        let part = self.runner.run().read(step.reads[0])?;
        let mut at = self.lap("all_reduce: read", start);
        if self.base().profile.is_some() {
            self.transport.exchange(&[])?;
            at = self.lap("all_reduce: wait", at);
        }
        let parts = self.transport.exchange(part.bytes())?;
        at = self.lap("all_reduce: transfer", at);
        let floats = |p: &Vec<u8>| Tensor::from_bytes(vec![p.len() / 4], DType::F32, p);
        let parts: Vec<Tensor> = parts.iter().map(floats).collect();
        let wire = wire(step);
        let sum = summed(parts.iter().map(|p| p.floats().expect("floats")), |x| {
            wire.round(x)
        });
        let sum = Tensor::f32(vec![step.write.len], sum);
        self.runner.run_mut().write(step.write, sum.view())?;
        self.lap("all_reduce: sum", at);
        self.lap("all_reduce", start);
        Ok(())
    }

    /// A host gather of each rank's rows.
    fn gather(&mut self, step: &Step) -> Result<(), Error> {
        let start = Instant::now();
        let part = self.runner.run().read(step.reads[0])?;
        let parts = self.transport.exchange(part.bytes())?;
        let rows = step.kernel.sizes()[0];
        let whole = Tensor::from_bytes(vec![step.write.len], part.dtype(), &gathered(&parts, rows));
        self.runner.run_mut().write(step.write, whole.view())?;
        self.lap("all_gather", start);
        Ok(())
    }

    /// Notes the seconds since `from` as `name`. Returns now.
    fn lap(&mut self, name: &str, from: Instant) -> Instant {
        let now = Instant::now();
        self.note(name, (now - from).as_secs_f64());
        now
    }
    /// Runs `steps`, omitting the trailing sync when `wait` is false.
    fn run_steps(&mut self, steps: Range<usize>, wait: bool) -> Result<(), Error> {
        let timed = self.base().profile.is_some();
        if let Some(grid) = &mut self.grid {
            grid.timing(timed);
        }
        if self.grid.as_ref().is_some_and(|g| g.capturable())
            && self.submit_graphed(steps.clone(), wait)?
        {
            if timed
                && wait
                && let Some(grid) = self.grid.as_mut()
                && let Some([waited, moved]) = grid.spent()?
            {
                self.note("all_reduce: wait", waited);
                self.note("all_reduce: transfer", moved);
            }
            return Ok(());
        }
        let mut from = steps.start;
        let mut sum_secs = 0.0f64;
        for i in steps.clone() {
            let step = &self.runner.run().plan().steps[i];
            if !crosses(step.kernel.name()) {
                continue;
            }
            let step = step.clone();
            self.run_span(from..i, !(self.grid.is_some() && !timed))?;
            from = i + 1;
            match (step.kernel.name() == AllGather::NAME, self.grid.is_some()) {
                (true, true) => {
                    if !self.grid_step(i, &step)? {
                        self.gather(&step)?;
                    }
                }
                (true, false) => self.gather(&step)?,
                (false, true) => {
                    let start = timed.then(Instant::now);
                    self.grid_step(i, &step)?;
                    if let Some(t0) = start
                        && self.grid.as_ref().is_some_and(|g| g.capturable())
                        && let Inner::Device(dev) = &self.runner
                    {
                        dev.finish_device()?;
                        sum_secs += t0.elapsed().as_secs_f64();
                    }
                }
                (false, false) => self.exchange(&step)?,
            }
        }
        let crossed = from > steps.start;
        if !wait && self.grid.is_some() && !timed {
            self.queued_sums |= crossed;
            return self.run_span(from..steps.end, false);
        }
        self.run_span(from..steps.end, true)?;
        let need_check = crossed || std::mem::take(&mut self.queued_sums);
        let Some(grid) = self.grid.as_mut().filter(|_| need_check) else {
            return Ok(());
        };
        grid.check()?;
        if let Some([wait, moved]) = grid.spent()? {
            self.note("all_reduce: wait", wait);
            self.note("all_reduce: transfer", moved);
            self.note("all_reduce", wait + moved);
        } else if timed && sum_secs > 0.0 {
            self.note("all_reduce", sum_secs);
        }
        Ok(())
    }
}

impl<T: Transport> Run for Rank<T> {
    fn base(&self) -> &Base {
        self.runner.run().base()
    }

    fn base_mut(&mut self) -> &mut Base {
        self.runner.run_mut().base_mut()
    }

    fn replan(&mut self, plan: &Plan) -> Result<(), Error> {
        carried(plan, self.grid.as_deref())?;
        match &mut self.runner {
            Inner::Device(dev) => dev.replan_with(plan, self.grid.as_deref_mut()),
            Inner::Plain(r) => r.replan(plan),
        }
    }

    fn upload_weight(
        &mut self,
        weight: u32,
        slot: tessera_lower::Slot,
        bytes: &[u8],
    ) -> Result<(), Error> {
        self.runner.run_mut().upload_weight(weight, slot, bytes)
    }

    /// Runs these steps. Sums wait for every rank.
    fn submit(&mut self, steps: Range<usize>) -> Result<(), Error> {
        self.run_steps(steps, true)
    }

    /// Queues steps without trailing sync when using a device grid.
    fn queue(&mut self, steps: Range<usize>) -> Result<(), Error> {
        self.run_steps(steps, false)
    }

    fn address(&self, slot: tessera_lower::Slot) -> Option<u64> {
        self.runner.run().address(slot)
    }

    fn copy_from(
        &mut self,
        dst: tessera_lower::Slot,
        dst_byte_off: usize,
        src_addr: u64,
        bytes: usize,
    ) -> Result<(), Error> {
        self.runner
            .run_mut()
            .copy_from(dst, dst_byte_off, src_addr, bytes)
    }

    fn launching(&mut self, how: Launching) {
        if let Some(grid) = &mut self.grid {
            let _ = grid.pause();
        }
        self.runner.run_mut().launching(how)
    }

    fn persist(&mut self, grid: Box<dyn crate::Grid>) -> Result<(), Error> {
        self.runner.run_mut().persist(grid)
    }

    fn profiling(&mut self, on: bool) {
        self.runner.run_mut().profiling(on)
    }

    fn take_profile(&mut self) -> Option<crate::Profile> {
        self.runner.run_mut().take_profile()
    }

    fn repeat(&mut self, step: usize, times: usize) -> Result<f64, Error> {
        self.runner.run_mut().repeat(step, times)
    }

    fn device(&self) -> String {
        self.runner.run().device()
    }

    fn compiled(&self) -> (usize, usize) {
        self.runner.run().compiled()
    }

    fn platform(&self) -> Platform {
        self.runner.run().platform()
    }

    fn precision(&self) -> Precision {
        self.runner.run().precision()
    }
}
