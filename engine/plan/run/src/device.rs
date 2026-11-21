//! Replays a plan on any GPU over the device traits.

use crate::Error;
use crate::kernels::{Kernels, dialect};
use crate::marks;
use crate::overrides::{self, Overrides};
use crate::profile::Marks;
use crate::run::{Base, Reload, Run};
use std::cell::Cell;
use std::collections::HashMap;
use std::ops::Range;
use std::path::PathBuf;
use std::time::Instant;
use tessera_codegen::Dialect;
use tessera_engine_settings::Settings;
use tessera_gpukit::device::{Device, DeviceBuffer, cached_context, cached_source};
use tessera_gpukit::launch::{KernelArgs, Launch, launch_1d, launch_grid};
use tessera_graph::DimName;
use tessera_lower::{Arg, Extent, Launch as Call, Launching, Plan, Shape, Slot};
use tessera_mem::Buffer;

pub(crate) struct Runner<D: Device> {
    ctx: &'static D,
    base: Base,
    /// Each step's running kernel: a file's, else the plan's.
    kernels: Vec<D::Kernel>,
    /// Each step's kernel as planned.
    planned: Vec<D::Kernel>,
    /// Kernel files in place of planned kernels, when set.
    overrides: Option<Overrides<D::Kernel>>,
    /// Each distinct entry, with its first step.
    entries: Vec<(String, usize)>,
    /// Each step's place in `entries`.
    uses: Vec<usize>,
    /// Kernels compiled with marks, by their marked text.
    marked: HashMap<String, D::Kernel>,
    /// Where marked kernels write, made on first use.
    ledger: Option<DeviceBuffer<D>>,
    /// Reloads that launches applied, for the next `refresh`.
    missed: Vec<Reload>,
    compiled: Kernels<D::Kernel>,
    /// The whole plan's launches per binding, recorded once.
    graphs: HashMap<Vec<(DimName, u32)>, D::Graph>,
    launching: Launching,
    /// Whether the device overlaps dependent launches at all.
    overlaps: bool,
    /// Which steps' kernels wait on the one before them.
    waits: Vec<bool>,
    /// Launches left before a closed gate opens. Zero when open.
    gated: Cell<usize>,
}

/// Launches a closed gate holds, under any driver queue.
const HELD: usize = 256;

impl<D: Device + Launch<Kernel = <D as Device>::Kernel>> Runner<D> {
    pub fn new(plan: Plan, settings: &Settings) -> Result<Self, Error> {
        let ctx = cached_context::<D>()?;
        let files = |dir: &PathBuf| Overrides::new(dir.clone(), dialect(D::BACKEND));
        let mut runner = Self {
            ctx,
            base: Base::new(settings.memory, &ctx.platform()),
            kernels: Vec::new(),
            planned: Vec::new(),
            overrides: settings.kernels.as_ref().map(files),
            entries: Vec::new(),
            uses: Vec::new(),
            marked: HashMap::new(),
            ledger: None,
            missed: Vec::new(),
            compiled: Kernels::default(),
            graphs: HashMap::new(),
            launching: Launching::default(),
            // Asked once: asking may launch, and captures come later.
            overlaps: ctx.platform().limits.pdl,
            waits: Vec::new(),
            gated: Cell::new(0),
        };
        runner.replan(&plan)?;
        Ok(runner)
    }

    fn buffer(&self, slot: Slot) -> &DeviceBuffer<D> {
        let backing = self.base.held.get(slot).backing().as_any();
        backing
            .downcast_ref::<DeviceBuffer<D>>()
            .expect("this runner allocates device buffers")
    }

    /// Each argument by position, in the device's form.
    fn bound(&self, args: &[Arg]) -> Result<KernelArgs, Error> {
        let mut bound = KernelArgs::new();
        for arg in args {
            match arg {
                Arg::Buffer(slot, Extent::Fixed(skip)) => {
                    let at = slot.at(*skip) as u64;
                    bound.device_ptr(self.buffer(*slot).device_ptr() + at)
                }
                Arg::Value(bytes) => bound.bytes(bytes),
                Arg::Size(_) | Arg::Buffer(..) => return Err(tessera_lower::Error::Unbound.into()),
            };
        }
        Ok(bound)
    }

    fn fire(
        &self,
        kernel: &<D as Device>::Kernel,
        shape: Shape,
        args: &mut KernelArgs,
    ) -> Result<(), Error> {
        match shape {
            Shape::Line(count) => launch_1d(self.ctx, kernel, count, args)?,
            Shape::Grid {
                across,
                down,
                threads,
            } => {
                let threads = threads as u32;
                launch_grid(self.ctx, kernel, across, down, threads, args)?
            }
            Shape::Runtime(_) => return Err(tessera_lower::Error::Unbound.into()),
        }
        Ok(())
    }

    /// Passes each argument by position, then enqueues.
    fn launch(&self, step: usize, launch: &Call) -> Result<(), Error> {
        let left = self.gated.get();
        if left == 1 {
            self.ctx.release();
        }
        self.gated.set(left.saturating_sub(1));
        let overlap = self.launching.pdl && self.overlaps && self.waits[step];
        self.ctx.programmatic(overlap);
        let mut args = self.bound(&launch.args)?;
        self.fire(&self.kernels[step], launch.shape, &mut args)
    }

    /// Holds the device queue for the next launches, if able.
    fn close(&self) -> Result<(), Error> {
        let held = self.ctx.gate()?;
        self.gated.set(if held { HELD } else { 0 });
        Ok(())
    }

    /// Lets held launches run.
    fn open(&self) {
        if self.gated.replace(0) > 0 {
            self.ctx.release();
        }
    }

    /// Each step's seconds by waiting, or `None` when stamped.
    fn walk(&self, steps: Range<usize>) -> Result<Option<Vec<f64>>, Error> {
        let stamped = self.ctx.stamp()?;
        let mut waited = Vec::new();
        for i in steps {
            let start = Instant::now();
            for launch in self.base.launches(i)? {
                self.launch(i, &launch)?;
            }
            match stamped {
                true => _ = self.ctx.stamp()?,
                false => {
                    let clocked = self.ctx.finish()?;
                    waited.push(clocked.unwrap_or_else(|| start.elapsed().as_secs_f64()));
                }
            }
        }
        Ok((!stamped).then_some(waited))
    }

    /// Opens the gate, then reads stamps. Errors discard stamps.
    fn opened<T>(&self, done: Result<T, Error>) -> Result<(T, Vec<f64>), Error> {
        self.open();
        match done {
            Ok(done) => Ok((done, self.ctx.stamps()?)),
            Err(e) => {
                _ = self.ctx.stamps();
                Err(e)
            }
        }
    }

    /// Times each step by device stamps, else by waiting.
    fn profiled(&mut self, steps: Range<usize>) -> Result<(), Error> {
        self.close()?;
        let walked = self.walk(steps.clone());
        let (waited, stamps) = self.opened(walked)?;
        for (i, s) in steps.zip(waited.unwrap_or(stamps)) {
            let name = self.base.plan.steps[i].kernel.name();
            self.note(name, s);
        }
        Ok(())
    }

    /// Runs `work`, then waits. Returns the device's seconds, if kept.
    fn waited(
        &self,
        timed: bool,
        work: impl FnOnce(&Self) -> Result<(), Error>,
    ) -> Result<Option<f64>, Error> {
        if !timed {
            work(self)?;
            return Ok(self.ctx.finish()?);
        }
        self.close()?;
        let stamped = self.ctx.stamp().map_err(Error::from).and_then(|stamped| {
            work(self)?;
            if stamped {
                self.ctx.stamp()?;
            }
            Ok(stamped)
        });
        match self.opened(stamped)? {
            (true, stamps) => Ok(stamps.first().copied()),
            (false, _) => Ok(self.ctx.finish()?),
        }
    }

    /// Launches every step, as a recorded graph when whole.
    fn run(&mut self, steps: Range<usize>, timed: bool) -> Result<Option<f64>, Error> {
        let whole = self.launching.graphs
            && self.ctx.graphs()
            && self.base.plan.id != 0
            && steps == (0..self.base.plan.steps.len());
        if !whole {
            return self.waited(timed, |r| r.launch_all(steps));
        }
        let mut key: Vec<(DimName, u32)> = self.base.bound.iter().map(|(k, v)| (*k, *v)).collect();
        key.sort_unstable();
        if !self.graphs.contains_key(&key) {
            let graph = self.ctx.capture(|| self.launch_all(steps.clone()))?;
            self.graphs.insert(key.clone(), graph);
        }
        self.waited(timed, |r| Ok(r.ctx.replay(&r.graphs[&key])?))
    }

    /// Notes each step's entry, once per distinct entry.
    fn index(&mut self, plan: &Plan) {
        (self.entries, self.uses) = (Vec::new(), Vec::new());
        if self.overrides.is_none() {
            return;
        }
        for (i, step) in plan.steps.iter().enumerate() {
            let source = step.kernel.source();
            let entry = overrides::entry(&source);
            let at = self.entries.iter().position(|(e, _)| e == entry);
            self.uses.push(at.unwrap_or(self.entries.len()));
            if at.is_none() {
                self.entries.push((entry.to_string(), i));
            }
        }
    }

    /// Gives each step its file's kernel, else the plan's.
    fn apply(&mut self) {
        let Some(files) = &self.overrides else {
            return;
        };
        for (i, kernel) in self.kernels.iter_mut().enumerate() {
            let entry = &self.entries[self.uses[i]].0;
            *kernel = files.kernel(entry).unwrap_or(&self.planned[i]).clone();
        }
    }

    /// Rereads changed files before a launch. Keeps what changed.
    fn reread(&mut self) -> Result<(), Error> {
        let found = self.reload()?;
        for reload in &found {
            eprintln!("tessera: {reload}");
        }
        self.missed.extend(found);
        Ok(())
    }

    /// Recompiles kernel files that changed. Returns what changed.
    fn reload(&mut self) -> Result<Vec<Reload>, Error> {
        let (ctx, plan) = (self.ctx, &self.base.plan);
        let Some(files) = self.overrides.as_mut() else {
            return Ok(Vec::new());
        };
        let mut reloads = Vec::new();
        for (entry, step) in &self.entries {
            let planned = || overrides::text(&plan.steps[*step], D::BACKEND);
            let compile = |text: &str| Ok(ctx.compile(&format!("{}{text}", marks::OFF), entry)?);
            reloads.extend(files.check(entry, planned, compile));
        }
        if !reloads.is_empty() {
            // Recorded launches name the kernels they replaced.
            self.graphs.clear();
            self.apply();
        }
        Ok(reloads)
    }

    /// Reads the ledger after one launch into `found`.
    fn collect(&self, ledger: u64, found: &mut Marks) -> Result<(), Error> {
        let mut bytes = vec![0u8; marks::COUNTS];
        self.ctx.download(ledger, &mut bytes)?;
        let counts: Vec<u32> = bytes
            .as_chunks()
            .0
            .iter()
            .map(|b| u32::from_ne_bytes(*b))
            .collect();
        let blocks = counts.iter().rposition(|c| *c > 0).map_or(0, |at| at + 1);
        let mut bytes = vec![0u8; marks::records(blocks)];
        self.ctx
            .download(ledger + marks::COUNTS as u64, &mut bytes)?;
        let records: Vec<u64> = bytes
            .as_chunks()
            .0
            .iter()
            .map(|b| u64::from_ne_bytes(*b))
            .collect();
        marks::read(&counts[..blocks], &records, self.ctx.ticks(), found);
        Ok(())
    }

    /// Every launch of `steps`, in order.
    fn launch_all(&self, steps: Range<usize>) -> Result<(), Error> {
        for i in steps {
            for launch in self.base.launches(i)? {
                self.launch(i, &launch)?;
            }
        }
        Ok(())
    }
}

/// Pinned host memory from the shared context.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) struct Pin<D: Device>(pub(crate) &'static D);

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl<D: Device> tessera_engine_comm::Pinning for Pin<D> {
    fn pin(&self, len: usize) -> Result<tessera_engine_comm::Pinned, tessera_engine_comm::Error> {
        let (host, device) = self
            .0
            .pin(len)
            .map_err(|e| tessera_engine_comm::Error::Link(e.to_string()))?;
        Ok(tessera_engine_comm::Pinned {
            host: host as *mut u8,
            device,
            len,
        })
    }

    fn unpin(&self, pinned: tessera_engine_comm::Pinned) {
        self.0.unpin(pinned.host as u64);
    }
}

impl<D: Device + Launch<Kernel = <D as Device>::Kernel>> Run for Runner<D> {
    fn base(&self) -> &Base {
        &self.base
    }

    fn base_mut(&mut self) -> &mut Base {
        &mut self.base
    }

    fn replan(&mut self, plan: &Plan) -> Result<(), Error> {
        if !self.base.fresh(plan)? {
            return Ok(());
        }
        // Buffers may move, so recorded launches go stale.
        self.graphs.clear();
        let ctx = self.ctx;
        self.base.held.bind(plan, |zone, bytes| {
            let buffer = DeviceBuffer::alloc(ctx, bytes)?;
            // Kernels may read what they are about to write.
            buffer.zero()?;
            Ok(Buffer::new(zone, buffer))
        })?;
        self.kernels.clear();
        self.waits.clear();
        for step in &plan.steps {
            let fixed = |text, entry| -> Result<_, Error> { Ok(cached_source::<D>(text, entry)?) };
            let made =
                |source: &str, entry: &str| -> Result<_, Error> { Ok(ctx.compile(source, entry)?) };
            // Generated kernels wait on `griddepcontrol`. Fixed ones do not.
            let (kernel, generated) = self.compiled.get(step, D::BACKEND, fixed, made)?;
            self.kernels.push(kernel);
            self.waits.push(generated);
        }
        self.planned = self.kernels.clone();
        self.index(plan);
        self.apply();
        self.base.adopt(plan);
        Ok(())
    }

    fn refresh(&mut self) -> Result<Vec<Reload>, Error> {
        let mut all = std::mem::take(&mut self.missed);
        all.extend(self.reload()?);
        Ok(all)
    }

    fn marks(&mut self, step: usize) -> Result<Option<Marks>, Error> {
        if dialect(D::BACKEND) == Dialect::Metal {
            return Ok(None);
        }
        let source = self.base.plan.steps[step].kernel.source();
        let entry = overrides::entry(&source).to_string();
        let loaded = self.overrides.as_ref().and_then(|f| f.text(&entry));
        let planned = || overrides::text(&self.base.plan.steps[step], D::BACKEND);
        let text = loaded.map_or_else(planned, str::to_string);
        if !text.contains("TS_MARK") {
            return Ok(Some(Marks::default()));
        }
        let marked = format!("{}{text}", marks::ON);
        let kernel = match self.marked.get(&marked) {
            Some(kernel) => kernel.clone(),
            None => {
                let kernel = self.ctx.compile(&marked, &entry)?;
                self.marked.insert(marked, kernel.clone());
                kernel
            }
        };
        let Some(at) = self.ctx.global(&kernel, "tessera_marks")? else {
            return Ok(None);
        };
        if self.ledger.is_none() {
            let bytes = marks::COUNTS + marks::records(marks::BLOCKS);
            self.ledger = Some(DeviceBuffer::alloc(self.ctx, bytes)?);
        }
        let ledger = self.ledger.as_ref().map_or(0, DeviceBuffer::device_ptr);
        self.ctx.finish()?;
        self.ctx.upload(at, &ledger.to_ne_bytes())?;
        let mut found = Marks::default();
        // Counts restart per launch, so launches never mix.
        for launch in self.base.launches(step)? {
            self.ctx.zero(ledger, marks::COUNTS)?;
            self.ctx.programmatic(false);
            let mut args = self.bound(&launch.args)?;
            self.fire(&kernel, launch.shape, &mut args)?;
            self.ctx.finish()?;
            self.collect(ledger, &mut found)?;
        }
        Ok(Some(found))
    }

    fn dump(&self) -> Result<Vec<PathBuf>, Error> {
        let Some(files) = &self.overrides else {
            return Ok(Vec::new());
        };
        let mut written = Vec::new();
        for (entry, step) in &self.entries {
            let text = overrides::text(&self.base.plan.steps[*step], D::BACKEND);
            written.extend(files.write(entry, &text)?);
        }
        Ok(written)
    }

    fn launching(&mut self, launching: Launching) {
        self.launching = launching;
        self.graphs.clear();
    }

    fn address(&self, slot: Slot) -> Option<u64> {
        Some(self.buffer(slot).device_ptr() + slot.at(0) as u64)
    }

    /// Hand-written kernels are CUDA text.
    fn fixed(&self) -> bool {
        dialect(D::BACKEND) == Dialect::Cuda
    }

    fn launch_fixed(
        &self,
        source: &'static str,
        entry: &'static str,
        args: &[Arg],
        shape: Shape,
    ) -> Result<(), Error> {
        let kernel = cached_source::<D>(source, entry)?;
        self.ctx.programmatic(false);
        let mut args = self.bound(args)?;
        self.fire(&kernel, shape, &mut args)?;
        self.ctx.finish()?;
        Ok(())
    }

    fn repeat(&mut self, step: usize, times: usize) -> Result<f64, Error> {
        self.reread()?;
        self.ctx.finish()?;
        let start = Instant::now();
        let launches = self.base.launches(step)?;
        let clocked = self.waited(true, |r| {
            for _ in 0..times {
                for launch in &launches {
                    r.launch(step, launch)?;
                }
            }
            Ok(())
        })?;
        let seconds = clocked.unwrap_or_else(|| start.elapsed().as_secs_f64());
        Ok(seconds / times as f64)
    }

    fn device(&self) -> String {
        format!("{} {}", D::BACKEND, self.ctx.name())
    }

    fn platform(&self) -> tessera_gpukit::Platform {
        self.ctx.platform()
    }

    fn precision(&self) -> tessera_codegen::Precision {
        crate::precision(D::BACKEND)
    }

    fn submit(&mut self, steps: Range<usize>) -> Result<(), Error> {
        self.reread()?;
        if self.base.profile.is_some() {
            return self.profiled(steps);
        }
        self.run(steps, false).map(|_| ())
    }

    fn clocked(&mut self, steps: Range<usize>) -> Result<Option<f64>, Error> {
        self.reread()?;
        if self.base.profile.is_some() {
            return self.profiled(steps).map(|_| None);
        }
        self.run(steps, true)
    }
}
