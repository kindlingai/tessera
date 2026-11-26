//! Replays a plan on any GPU over the device traits.

use crate::Error;
use crate::cache::{GraphKey, GraphSet, Lru, MAX_GRAPHS, MAX_PLANS, Prepared, store_graph};
use crate::grid::{self, Grid};
use crate::held::Holds;
use crate::kernels::Kernels;
use crate::lanes::LaneSync;
use crate::marks;
use crate::overrides::{self, Overrides};
use crate::paged::{self, Pager};
use crate::profile::Marks;
use crate::rank::{Capture, Sum, grid_cross};
use crate::run::{Base, Reload, Run};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ops::Range;
use std::path::PathBuf;
use std::time::Instant;
use tessera_codegen::Dialect;
use tessera_engine_settings::{Memory, Settings};
use tessera_gpukit::backend::OnGpu;
use tessera_gpukit::device::{Device, DeviceBuffer, cached_context, cached_source};
use tessera_gpukit::launch::{KernelArgs, Launch, launch_1d, launch_grid};
use tessera_lower::kernels::crosses;
use tessera_lower::{Arg, Extent, Launch as Call, Launching, Plan, Shape, Slot};
use tessera_mem::{Buffer, DType};

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
    /// In-graph GPU clock accumulators, made on first profiled graph.
    stamps: Option<marks::Stamps<D>>,
    /// Reloads that launches applied, for the next `refresh`.
    missed: Vec<Reload>,
    compiled: Kernels<D::Kernel>,
    /// Inactive plans' compiled tables, for fast switching.
    plans: Lru<u64, Prepared<D::Kernel>>,
    /// Active plan's bound device addresses, in buffer order.
    ptrs: Vec<u64>,
    /// Recorded step ranges per `(plan, bound)`, bounded by LRU.
    graphs: Lru<GraphKey, GraphSet<D::Graph>>,
    launching: Launching,
    /// Whether the device overlaps dependent launches at all.
    overlaps: bool,
    /// Which steps' kernels wait on the one before them.
    waits: Vec<bool>,
    /// Launches left before a closed gate opens. Zero when open.
    gated: Cell<usize>,
    /// Runs resident steps, once given.
    grid: RefCell<Option<Box<dyn Grid>>>,
    /// Which steps a persistent grid runs.
    resident: Vec<bool>,
    /// Hot-cold expert weight pager, opened on VRAM overflow.
    pager: Option<Pager<D>>,
}

/// Launches a closed gate holds, under any driver queue.
const HELD: usize = 256;

fn device_buf<D: Device>(buf: &Buffer) -> &DeviceBuffer<D> {
    buf.backing()
        .as_any()
        .downcast_ref::<DeviceBuffer<D>>()
        .expect("this runner allocates device buffers")
}

impl<D: Device + Launch<Kernel = <D as Device>::Kernel>> Runner<D> {
    pub fn new(plan: Plan, settings: &Settings) -> Result<Self, Error> {
        let ctx = cached_context::<D>()?;
        ctx.bind_numa();
        let files = |dir: &PathBuf| Overrides::new(dir.clone(), Dialect::of(D::BACKEND));
        let mut runner = Self {
            ctx,
            base: {
                let mut base = Base::new(settings.memory, &ctx.platform());
                base.streams = settings.weight_streaming;
                if settings.memory == Memory::Device {
                    base.extend_budget(tessera_gpukit::attic::available(settings));
                }
                base
            },
            kernels: Vec::new(),
            planned: Vec::new(),
            overrides: settings.kernels.as_ref().map(files),
            entries: Vec::new(),
            uses: Vec::new(),
            marked: HashMap::new(),
            ledger: None,
            stamps: None,
            missed: Vec::new(),
            compiled: Kernels::default(),
            plans: Lru::new(MAX_PLANS),
            ptrs: Vec::new(),
            graphs: Lru::new(MAX_GRAPHS),
            launching: Launching::default(),
            // Asked once: asking may launch, and captures come later.
            overlaps: ctx.platform().limits.pdl,
            waits: Vec::new(),
            gated: Cell::new(0),
            grid: RefCell::new(None),
            resident: Vec::new(),
            pager: None,
        };
        runner.replan(&plan)?;
        Ok(runner)
    }

    fn buffer(&self, slot: Slot) -> &DeviceBuffer<D> {
        device_buf::<D>(self.base.held.get(slot))
    }

    fn bound_ptrs(&self) -> Vec<u64> {
        self.base
            .held
            .bound()
            .map(|b| device_buf::<D>(b).device_ptr())
            .collect()
    }

    /// Destroys all recorded graphs after draining the stream.
    fn clear_graphs(&mut self) {
        if !self.graphs.is_empty() {
            let _ = self.ctx.finish();
            self.graphs.clear();
        }
    }

    /// Moves the active plan's tables into the plan cache.
    fn stash_plan(&mut self) {
        let id = self.base.plan.id;
        if id == 0 {
            return;
        }
        let mut plan = Plan::default();
        let mut launches = Vec::new();
        self.base.swap(&mut plan, &mut launches);
        let saved = Prepared {
            plan,
            launches,
            kernels: std::mem::take(&mut self.kernels),
            planned: std::mem::take(&mut self.planned),
            entries: std::mem::take(&mut self.entries),
            uses: std::mem::take(&mut self.uses),
            waits: std::mem::take(&mut self.waits),
            resident: std::mem::take(&mut self.resident),
        };
        self.plans.insert(id, saved);
    }

    /// Installs a cached plan's tables as active.
    fn restore_plan(&mut self, mut cached: Prepared<<D as Device>::Kernel>) {
        self.base.swap(&mut cached.plan, &mut cached.launches);
        self.kernels = cached.kernels;
        self.planned = cached.planned;
        self.entries = cached.entries;
        self.uses = cached.uses;
        self.waits = cached.waits;
        self.resident = cached.resident;
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
        self.ctx.programmatic(overlap && !self.resident[step]);
        let fired = match self.resident[step] {
            true => self.persistent(step, launch),
            false => {
                let mut args = self.bound(&launch.args)?;
                self.fire(&self.kernels[step], launch.shape, &mut args)
            }
        };
        fired.map_err(|e| {
            let name = self.base.plan.steps[step].kernel.name().to_string();
            Error::Launch(step, name, Box::new(e))
        })
    }

    /// One grid launch. It waits, so held launches go first.
    fn persistent(&self, step: usize, launch: &Call) -> Result<(), Error> {
        self.open();
        let plan = &self.base.plan;
        let base = |slot| self.buffer(slot).device_ptr();
        let (items, values) = grid::table(plan, &plan.steps[step], launch, &base)?;
        let kept = grid::kept_tables(plan, &base);
        let mut grid = self.grid.borrow_mut();
        let grid = grid.as_mut().ok_or(Error::Gridless(step))?;
        grid.launch(&plan.steps[step].kernel, &items, &kept, &values)
    }

    /// Opens the grid for each resident step of `plan`.
    fn ready(&mut self, plan: &Plan) -> Result<(), Error> {
        let Some(grid) = self.grid.get_mut().as_mut() else {
            return Ok(());
        };
        for (i, step) in plan.steps.iter().enumerate() {
            if self.resident[i] {
                grid.open(&step.kernel, grid::step_layout(plan, step))?;
            }
        }
        Ok(())
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

    fn grid_marks(&mut self) -> Vec<(String, Marks)> {
        self.grid
            .get_mut()
            .as_mut()
            .map_or_else(Vec::new, |g| g.take_marks())
    }

    /// Times each step by device stamps, else by waiting.
    fn profiled(&mut self, steps: Range<usize>) -> Result<(), Error> {
        self.close()?;
        let walked = self.walk(steps.clone());
        let (waited, stamps) = self.opened(walked)?;
        for (i, s) in steps.zip(waited.unwrap_or(stamps)) {
            self.note_step(i, s);
        }
        for (entry, found) in self.grid_marks() {
            for region in &found.regions {
                let (middle, _) = region.spread();
                let label = format!("mark {entry}: L{}-L{}", region.from, region.to);
                self.note(&label, middle);
            }
        }
        Ok(())
    }

    /// Runs `work`, optionally waiting. Returns the device's seconds.
    fn waited(
        &self,
        wait: bool,
        timed: bool,
        work: impl FnOnce(&Self) -> Result<(), Error>,
    ) -> Result<Option<f64>, Error> {
        if !wait || !timed {
            work(self)?;
            return Ok(if wait { self.ctx.finish()? } else { None });
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

    /// Drains the device stream.
    pub(crate) fn finish_ctx(&self) -> Result<(), Error> {
        self.ctx.finish().map(|_| ()).map_err(Into::into)
    }

    /// Whether `steps` can record into one device graph.
    pub(crate) fn can_record(&self, steps: &Range<usize>) -> bool {
        // A recorded graph cannot hold a grid that waits.
        self.launching.graphs
            && self.ctx.graphs()
            && !self.resident[steps.clone()].contains(&true)
            && self.base.plan.id != 0
    }

    fn flush_stamps(&mut self) -> Result<(), Error> {
        let Some(stamps) = &mut self.stamps else {
            return Ok(());
        };
        stamps.flush(self.ctx, &self.base.plan, self.base.profile.as_mut())
    }

    fn ensure_stamps(
        &mut self,
        len: usize,
        sums: &mut Option<&mut (dyn Sum + 'static)>,
    ) -> Result<(), Error> {
        let need = self.stamps.as_ref().is_none_or(|s| s.needs_grow(len));
        if !need {
            return Ok(());
        }
        self.flush_stamps()?;
        if let Some(grid) = sums.as_deref_mut() {
            grid.pause()?;
        }
        match &mut self.stamps {
            Some(stamps) => {
                stamps.grow(self.ctx, len)?;
                self.clear_graphs();
            }
            None => self.stamps = Some(marks::Stamps::open(self.ctx, len)?),
        }
        Ok(())
    }

    /// Promotes hot experts after a completed full-plan step.
    fn maintain_pager(&mut self, wait: bool, end: usize) -> Result<(), Error> {
        if wait
            && end == self.base.plan.steps.len()
            && let Some(pager) = &mut self.pager
        {
            pager.maintain(self.ctx)?;
        }
        Ok(())
    }

    /// Launches every step, as a recorded graph when eligible.
    pub(crate) fn run(
        &mut self,
        steps: Range<usize>,
        timed: bool,
        wait: bool,
        cross: bool,
        mut sums: Option<&mut (dyn Sum + 'static)>,
    ) -> Result<(Option<f64>, Option<usize>), Error> {
        if self.overrides.is_some()
            && let Some(grid) = sums.as_deref_mut()
        {
            grid.pause()?;
        }
        self.reread()?;
        let (end, profiled) = (steps.end, self.base.profile.is_some());
        if steps.is_empty() {
            self.maintain_pager(wait, end)?;
            return Ok((None, None));
        }
        if !self.can_record(&steps) {
            if profiled {
                self.profiled(steps)?;
                self.maintain_pager(wait, end)?;
                return Ok((None, None));
            }
            let plain = |r: &Self| {
                r.launch_all(steps, None, false)
                    .map_err(Capture::into_error)
            };
            let secs = self.waited(wait, timed, plain)?;
            self.maintain_pager(wait, end)?;
            return Ok((secs, None));
        }
        if profiled {
            self.ensure_stamps(steps.len(), &mut sums)?;
            if self
                .stamps
                .as_ref()
                .is_some_and(|s| s.differs(self.base.plan.id, steps.start, steps.end))
            {
                self.flush_stamps()?;
            }
        }
        let mut bound: Vec<_> = self.base.bound.iter().map(|(k, v)| (*k, *v)).collect();
        bound.sort_unstable();
        let (key, range) = (
            (self.base.plan.id, bound),
            (steps.start, steps.end, profiled),
        );
        let hit = self
            .graphs
            .get_mut(&key)
            .is_some_and(|s| s.ptrs == self.ptrs && s.ranges.contains_key(&range));
        if !hit {
            if let Some(grid) = sums.as_deref_mut() {
                grid.pause()?;
            }
            let graph = match self.ctx.capture(|| {
                let cap = if cross { sums.as_deref_mut() } else { None };
                self.launch_all(steps.clone(), cap, profiled)
            }) {
                Ok(graph) => graph,
                Err(Capture::Split(at)) => return Ok((None, Some(at))),
                Err(Capture::Err(e)) => return Err(e),
            };
            let evicted = store_graph(&mut self.graphs, key.clone(), range, &self.ptrs, graph);
            if evicted.is_some() {
                self.ctx.finish()?;
                drop(evicted);
            }
        }
        if cross && let Some(grid) = sums {
            grid.sum(0, 0, 0, DType::F32)?;
        }
        let replay = |r: &Self| {
            let graph = &r.graphs.get(&key).expect("captured above").ranges[&range];
            Ok(r.ctx.replay(graph)?)
        };
        let secs = self.waited(wait, timed, replay)?;
        if profiled && let Some(stamps) = &mut self.stamps {
            stamps.record(self.base.plan.id, steps.start, steps.end);
        }
        self.maintain_pager(wait, end)?;
        Ok((secs, None))
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
        // A served model's kernels need files to edit too.
        match self.write_files(plan) {
            Ok(written) if !written.is_empty() => {
                eprintln!("tessera: wrote {} kernel files", written.len())
            }
            Ok(_) => {}
            Err(e) => eprintln!("tessera: kernel files: {e}"),
        }
    }

    /// Writes `plan`'s kernel text as files, keeping existing ones.
    fn write_files(&self, plan: &Plan) -> Result<Vec<PathBuf>, Error> {
        let Some(files) = &self.overrides else {
            return Ok(Vec::new());
        };
        let mut written = Vec::new();
        for (entry, step) in &self.entries {
            let text = overrides::text(&plan.steps[*step], D::BACKEND);
            written.extend(files.write(entry, &text)?);
        }
        Ok(written)
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
            self.clear_graphs();
            self.apply();
            self.sync_grid();
        }
        Ok(reloads)
    }

    /// Passes profiling state and resident overrides to the grid.
    fn sync_grid(&mut self) {
        let Some(grid) = self.grid.get_mut().as_mut() else {
            return;
        };
        grid.profiling(self.base.profile.is_some());
        let Some(files) = &self.overrides else {
            return;
        };
        for (entry, step) in &self.entries {
            if self.resident[*step] {
                grid.replace(entry, files.text(entry).map(str::to_string));
            }
        }
    }

    /// Collects mark regions from a resident step launch.
    fn resident_marks(&mut self, step: usize) -> Result<Option<Marks>, Error> {
        if self.grid.get_mut().is_none() {
            return Ok(None);
        }
        let was_profiling = self.base.profile.is_some();
        if !was_profiling && let Some(grid) = self.grid.get_mut().as_mut() {
            grid.profiling(true);
        }
        let launches = self.base.launches(step)?;
        let mut ran = Ok(());
        for launch in &launches {
            if let Err(e) = self.persistent(step, launch) {
                ran = Err(e);
                break;
            }
        }
        let taken = self.grid_marks();
        if !was_profiling && let Some(grid) = self.grid.get_mut().as_mut() {
            grid.profiling(false);
        }
        ran?;
        Ok(Some(marks::merge(taken.into_iter().map(|(_, m)| m))))
    }

    /// Every launch of `steps`, in order.
    fn launch_all(
        &self,
        steps: Range<usize>,
        mut sums: Option<&mut (dyn Sum + 'static)>,
        stamped: bool,
    ) -> Result<(), Capture> {
        let sync = match self.ctx.has_lanes() {
            true => LaneSync::plan(&self.base.plan.steps[steps.clone()]),
            false => LaneSync::default(),
        };
        let stamps = stamped.then_some(self.stamps.as_ref()).flatten();
        let (start, len) = (steps.start, steps.len());
        if let Some(st) = stamps {
            st.emit(self.ctx, len, marks::BEGIN)?;
        }
        sync.begin(self.ctx)?;
        let run = (|| -> Result<(), Capture> {
            for i in steps {
                let step = &self.base.plan.steps[i];
                let rel = i - start;
                sync.before(self.ctx, rel, step.lane)?;
                if sync.active
                    && let Some(st) = stamps
                {
                    st.emit(self.ctx, rel, marks::LANE_BEGIN)?;
                }
                if crosses(step.kernel.name())
                    && let Some(grid) = sums.as_deref_mut()
                {
                    let at = |s: Slot| self.buffer(s).device_ptr() + s.at(0) as u64;
                    if !grid_cross(step, at(step.reads[0]), at(step.write), grid)? {
                        return Err(Capture::Split(i));
                    }
                } else {
                    for launch in self.base.launches(i)? {
                        self.launch(i, &launch)?;
                    }
                }
                if let Some(st) = stamps {
                    let mode = match (sync.active, rel + 1 == len) {
                        (true, _) => marks::LANE_END,
                        (false, true) => marks::LAST_STEP,
                        (false, false) => marks::STEP,
                    };
                    st.emit(self.ctx, rel, mode)?;
                }
                sync.after(self.ctx, rel, step.lane)?;
            }
            sync.finish(self.ctx)?;
            if sync.active
                && let Some(st) = stamps
            {
                st.emit(self.ctx, len, marks::END)?;
            }
            Ok(())
        })();
        if sync.active {
            self.ctx.set_lane(0);
        }
        run
    }

    /// Replans, pausing `sum` before any allocation or compilation.
    pub(crate) fn replan_with(
        &mut self,
        plan: &Plan,
        mut sum: Option<&mut (dyn Sum + 'static)>,
    ) -> Result<(), Error> {
        if plan.id != 0 && plan.id == self.base.plan.id {
            return Ok(());
        }
        self.flush_stamps()?;
        if self.pager.is_none()
            && self
                .base
                .budget
                .is_some_and(|l| plan.buffers.iter().map(|&b| b as u64).sum::<u64>() > l)
            && let Some(g) = sum.as_deref_mut()
        {
            g.pause()?;
        }
        let (plan, extra) = paged::prepare(&mut self.pager, self.ctx, plan, &self.base)?;
        let pager = self.pager.as_ref();
        let sized = |w, p| pager.and_then(|g| g.slot_bytes(w, p));
        let plan = self.base.held.reuse_weights(plan.as_ref(), sized);
        let plan = plan.as_ref();
        if !self.base.fresh_with(plan, extra)? {
            return Ok(());
        }
        let (ctx, unified) = (self.ctx, self.ctx.unified());
        let pager = self.pager.as_ref();
        let sized = |w, p| pager.and_then(|g| g.slot_bytes(w, p));
        let dropped = match self
            .base
            .held
            .bind_sized(plan, sized, |zone, bytes, holds| {
                if let Some(g) = sum.as_deref_mut() {
                    g.pause()?;
                }
                // Weights pin when unified. Kept rows borrow carveout when enabled.
                let buffer = match holds {
                    Holds::Weights if unified => DeviceBuffer::pinned(ctx, bytes)?,
                    Holds::Kept => DeviceBuffer::kept(ctx, bytes)?,
                    _ => DeviceBuffer::alloc(ctx, bytes)?,
                };
                // Kernels may read what they are about to write.
                buffer.zero()?;
                Ok(Buffer::new(zone, buffer))
            }) {
            Ok(dropped) => dropped,
            Err(e) => {
                if let Some(g) = sum.as_deref_mut() {
                    let _ = g.pause();
                }
                self.clear_graphs();
                return Err(e);
            }
        };
        if dropped {
            // Dropped buffers leave recorded device addresses stale.
            self.clear_graphs();
        }
        self.ptrs = self.bound_ptrs();
        if plan.id != 0
            && let Some(cached) = self.plans.remove(&plan.id)
        {
            self.stash_plan();
            self.restore_plan(cached);
            self.apply();
            self.sync_grid();
            self.ready(plan)?;
            return Ok(());
        }
        if let Some(g) = sum {
            g.pause()?;
        }
        let resident: Vec<bool> = plan
            .steps
            .iter()
            .map(|s| s.kernel.resident().is_some())
            .collect();
        let mut kernels = Vec::with_capacity(plan.steps.len());
        let mut waits = Vec::with_capacity(plan.steps.len());
        for step in &plan.steps {
            let fixed = |text: &'static str, entry: &'static str| -> Result<_, Error> {
                match text.contains("TS_MARK") {
                    true => Ok(ctx.compile(&format!("{}{text}", marks::OFF), entry)?),
                    false => Ok(cached_source::<D>(text, entry)?),
                }
            };
            let made = |s: &str, e: &str| -> Result<_, Error> { Ok(ctx.compile(s, e)?) };
            // Generated kernels wait on `griddepcontrol`. Fixed ones do not.
            let (kernel, generated) = self.compiled.get(step, D::BACKEND, fixed, made)?;
            kernels.push(kernel);
            waits.push(generated);
        }
        self.stash_plan();
        self.planned = kernels.clone();
        self.kernels = kernels;
        self.waits = waits;
        self.resident = resident;
        self.index(plan);
        self.apply();
        self.sync_grid();
        self.ready(plan)?;
        self.base.adopt(plan);
        Ok(())
    }
}

/// Opens a runner on the context type `on` names.
pub(crate) struct Open<'a>(pub(crate) Plan, pub(crate) &'a Settings);

impl OnGpu for Open<'_> {
    type Out = Result<Box<dyn Run>, Error>;

    fn on<D: Device + Launch<Kernel = <D as Device>::Kernel>>(self) -> Self::Out {
        Ok(Box::new(Runner::<D>::new(self.0, self.1)?))
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
        self.replan_with(plan, None)
    }

    fn refresh(&mut self) -> Result<Vec<Reload>, Error> {
        let mut all = std::mem::take(&mut self.missed);
        all.extend(self.reload()?);
        Ok(all)
    }

    fn marks(&mut self, step: usize) -> Result<Option<Marks>, Error> {
        if Dialect::of(D::BACKEND) == Dialect::Metal {
            return Ok(None);
        }
        if self.resident[step] {
            return self.resident_marks(step);
        }
        let source = self.base.plan.steps[step].kernel.source();
        let entry = overrides::entry(&source).to_string();
        let loaded = self.overrides.as_ref().and_then(|f| f.text(&entry));
        let planned = || overrides::text(&self.base.plan.steps[step], D::BACKEND);
        let text = loaded.map_or_else(planned, str::to_string);
        let launches = self.base.launches(step)?;
        let args: Result<Vec<_>, _> = launches.iter().map(|l| self.bound(&l.args)).collect();
        let mut args = args?;
        let ctx = self.ctx;
        let mut found = Marks::default();
        let mut ran = false;
        for (launch, a) in launches.iter().zip(&mut args) {
            let one = marks::step_marks(
                ctx,
                &mut self.marked,
                &mut self.ledger,
                (&entry, &text),
                |k| {
                    ctx.programmatic(false);
                    match launch.shape {
                        Shape::Line(n) => Ok(launch_1d(ctx, k, n, a)?),
                        Shape::Grid {
                            across,
                            down,
                            threads,
                        } => Ok(launch_grid(ctx, k, across, down, threads as u32, a)?),
                        Shape::Runtime(_) => Err(tessera_lower::Error::Unbound.into()),
                    }
                },
            )?;
            let Some(one) = one else {
                return Ok(None);
            };
            ran = true;
            found.lost += one.lost;
            found.regions.extend(one.regions);
        }
        Ok(ran.then_some(found).or_else(|| Some(Marks::default())))
    }

    fn dump(&self) -> Result<Vec<PathBuf>, Error> {
        self.write_files(&self.base.plan)
    }

    fn launching(&mut self, launching: Launching) {
        self.launching = launching;
        self.clear_graphs();
    }

    fn persist(&mut self, grid: Box<dyn Grid>) -> Result<(), Error> {
        *self.grid.get_mut() = Some(grid);
        self.sync_grid();
        let plan = self.base.plan.clone();
        self.ready(&plan)
    }

    fn profiling(&mut self, on: bool) {
        if !on && let Some(stamps) = &mut self.stamps {
            stamps.discard();
        }
        self.base.profile = on.then(crate::Profile::default);
        if let Some(grid) = self.grid.get_mut().as_mut() {
            grid.profiling(on);
        }
    }

    fn take_profile(&mut self) -> Option<crate::Profile> {
        let _ = self.flush_stamps();
        self.base.profile.as_mut().map(std::mem::take)
    }

    fn address(&self, slot: Slot) -> Option<u64> {
        Some(self.buffer(slot).device_ptr() + slot.at(0) as u64)
    }

    fn copy_from(
        &mut self,
        dst: Slot,
        dst_byte_off: usize,
        src_addr: u64,
        bytes: usize,
    ) -> Result<(), Error> {
        assert!(dst_byte_off + bytes <= dst.bytes(), "copy_from past slot");
        let at = self.buffer(dst).device_ptr() + (dst.at(0) + dst_byte_off) as u64;
        Ok(self.ctx.copy_d2d(at, src_addr, bytes)?)
    }

    fn repeat(&mut self, step: usize, times: usize) -> Result<f64, Error> {
        self.reread()?;
        self.ctx.finish()?;
        let start = Instant::now();
        let launches = self.base.launches(step)?;
        let clocked = self.waited(true, true, |r| {
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
        tessera_codegen::Precision::of(D::BACKEND)
    }

    fn upload_weight(&mut self, weight: u32, slot: Slot, bytes: &[u8]) -> Result<(), Error> {
        let desc_ptr = self.buffer(slot).device_ptr() + slot.at(0) as u64;
        if let Some(pager) = &mut self.pager
            && pager.upload(self.ctx, weight, slot, desc_ptr, bytes, &mut self.base.held)?
        {
            return Ok(());
        }
        self.held_mut().upload(slot, bytes)
    }

    fn submit(&mut self, steps: Range<usize>) -> Result<(), Error> {
        self.run(steps, false, true, false, None).map(|_| ())
    }

    fn queue(&mut self, steps: Range<usize>) -> Result<(), Error> {
        self.run(steps, false, false, false, None).map(|_| ())
    }

    fn clocked(&mut self, steps: Range<usize>) -> Result<Option<f64>, Error> {
        self.run(steps, true, true, false, None)
            .map(|(secs, _)| secs)
    }
}
