//! Replays a plan on any GPU over the device traits.

use crate::cache::{GraphKey, GraphSet, Lru, MAX_GRAPHS, MAX_PLANS, Prepared, store_graph};
use crate::grid::{self, Grid};
use crate::kernels::{self, Kernels};
use crate::lanes::LaneSync;
use crate::marks;
use crate::overrides::{self, Overrides};
use crate::paged::{KvMem, Pager};
use crate::profile::Marks;
use crate::rank::{Capture, Sum, grid_cross};
use crate::run::{Base, Reload, Run};
use crate::{Backed, Error, HostPool};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::PathBuf;
use std::time::Instant;
use tessera_codegen::Dialect;
use tessera_engine_settings::{Memory, Settings};
use tessera_gpukit::backend::OnGpu;
use tessera_gpukit::device::{Device, DeviceBuffer, KernelAttrs, cached_context};
use tessera_gpukit::launch::{KernelArgs, Launch, launch_1d, launch_grid};
use tessera_graph::DimName;
use tessera_lower::kernels::crosses;
use tessera_lower::{Arg, Extent, Launch as Call, Launching, Plan, Shape, Slot};
use tessera_mem::{Buffer, DType};

mod replan;
mod timing;

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
    /// Drained graphs waiting on stream sync before drop.
    evicted: Vec<D::Graph>,
    /// Replaced buffers waiting on graph eviction and sync.
    retired: Vec<Buffer>,
    /// Set when a failure left bindings or retirement incomplete.
    dirty: bool,
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
    /// Compiled resource counts and occupancy for each step.
    attrs: Vec<Option<KernelAttrs>>,
    /// Runtime dims read by each cached step range.
    range_dims: HashMap<(usize, usize), Vec<DimName>>,
    /// Whether load-time spill notices print without active profiling.
    log_attrs: bool,
    /// Spilling kernel entries already reported on load.
    spills: HashSet<String>,
    /// Hot-cold expert weight pager, opened on VRAM overflow.
    pager: Option<Pager<D>>,
    /// Pinned host staging and `TIER_HOST` KV pool.
    kv: KvMem<D>,
}

/// Launches a closed gate holds, under any driver queue.
const HELD: usize = 256;

fn device_buf<D: Device>(buf: &Buffer) -> &DeviceBuffer<D> {
    buf.backing()
        .as_any()
        .downcast_ref::<DeviceBuffer<D>>()
        .expect("this runner allocates device buffers")
}

/// Launches `kernel` with a fixed shape.
fn fire_shape<D: Device + Launch<Kernel = <D as Device>::Kernel>>(
    ctx: &D,
    kernel: &<D as Device>::Kernel,
    shape: Shape,
    shared: u32,
    args: &mut KernelArgs,
) -> Result<(), Error> {
    match shape {
        Shape::Line(count) => launch_1d(ctx, kernel, count, args)?,
        Shape::Grid {
            across,
            down,
            threads,
        } => launch_grid(ctx, kernel, across, down, threads as u32, shared, args)?,
        Shape::Runtime(_) => return Err(tessera_lower::Error::Unbound.into()),
    }
    Ok(())
}

impl<D: Device + Launch<Kernel = <D as Device>::Kernel>> Runner<D> {
    pub fn new(plan: Plan, settings: &Settings) -> Result<Self, Error> {
        let ctx = cached_context::<D>()?;
        ctx.bind_numa();
        let files = |dir: &PathBuf| Overrides::new(dir.clone(), D::BACKEND);
        let mut base = Base::new(settings.memory, &ctx.platform());
        base.streams = settings.weight_streaming;
        base.stages = settings.weight_staging;
        if settings.memory == Memory::Device {
            base.extend_budget(tessera_gpukit::attic::available(settings));
        }
        let mut runner = Self {
            ctx,
            base,
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
            evicted: Vec::new(),
            retired: Vec::new(),
            dirty: false,
            launching: Launching::default(),
            // Asked once: asking may launch, and captures come later.
            overlaps: ctx.platform().limits.pdl,
            waits: Vec::new(),
            gated: Cell::new(0),
            grid: RefCell::new(None),
            resident: Vec::new(),
            attrs: Vec::new(),
            range_dims: HashMap::new(),
            log_attrs: settings.profile.is_some() || settings.kernels.is_some(),
            spills: HashSet::new(),
            pager: None,
            kv: KvMem::default(),
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

    /// Each argument by position, in the device's form.
    fn bound(&self, args: &[Arg]) -> Result<KernelArgs, Error> {
        let mut bound = KernelArgs::new();
        for arg in args {
            match arg {
                Arg::Buffer(slot, Extent::Fixed(skip)) => {
                    let at = slot.at(*skip) as u64;
                    bound.device_ptr(self.buffer(*slot).device_ptr() + at)
                }
                Arg::KeptHost(slot) => bound.device_ptr(self.kv.host_ptr(*slot)),
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
        shared: u32,
        args: &mut KernelArgs,
    ) -> Result<(), Error> {
        fire_shape(self.ctx, kernel, shape, shared, args)
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
            true => {
                if self.override_launch(step).is_some_and(|o| !o.is_empty()) {
                    Err(Error::Grid(
                        "launch.toml overrides are unsupported on resident kernels".into(),
                    ))
                } else {
                    self.persistent(step, launch)
                }
            }
            false => {
                let base_shared = self.base.plan.steps[step].kernel.needs().dynamic_shared as u32;
                let (shape, shared) =
                    overrides::apply_launch(launch.shape, base_shared, self.override_launch(step));
                let mut args = self.bound(&launch.args)?;
                self.fire(&self.kernels[step], shape, shared, &mut args)
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
        self.check_ready()?;
        self.reread_with(sums.as_deref_mut())?;
        let (end, profiled) = (steps.end, self.base.profile.is_some());
        if let Some(pager) = &mut self.pager {
            let ends = end >= self.base.plan.forward().end;
            pager.enter(self.ctx, steps.start, ends)?;
        }
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
            self.ensure_stamps(&steps, &mut sums)?;
            if self
                .stamps
                .as_ref()
                .is_some_and(|s| s.differs(self.base.plan.id, steps.start, steps.end))
            {
                self.flush_stamps()?;
            }
        }
        let dims = self
            .range_dims
            .entry((steps.start, steps.end))
            .or_insert_with(|| self.base.step_dims(steps.clone()));
        let bound: Vec<_> = dims
            .iter()
            .filter_map(|&name| self.base.bound.get(&name).map(|&v| (name, v)))
            .collect();
        let (key, range) = (
            (self.base.plan.id, bound),
            (steps.start, steps.end, profiled),
        );
        let hit = self
            .graphs
            .get_mut(&key)
            .is_some_and(|s| s.ptrs == self.ptrs && s.ranges.contains_key(&range));
        if !hit {
            let cap_start = profiled.then(Instant::now);
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
            if let Some(evicted) =
                store_graph(&mut self.graphs, key.clone(), range, &self.ptrs, graph)
            {
                self.evicted.extend(evicted.into_values());
                if let Err(e) = self.finish_evicted() {
                    self.dirty = true;
                    return Err(e);
                }
            }
            if let Some(t0) = cap_start {
                self.note_cold(
                    "plan: capture",
                    t0,
                    format_args!(
                        "plan={} steps={}..{} bound={:?}",
                        key.0, steps.start, steps.end, key.1
                    ),
                );
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

    /// Rereads changed files before a launch, pausing `sum` if needed.
    fn reread_with(&mut self, sum: Option<&mut (dyn Sum + 'static)>) -> Result<(), Error> {
        let found = self.reload_with(sum)?;
        for reload in &found {
            kernels::log_reload(reload, &self.entries, &self.attrs);
        }
        self.missed.extend(found);
        Ok(())
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
        let order = marks::slots(&self.base.plan, steps.clone());
        let sync = match self.ctx.has_lanes() {
            true => LaneSync::plan(order.iter().map(|(.., s)| *s)),
            false => LaneSync::default(),
        };
        let stamps = stamped.then_some(self.stamps.as_ref()).flatten();
        let len = steps.len();
        if let Some(st) = stamps {
            st.emit(self.ctx, len, marks::BEGIN)?;
        }
        sync.begin(self.ctx)?;
        let run = (|| -> Result<(), Capture> {
            for (at, &(rel, i, step)) in order.iter().enumerate() {
                sync.before(self.ctx, at, step.lane)?;
                let Some(i) = i else {
                    self.stage(step, stamps.map(|st| (st, rel)))?;
                    sync.after(self.ctx, at, step.lane)?;
                    continue;
                };
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
                sync.after(self.ctx, at, step.lane)?;
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

    fn stage_weights(&mut self, plan: &Plan) -> Result<(), Error> {
        let (ctx, uni) = (self.ctx, self.ctx.unified());
        self.base.held.stage_weights(plan, |zone, bytes, h| {
            Ok(Buffer::new(
                zone,
                crate::held::alloc_buf(ctx, uni, bytes, h)?,
            ))
        })?;
        self.base.record_extra_weights(plan);
        let mut base = self.base.plan.clone();
        base.id = Plan::fresh();
        self.replan(&base)
    }

    fn refresh(&mut self) -> Result<Vec<Reload>, Error> {
        self.check_ready()?;
        let mut all = std::mem::take(&mut self.missed);
        all.extend(self.reload_with(None)?);
        Ok(all)
    }

    fn marks(&mut self, step: usize) -> Result<Option<Marks>, Error> {
        self.check_ready()?;
        if Dialect::of(D::BACKEND) == Dialect::Metal {
            return Ok(None);
        }
        if self.resident[step] {
            return self.resident_marks(step);
        }
        let source = self.base.plan.steps[step].kernel.source();
        let entry = overrides::entry(&source).to_string();
        let over = self
            .overrides
            .as_ref()
            .and_then(|f| f.launch(&entry))
            .cloned();
        let flags = over
            .as_ref()
            .map_or_else(Vec::new, overrides::LaunchOverride::flags);
        let loaded = self.overrides.as_ref().and_then(|f| f.text(&entry));
        let planned = || overrides::text(&self.base.plan.steps[step], D::BACKEND);
        let text = loaded.map_or_else(planned, str::to_string);
        let launches = self.base.launches(step)?;
        let args: Result<Vec<_>, _> = launches.iter().map(|l| self.bound(&l.args)).collect();
        let mut args = args?;
        let ctx = self.ctx;
        let mut found = Marks::default();
        let shared = self.base.plan.steps[step].kernel.needs().dynamic_shared as u32;
        let mut ran = false;
        for (launch, a) in launches.iter().zip(&mut args) {
            let (shape, sh) = overrides::apply_launch(launch.shape, shared, over.as_ref());
            let one = marks::step_marks(
                ctx,
                &mut self.marked,
                &mut self.ledger,
                (&entry, &text, &flags),
                |k| {
                    ctx.programmatic(false);
                    fire_shape(ctx, k, shape, sh, a)
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

    fn step_attrs(&self, step: usize) -> Option<KernelAttrs> {
        self.attrs.get(step).copied().flatten()
    }

    fn dump(&self) -> Result<Vec<PathBuf>, Error> {
        self.write_files(&self.base.plan)
    }

    fn launching(&mut self, launching: Launching) {
        self.launching = launching;
        self.drop_graphs();
    }

    fn persist(&mut self, grid: Box<dyn Grid>) -> Result<(), Error> {
        self.check_ready()?;
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

    fn check_ready(&self) -> Result<(), Error> {
        Runner::check_ready(self)
    }

    fn address(&self, slot: Slot) -> Option<u64> {
        Runner::check_ready(self).ok()?;
        Some(self.buffer(slot).device_ptr() + slot.at(0) as u64)
    }

    fn copy_from(
        &mut self,
        dst: Slot,
        dst_byte_off: usize,
        src_addr: u64,
        bytes: usize,
    ) -> Result<(), Error> {
        self.check_ready()?;
        assert!(dst_byte_off + bytes <= dst.bytes(), "copy_from past slot");
        let at = self.buffer(dst).device_ptr() + (dst.at(0) + dst_byte_off) as u64;
        Ok(self.ctx.copy_d2d(at, src_addr, bytes)?)
    }

    fn repeat(&mut self, step: usize, times: usize) -> Result<f64, Error> {
        self.check_ready()?;
        self.reread_with(None)?;
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
        self.check_ready()?;
        let desc_ptr = self.buffer(slot).device_ptr() + slot.at(0) as u64;
        if let Some(pager) = &mut self.pager
            && pager.upload(self.ctx, weight, slot, desc_ptr, bytes, &mut self.base.held)?
        {
            return Ok(());
        }
        self.held_mut().upload(slot, bytes)
    }

    fn download_kept_pages(&mut self, p: &[u32], g: usize, dst: &mut [u8]) -> Result<bool, Error> {
        self.check_ready()?;
        let (ctx, st, b) = (self.ctx, &mut self.kv.stage, &self.base);
        b.held.download_pages(ctx, st, &b.plan, p, g, dst)
    }

    fn upload_kept_pages(&mut self, p: &[u32], g: usize, src: &[u8]) -> Result<bool, Error> {
        self.check_ready()?;
        let (ctx, st, b) = (self.ctx, &mut self.kv.stage, &mut self.base);
        b.held.upload_pages(ctx, st, &b.plan, p, g, src)
    }

    fn open_host_kv(&mut self, pages: usize, page: usize, by: Backed) -> Result<HostPool, Error> {
        self.check_ready()?;
        let plan = &self.base.plan;
        let ok = self.kv.open_host(self.ctx, plan, pages, page, by);
        self.clear_graphs()?;
        Ok(ok)
    }

    fn move_kv_pages(&mut self, kind: tessera_kv::Move, p: &[(u32, u32)]) -> Result<bool, Error> {
        self.check_ready()?;
        self.kv
            .move_pages(&self.base.held, &self.base.plan, kind, p)
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
