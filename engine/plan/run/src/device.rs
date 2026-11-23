//! Replays a plan on any GPU over the device traits.

use crate::Error;
use crate::cache::Lru;
use crate::grid::{self, Grid};
use crate::held::Holds;
use crate::kernels::Kernels;
use crate::marks;
use crate::overrides::{self, Overrides};
use crate::profile::Marks;
use crate::run::{Base, Reload, Run};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use tessera_codegen::Dialect;
use tessera_engine_settings::Settings;
use tessera_gpukit::backend::OnGpu;
use tessera_gpukit::device::{Device, DeviceBuffer, cached_context, cached_source};
use tessera_gpukit::launch::{KernelArgs, Launch, launch_1d, launch_grid};
use tessera_graph::DimName;
use tessera_lower::{Arg, Extent, Launch as Call, Launching, Plan, Shape, Slot, Step};
use tessera_mem::Buffer;
use tessera_mem::form::Read;

/// Most inactive plans kept ready for instant switching.
const MAX_PLANS: usize = 32;

/// Most `(plan, bound)` graph sets kept alive at once.
const MAX_GRAPHS: usize = 32;

/// Plan id and bound dims sharing one graph set.
type GraphKey = (u64, Vec<(DimName, u32)>);

/// Recorded step ranges for one binding and its addresses.
struct GraphSet<G> {
    ptrs: Vec<u64>,
    ranges: HashMap<(usize, usize), G>,
}

/// Compiled kernel and dispatch tables for one inactive plan.
struct Prepared<K> {
    plan: Plan,
    launches: Vec<OnceLock<Vec<Call>>>,
    kernels: Vec<K>,
    planned: Vec<K>,
    entries: Vec<(String, usize)>,
    uses: Vec<usize>,
    waits: Vec<bool>,
    resident: Vec<bool>,
}

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
        let files = |dir: &PathBuf| Overrides::new(dir.clone(), Dialect::of(D::BACKEND));
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
        let (items, values) = table(plan, &plan.steps[step], launch, &base)?;
        let kept = kept_tables(plan, &base);
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
                grid.open(&step.kernel, step_layout(plan, step))?;
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

    /// Times each step by device stamps, else by waiting.
    fn profiled(&mut self, steps: Range<usize>) -> Result<(), Error> {
        self.close()?;
        let walked = self.walk(steps.clone());
        let (waited, stamps) = self.opened(walked)?;
        for (i, s) in steps.zip(waited.unwrap_or(stamps)) {
            let name = self.base.plan.steps[i].kernel.name();
            self.note(name, s);
        }
        let grid_marks = self
            .grid
            .get_mut()
            .as_mut()
            .map_or_else(Vec::new, |g| g.take_marks());
        for (entry, found) in grid_marks {
            for region in &found.regions {
                let (middle, _) = region.spread();
                let label = format!("mark {entry}: L{}-L{}", region.from, region.to);
                self.note(&label, middle);
            }
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

    /// Launches every step, as a recorded graph when eligible.
    fn run(&mut self, steps: Range<usize>, timed: bool, wait: bool) -> Result<Option<f64>, Error> {
        if steps.is_empty() {
            return Ok(None);
        }
        // A recorded graph cannot hold a grid that waits.
        let record = self.launching.graphs
            && self.ctx.graphs()
            && !self.resident[steps.clone()].contains(&true)
            && self.base.plan.id != 0;
        if !record {
            if !wait {
                self.launch_all(steps)?;
                return Ok(None);
            }
            return self.waited(timed, |r| r.launch_all(steps));
        }
        let mut bound: Vec<(DimName, u32)> =
            self.base.bound.iter().map(|(k, v)| (*k, *v)).collect();
        bound.sort_unstable();
        let key = (self.base.plan.id, bound);
        let range = (steps.start, steps.end);
        let hit = self
            .graphs
            .get_mut(&key)
            .is_some_and(|s| s.ptrs == self.ptrs && s.ranges.contains_key(&range));
        if !hit {
            let graph = self.ctx.capture(|| self.launch_all(steps.clone()))?;
            let evicted = match self.graphs.get_mut(&key) {
                Some(set) => {
                    let stale = if set.ptrs != self.ptrs {
                        set.ptrs = self.ptrs.clone();
                        Some(std::mem::take(&mut set.ranges))
                    } else {
                        None
                    };
                    set.ranges.insert(range, graph);
                    stale
                }
                None => {
                    let mut ranges = HashMap::new();
                    ranges.insert(range, graph);
                    let set = GraphSet {
                        ptrs: self.ptrs.clone(),
                        ranges,
                    };
                    self.graphs.insert(key.clone(), set).map(|old| old.ranges)
                }
            };
            if evicted.is_some() {
                self.ctx.finish()?;
            }
            drop(evicted);
        }
        if !wait {
            let graph = &self.graphs.get(&key).expect("captured above").ranges[&range];
            self.ctx.replay(graph)?;
            return Ok(None);
        }
        self.waited(timed, |r| {
            let graph = &r.graphs.get(&key).expect("captured above").ranges[&range];
            Ok(r.ctx.replay(graph)?)
        })
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
        let taken = self
            .grid
            .get_mut()
            .as_mut()
            .map_or_else(Vec::new, |g| g.take_marks());
        if !was_profiling && let Some(grid) = self.grid.get_mut().as_mut() {
            grid.profiling(false);
        }
        ran?;
        let mut merged = Marks::default();
        for (_, found) in taken {
            merged.lost += found.lost;
            for region in found.regions {
                match merged
                    .regions
                    .iter_mut()
                    .find(|r| (r.from, r.to) == (region.from, region.to))
                {
                    Some(existing) => existing.seconds.extend(region.seconds),
                    None => merged.regions.push(region),
                }
            }
        }
        Ok(Some(merged))
    }

    /// Reads the ledger after one launch into `found`.
    fn collect(&self, ledger: u64, found: &mut Marks) -> Result<(), Error> {
        Ok(marks::collect(self.ctx, ledger, found)?)
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

/// Matrix rows and columns from a packed shape.
fn matrix_dims(dims: u32, shape: [u32; 4]) -> (u32, u32) {
    match dims {
        0 => (0, 0),
        1 => (1, shape[0]),
        d => (shape[d as usize - 2], shape[d as usize - 1]),
    }
}

/// Companion `suh` and `svh` device addresses for a rotated weight.
fn companions(
    plan: &Plan,
    weight: u32,
    address: &impl Fn(Slot) -> u64,
) -> (Option<u64>, Option<u64>) {
    let mut suh = None;
    let mut svh = None;
    for l in &plan.weights {
        if l.weight != weight {
            continue;
        }
        let at = address(l.slot) + l.slot.start() as u64;
        match l.part {
            Some(1) => suh = Some(at),
            Some(2) => svh = Some(at),
            _ => {}
        }
    }
    (suh, svh)
}

/// A bound launch as items and values, by argument order.
fn table(
    plan: &Plan,
    step: &Step,
    launch: &Call,
    address: &impl Fn(Slot) -> u64,
) -> Result<(Vec<grid::Item>, Vec<u32>), Error> {
    let (mut items, mut values) = (Vec::new(), Vec::new());
    let mut buf_index = 0usize;
    for arg in &launch.args {
        match arg {
            Arg::Buffer(slot, Extent::Fixed(skip)) => {
                let kind = if buf_index == step.reads.len() && *slot == step.write {
                    grid::ItemKind::Output
                } else {
                    grid::ItemKind::Input
                };
                let at = address(*slot) + slot.at(*skip) as u64;
                let bytes = slot.bytes() - (slot.at(*skip) - slot.start());
                let (dims, shape) = if *skip == 0 && slot.dims > 0 {
                    (u32::from(slot.dims), slot.shape)
                } else {
                    (1, [bytes as u32, 0, 0, 0])
                };
                let loaded = plan
                    .weights
                    .iter()
                    .find(|l| l.slot.buffer == slot.buffer && l.slot.offset == slot.offset);
                let share = match loaded.and_then(|l| l.share) {
                    Some(s) => [s.axis, s.first as u32, s.count as u32, s.whole as u32],
                    None => [0; 4],
                };
                let form = if slot.form.read() == Read::Decode {
                    let (rows, cols) = matrix_dims(dims, shape);
                    let (suh, svh) = match loaded.filter(|l| l.part == Some(0)) {
                        Some(l) => companions(plan, l.weight, address),
                        None => (None, None),
                    };
                    Some(grid::FormMeta {
                        name: slot.form.name(),
                        rows,
                        cols,
                        suh,
                        svh,
                    })
                } else {
                    None
                };
                items.push(grid::Item {
                    at,
                    bytes,
                    kind,
                    elem: slot.form.name(),
                    dims,
                    shape,
                    share,
                    form,
                });
                buf_index += 1;
            }
            arg => values.push(arg.u32().ok_or(tessera_lower::Error::Unbound)?),
        }
    }
    Ok((items, values))
}

/// Kept-row table descriptors for the KV convention.
fn kept_tables(plan: &Plan, address: &impl Fn(Slot) -> u64) -> Vec<grid::KeptDesc> {
    let pages = plan.table.map(|t| address(t) + t.start() as u64);
    let mut out = Vec::with_capacity(plan.kept.len());
    for ((_, slot), info) in plan.kept.iter().zip(&plan.kept_info) {
        let page_rows = info.page_rows.max(1);
        out.push(grid::KeptDesc {
            base: address(*slot) + slot.start() as u64,
            pages,
            len: None,
            slots: info.slots,
            context: info.context,
            pages_per_slot: info.context.div_ceil(page_rows),
            page_rows,
            row_bytes: slot.form.bytes(info.width as usize) as u32,
            elem: slot.form.name(),
            interleaved: info.interleaved,
        });
    }
    out
}

/// Control table counts for a resident step.
fn step_layout(plan: &Plan, step: &Step) -> grid::Layout {
    let mut items = 0usize;
    let mut forms = 0usize;
    for launch in step.launches() {
        let mut n_items = 0;
        let mut n_forms = 0;
        for arg in &launch.args {
            if let Arg::Buffer(slot, _) = arg {
                n_items += 1;
                if slot.form.read() == Read::Decode {
                    n_forms += 1;
                }
            }
        }
        items = items.max(n_items);
        forms = forms.max(n_forms);
    }
    let kept = plan.kept.len().min(plan.kept_info.len());
    let pages = match (plan.table, plan.kept_info.first()) {
        (None, Some(info)) if info.interleaved => (info.slots * info.context) as usize,
        (None, Some(info)) => info.slots as usize,
        _ => 0,
    };
    grid::Layout {
        items,
        forms,
        kept,
        pages,
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

/// Pins host memory through the shared context.
pub(crate) struct Pins;

impl OnGpu for Pins {
    type Out = Option<Arc<dyn tessera_engine_comm::Pinning>>;

    fn on<D: Device + Launch<Kernel = <D as Device>::Kernel>>(self) -> Self::Out {
        Some(Arc::new(Pin(cached_context::<D>().ok()?)))
    }
}

/// Pinned host memory from the shared context.
struct Pin<D: Device>(&'static D);

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
        let ctx = self.ctx;
        let unified = ctx.unified();
        let dropped = match self.base.held.bind(plan, |zone, bytes, holds| {
            // The host reads weight files straight into pinned memory.
            let buffer = match unified && holds == Holds::Weights {
                true => DeviceBuffer::pinned(ctx, bytes)?,
                false => DeviceBuffer::alloc(ctx, bytes)?,
            };
            // Kernels may read what they are about to write.
            buffer.zero()?;
            Ok(Buffer::new(zone, buffer))
        }) {
            Ok(dropped) => dropped,
            Err(e) => {
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
            let made =
                |source: &str, entry: &str| -> Result<_, Error> { Ok(ctx.compile(source, entry)?) };
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
        self.base.profile = on.then(crate::Profile::default);
        if let Some(grid) = self.grid.get_mut().as_mut() {
            grid.profiling(on);
        }
    }

    fn address(&self, slot: Slot) -> Option<u64> {
        Some(self.buffer(slot).device_ptr() + slot.at(0) as u64)
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
        tessera_codegen::Precision::of(D::BACKEND)
    }

    fn submit(&mut self, steps: Range<usize>) -> Result<(), Error> {
        self.reread()?;
        if self.base.profile.is_some() {
            return self.profiled(steps);
        }
        self.run(steps, false, true).map(|_| ())
    }

    fn queue(&mut self, steps: Range<usize>) -> Result<(), Error> {
        self.reread()?;
        if self.base.profile.is_some() {
            return self.profiled(steps);
        }
        self.run(steps, false, false).map(|_| ())
    }

    fn clocked(&mut self, steps: Range<usize>) -> Result<Option<f64>, Error> {
        self.reread()?;
        if self.base.profile.is_some() {
            return self.profiled(steps).map(|_| None);
        }
        self.run(steps, true, true)
    }
}
