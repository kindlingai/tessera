//! A recipe's resident step: one grid per launch, sums inside.

use crate::layout::{BYTES, Item, NONE, Ref};
use crate::proxy::Op;
use crate::resident::{BASE, Error, LanePlan, Plan, Resident, USER, backend};
use crate::sums::pool;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::path::Path;
use std::time::Duration;
use tessera_engine_comm::{Link, ShmLink};
use tessera_gpukit::backend::Backend as Gpu;
use tessera_gpukit::device::{Device, DeviceBuffer, KernelStats, cached_source};
use tessera_gpukit::launch::Launch;
use tessera_lower::{Collective, KernelRef, Source};
use tessera_run::{Grid, Item as Placed, Marks, marks};

/// Ranks may still load when the first launch starts.
const TIMEOUT: Duration = Duration::from_secs(120);

/// The regions a link must hold for `ranks`.
pub use crate::sums::{RELAY, regions};

/// The idle link, or the open grid with the link.
enum Held<D: Device, L> {
    Idle(L),
    Open {
        key: String,
        entry: String,
        resident: Box<Resident<D, L>>,
        marked: bool,
    },
    /// A failed open took the link.
    Gone,
}

struct Whole<D: Device, L> {
    ctx: D,
    held: Held<D, L>,
    /// The open kernel's scratch, its last item.
    scratch: Option<DeviceBuffer<D>>,
    /// Grids that share the device at once.
    share: u32,
    profiling: bool,
    /// Replacement kernel text per entry name.
    overrides: HashMap<String, String>,
    /// Incremented whenever an entry override changes.
    revision: u64,
    /// Device ledger for `TS_MARK()` samples.
    ledger: Option<DeviceBuffer<D>>,
    /// Mark samples gathered since the last take.
    marks: Vec<(String, Marks)>,
}

/// Selects source text and mark mode for one resident compile.
pub(crate) fn prepare_source(
    planned: &'static str,
    overridden: Option<&str>,
    profiling: bool,
) -> (Option<String>, bool) {
    let text = overridden.unwrap_or(planned);
    let has_marks = text.contains("TS_MARK");
    if profiling && has_marks {
        return (Some(format!("{}{text}", marks::ON)), true);
    }
    if overridden.is_some() || has_marks {
        return (Some(format!("{}{text}", marks::OFF)), false);
    }
    (None, false)
}

/// Formats one concise telemetry line for a resident kernel.
pub(crate) fn format_summary(
    entry: &str,
    grid: u32,
    threads: u32,
    fit: u32,
    stats: Option<KernelStats>,
) -> String {
    match stats {
        Some(s) => format!(
            "tessera: resident {entry}: grid={grid} threads={threads} blocks/sm={fit} regs={} shared_bytes={} local_bytes={}",
            s.regs, s.shared_bytes, s.local_bytes
        ),
        None => format!("tessera: resident {entry}: grid={grid} threads={threads} blocks/sm={fit}"),
    }
}

fn failed(e: Error) -> tessera_run::Error {
    tessera_run::Error::Grid(Box::new(e))
}

/// The launcher's plan for `kernel` with `items` items.
fn plan(kernel: &KernelRef, items: usize, ranks: u32, share: u32) -> Result<Plan, Error> {
    let resident = kernel
        .resident()
        .ok_or_else(|| Error::Plan(format!("{} is not resident", kernel.name())))?;
    let lane = |c: &Collective| match c {
        Collective::Sum => LanePlan {
            op: Op::Sum,
            elem: crate::layout::F32,
            pool: 0,
            block: 0,
        },
    };
    Ok(Plan {
        pools: vec![pool(ranks)],
        lanes: resident.lanes.iter().map(lane).collect(),
        items: vec![blank(); items],
        // Base zero, so offsets are device addresses.
        arenas: vec![0],
        threads: resident.threads,
        shared: resident.shared,
        share,
    })
}

impl<D, L> Whole<D, L>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Link + 'static,
{
    /// The link, and the next ordinal on it.
    fn link(&mut self) -> Result<(L, u64), Error> {
        match std::mem::replace(&mut self.held, Held::Gone) {
            Held::Idle(link) => Ok((link, BASE)),
            Held::Open { resident, .. } => {
                let next = resident.next()?;
                Ok(((*resident).close()?, next))
            }
            Held::Gone => Err(Error::Plan("an earlier open lost the link".into())),
        }
    }

    /// Scratch of at least the kernel's bytes.
    fn scratch(&mut self, kernel: &KernelRef) -> Result<(), Error> {
        let bytes = kernel.resident().map_or(0, |r| r.scratch).max(16);
        if self.scratch.as_ref().is_none_or(|s| s.size() < bytes) {
            self.scratch = Some(DeviceBuffer::alloc(&self.ctx, bytes)?);
        }
        Ok(())
    }

    /// Binds `tessera_marks` to the device ledger when present.
    fn bind_marks(&mut self, kernel: &<D as Device>::Kernel) -> Result<bool, Error> {
        let Some(global) = self.ctx.global(kernel, "tessera_marks")? else {
            return Ok(false);
        };
        if self.ledger.is_none() {
            let bytes = marks::COUNTS + marks::records(marks::BLOCKS);
            self.ledger = Some(DeviceBuffer::alloc(&self.ctx, bytes)?);
        }
        let ledger = self.ledger.as_ref().map_or(0, DeviceBuffer::device_ptr);
        self.ctx.upload(global, &ledger.to_ne_bytes())?;
        Ok(true)
    }

    fn open(&mut self, kernel: &KernelRef, items: usize) -> Result<(), Error> {
        self.scratch(kernel)?;
        let Source::Fixed {
            cuda, hip, entry, ..
        } = kernel.source()
        else {
            return Err(Error::Plan(format!("{} has no fixed text", kernel.name())));
        };
        let planned = match backend::<D>()? {
            crate::Backend::Hip => hip,
            _ => cuda,
        };
        let overridden = self.overrides.get(entry).map(String::as_str);
        let replaced = overridden.is_some();
        let (prepared, want_marks) = prepare_source(planned, overridden, self.profiling);
        // Counts and scratch change with sizes, not the grid.
        let threads = kernel.resident().map(|r| (r.threads, r.shared, r.lanes));
        let rev = self.revision;
        let key = format!("{}/{threads:?}/{items}/{rev}/{want_marks}", kernel.name());
        if matches!(&self.held, Held::Open { key: k, .. } if *k == key) {
            return Ok(());
        }
        let compiled = match prepared {
            Some(ref text) => self.ctx.compile(text, entry)?,
            None => cached_source::<D>(planned, entry)?,
        };
        // Ordinals go on, so the link's old flags never match.
        let (link, next) = self.link()?;
        let plan = plan(kernel, items + 1, link.ranks(), self.share)?;
        let resident = Resident::compiled(&self.ctx, compiled, entry, link, &plan, next)?;
        if self.profiling || replaced {
            let summary = format_summary(
                entry,
                resident.grid(),
                resident.threads(),
                resident.fit(),
                resident.stats(),
            );
            eprintln!("{summary}");
        }
        let marked = match want_marks {
            true => self.bind_marks(resident.kernel())?,
            false => false,
        };
        self.held = Held::Open {
            key,
            entry: entry.to_string(),
            resident: Box::new(resident),
            marked,
        };
        Ok(())
    }

    fn launch(
        &mut self,
        kernel: &KernelRef,
        items: &[Placed],
        values: &[u32],
    ) -> Result<(), Error> {
        // Each step runs in the grid built for its kernel.
        self.open(kernel, items.len())?;
        let mut all = Vec::with_capacity(items.len() + 1);
        let scratch = self.scratch.as_ref().map(|s| Placed {
            at: s.device_ptr(),
            bytes: s.size(),
        });
        for p in items.iter().chain(&scratch) {
            let bytes = u32::try_from(p.bytes)
                .map_err(|_| Error::Plan(format!("an item of {} bytes", p.bytes)))?;
            all.push(Item {
                at: Ref::at(USER, p.at),
                shape: [bytes, 0, 0, 0],
                ..blank()
            });
        }
        let ledger = self.ledger.as_ref().map(DeviceBuffer::device_ptr);
        let Held::Open {
            entry,
            resident,
            marked,
            ..
        } = &mut self.held
        else {
            return Err(Error::Plan("no grid is open".into()));
        };
        if *marked && let Some(ptr) = ledger {
            self.ctx.zero(ptr, marks::COUNTS)?;
        }
        resident.refill(&all)?;
        resident.launch(values, TIMEOUT)?;
        if *marked && let Some(ptr) = ledger {
            let mut found = Marks::default();
            marks::collect(&self.ctx, ptr, &mut found)?;
            self.marks.push((entry.clone(), found));
        }
        Ok(())
    }
}

/// An item with no place yet, of untyped bytes.
fn blank() -> Item {
    Item {
        at: Ref::NONE,
        kind: 0,
        elem: BYTES,
        dims: 1,
        shape: [0; 4],
        share: [0; 4],
        form: NONE,
        forms: 0,
        unused: 0,
    }
}

impl<D, L> Grid for Whole<D, L>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Link + 'static,
{
    fn open(&mut self, kernel: &KernelRef, items: usize) -> Result<(), tessera_run::Error> {
        Whole::open(self, kernel, items).map_err(failed)
    }

    fn launch(
        &mut self,
        kernel: &KernelRef,
        items: &[Placed],
        values: &[u32],
    ) -> Result<(), tessera_run::Error> {
        Whole::launch(self, kernel, items, values).map_err(failed)
    }

    fn profiling(&mut self, on: bool) {
        self.profiling = on;
    }

    fn replace(&mut self, entry: &str, source: Option<String>) {
        if self.overrides.get(entry).map(String::as_str) == source.as_deref() {
            return;
        }
        match source {
            Some(text) => {
                self.overrides.insert(entry.to_string(), text);
            }
            None => {
                self.overrides.remove(entry);
            }
        }
        self.revision += 1;
    }

    fn take_marks(&mut self) -> Vec<(String, Marks)> {
        std::mem::take(&mut self.marks)
    }
}

/// Resident steps over `link` on `ctx`, `share` grids at once.
pub fn on<D, L>(ctx: &D, link: L, share: u32) -> Box<dyn Grid>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Link + 'static,
{
    Box::new(Whole {
        ctx: ctx.clone(),
        held: Held::Idle(link),
        scratch: None,
        share,
        profiling: false,
        overrides: HashMap::new(),
        revision: 0,
        ledger: None,
        marks: Vec::new(),
    })
}

/// Resident steps over `link` on the GPU `gpu` names.
pub fn open<L: Link + 'static>(
    gpu: Option<Gpu>,
    link: L,
    share: u32,
) -> Result<Box<dyn Grid>, Error> {
    match gpu {
        #[cfg(feature = "cuda")]
        Some(Gpu::Cuda) => Ok(on(tessera_gpukit::cuda::cached_context()?, link, share)),
        #[cfg(feature = "rocm")]
        Some(Gpu::Rocm) => Ok(on(tessera_gpukit::rocm::cached_context()?, link, share)),
        _ => {
            drop((link, share));
            Err(Error::Plan(
                "persistent grids need a CUDA or ROCm GPU".into(),
            ))
        }
    }
}

/// One rank alone, over a new file at `path`.
pub fn alone(gpu: Option<Gpu>, path: &Path) -> Result<Box<dyn Grid>, Error> {
    let regions = regions(1);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(tessera_engine_comm::Error::from)?;
    file.set_len(ShmLink::room(1, &regions) as u64)
        .map_err(tessera_engine_comm::Error::from)?;
    let link = ShmLink::over(0, 1, file, &regions);
    // The mapping stays after the name goes.
    let _ = std::fs::remove_file(path);
    open(gpu, link?, 1)
}
