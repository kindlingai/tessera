//! A recipe's resident step: one grid per launch, sums inside.

use crate::layout::{BYTES, Item, NONE, Ref};
use crate::proxy::Op;
use crate::resident::{BASE, Error, LanePlan, Plan, Resident, USER, backend};
use crate::sums::pool;
use std::fs::OpenOptions;
use std::path::Path;
use std::time::Duration;
use tessera_engine_comm::{Link, ShmLink};
use tessera_gpukit::backend::Backend as Gpu;
use tessera_gpukit::device::{Device, DeviceBuffer, cached_source};
use tessera_gpukit::launch::Launch;
use tessera_lower::{Collective, KernelRef, Source};
use tessera_run::{Grid, Item as Placed};

/// Ranks may still load when the first launch starts.
const TIMEOUT: Duration = Duration::from_secs(120);

/// The regions a link must hold for `ranks`.
pub use crate::sums::{RELAY, regions};

/// The idle link, or the open grid with the link.
enum Held<D: Device, L> {
    Idle(L),
    Open {
        key: String,
        resident: Box<Resident<D, L>>,
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

    fn open(&mut self, kernel: &KernelRef, items: usize) -> Result<(), Error> {
        self.scratch(kernel)?;
        // Counts and scratch change with sizes, not the grid.
        let threads = kernel.resident().map(|r| (r.threads, r.lanes));
        let key = format!("{}/{threads:?}/{items}", kernel.name());
        if matches!(&self.held, Held::Open { key: k, .. } if *k == key) {
            return Ok(());
        }
        let Source::Fixed {
            cuda, hip, entry, ..
        } = kernel.source()
        else {
            return Err(Error::Plan(format!("{} has no fixed text", kernel.name())));
        };
        let text = match backend::<D>()? {
            crate::Backend::Hip => hip,
            _ => cuda,
        };
        // Ordinals go on, so the link's old flags never match.
        let (link, next) = self.link()?;
        let plan = plan(kernel, items + 1, link.ranks(), self.share)?;
        let compiled = cached_source::<D>(text, entry)?;
        let resident = Resident::compiled(&self.ctx, compiled, entry, link, &plan, next)?;
        let resident = Box::new(resident);
        self.held = Held::Open { key, resident };
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
        let Held::Open { resident, .. } = &mut self.held else {
            return Err(Error::Plan("no grid is open".into()));
        };
        resident.refill(&all)?;
        resident.launch(values, TIMEOUT)
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
