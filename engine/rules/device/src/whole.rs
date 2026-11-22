//! A recipe's resident step: one grid per launch, sums inside.

use crate::layout::{
    BYTES, FORM, Form, INPUT, Item, Kept, NONE, OUTPUT, Ref, SCRATCH, dtype_code, form_kind,
};
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
use tessera_run::{Grid, Item as Placed, ItemKind, KeptDesc, Layout as GridLayout, Marks, marks};

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

/// Synthetic page entries for unpaged kept rows.
pub(crate) fn synth_pages(kept: &[KeptDesc]) -> Vec<u32> {
    let Some(first) = kept.first() else {
        return Vec::new();
    };
    if first.pages.is_some() {
        return Vec::new();
    }
    if first.interleaved {
        let mut out = Vec::with_capacity((first.slots * first.context) as usize);
        for slot in 0..first.slots {
            for pos in 0..first.context {
                out.push(pos * first.slots + slot);
            }
        }
        return out;
    }
    (0..first.slots).collect()
}

/// Packed control-table records for one launch.
pub(crate) type Records = (Vec<Item>, Vec<Form>, Vec<Kept>);

/// Converts run metadata into `core.h` items, forms and kept rows.
pub(crate) fn pack_records(
    items: &[Placed],
    scratch: Option<(u64, usize)>,
    kept: &[KeptDesc],
    pages_ref: Ref,
) -> Result<Records, Error> {
    let mut out_items = Vec::with_capacity(items.len() + usize::from(scratch.is_some()));
    let mut out_forms = Vec::new();
    for p in items {
        let kind = match p.kind {
            ItemKind::Input => INPUT,
            ItemKind::Output => OUTPUT,
            ItemKind::Scratch => SCRATCH,
        };
        let (elem, form, forms) = match p.form {
            Some(meta) => {
                let idx = out_forms.len() as u32;
                out_forms.push(Form {
                    codes: Ref::at(USER, p.at),
                    suh: meta.suh.map_or(Ref::NONE, |at| Ref::at(USER, at)),
                    svh: meta.svh.map_or(Ref::NONE, |at| Ref::at(USER, at)),
                    kind: form_kind(meta.name),
                    rows: meta.rows,
                    cols: meta.cols,
                    unused: 0,
                });
                (FORM, idx, 1)
            }
            None => (dtype_code(p.elem), NONE, 0),
        };
        let (dims, shape) = if elem == BYTES || p.dims == 0 {
            let bytes = u32::try_from(p.bytes)
                .map_err(|_| Error::Plan(format!("an item of {} bytes", p.bytes)))?;
            (1, [bytes, 0, 0, 0])
        } else {
            (p.dims, p.shape)
        };
        out_items.push(Item {
            at: Ref::at(USER, p.at),
            kind,
            elem,
            dims,
            shape,
            share: p.share,
            form,
            forms,
            unused: 0,
        });
    }
    if let Some((at, size)) = scratch {
        let bytes =
            u32::try_from(size).map_err(|_| Error::Plan(format!("an item of {size} bytes")))?;
        out_items.push(Item {
            at: Ref::at(USER, at),
            kind: SCRATCH,
            elem: BYTES,
            dims: 1,
            shape: [bytes, 0, 0, 0],
            share: [0; 4],
            form: NONE,
            forms: 0,
            unused: 0,
        });
    }
    let mut out_kept = Vec::with_capacity(kept.len());
    for k in kept {
        out_kept.push(Kept {
            base: Ref::at(USER, k.base),
            host: Ref::NONE,
            pages: k.pages.map_or(pages_ref, |at| Ref::at(USER, at)),
            len: k.len.map_or(Ref::NONE, |at| Ref::at(USER, at)),
            pages_per_slot: k.pages_per_slot,
            page_rows: k.page_rows,
            row_bytes: k.row_bytes,
            elem: dtype_code(k.elem),
        });
    }
    Ok((out_items, out_forms, out_kept))
}

/// The launcher's plan for `kernel` with `layout` counts.
fn plan(kernel: &KernelRef, layout: GridLayout, ranks: u32, share: u32) -> Result<Plan, Error> {
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
        items: vec![blank(); layout.items],
        forms: vec![blank_form(); layout.forms],
        kept: vec![blank_kept(); layout.kept],
        pages: vec![0; layout.pages],
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

    fn open(&mut self, kernel: &KernelRef, layout: GridLayout) -> Result<(), Error> {
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
        let key = format!(
            "{}/{threads:?}/{}/{}/{}/{}/{rev}/{want_marks}",
            kernel.name(),
            layout.items,
            layout.forms,
            layout.kept,
            layout.pages,
        );
        if matches!(&self.held, Held::Open { key: k, .. } if *k == key) {
            return Ok(());
        }
        let compiled = match prepared {
            Some(ref text) => self.ctx.compile(text, entry)?,
            None => cached_source::<D>(planned, entry)?,
        };
        // Ordinals go on, so the link's old flags never match.
        let (link, next) = self.link()?;
        let sized = GridLayout {
            items: layout.items + 1,
            ..layout
        };
        let plan = plan(kernel, sized, link.ranks(), self.share)?;
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
        kept: &[KeptDesc],
        values: &[u32],
    ) -> Result<(), Error> {
        let synth = synth_pages(kept);
        let forms = items.iter().filter(|p| p.form.is_some()).count();
        let layout = GridLayout {
            items: items.len(),
            forms,
            kept: kept.len(),
            pages: synth.len(),
        };
        // Each step runs in the grid built for its kernel.
        self.open(kernel, layout)?;
        let scratch = self.scratch.as_ref().map(|s| (s.device_ptr(), s.size()));
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
        let (all_items, all_forms, all_kept) =
            pack_records(items, scratch, kept, resident.pages_ref())?;
        if *marked && let Some(ptr) = ledger {
            self.ctx.zero(ptr, marks::COUNTS)?;
        }
        resident.refill(&all_items, &all_forms, &all_kept, &synth)?;
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
        kind: INPUT,
        elem: BYTES,
        dims: 1,
        shape: [0; 4],
        share: [0; 4],
        form: NONE,
        forms: 0,
        unused: 0,
    }
}

fn blank_form() -> Form {
    Form {
        codes: Ref::NONE,
        suh: Ref::NONE,
        svh: Ref::NONE,
        kind: 0,
        rows: 0,
        cols: 0,
        unused: 0,
    }
}

fn blank_kept() -> Kept {
    Kept {
        base: Ref::NONE,
        host: Ref::NONE,
        pages: Ref::NONE,
        len: Ref::NONE,
        pages_per_slot: 0,
        page_rows: 0,
        row_bytes: 0,
        elem: 0,
    }
}

impl<D, L> Grid for Whole<D, L>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Link + 'static,
{
    fn open(&mut self, kernel: &KernelRef, layout: GridLayout) -> Result<(), tessera_run::Error> {
        Whole::open(self, kernel, layout).map_err(failed)
    }

    fn launch(
        &mut self,
        kernel: &KernelRef,
        items: &[Placed],
        kept: &[KeptDesc],
        values: &[u32],
    ) -> Result<(), tessera_run::Error> {
        Whole::launch(self, kernel, items, kept, values).map_err(failed)
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
