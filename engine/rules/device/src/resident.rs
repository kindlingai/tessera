//! One persistent grid per rank, with its proxy beside it.

use crate::layout::{
    ARENAS, Barrier, ConventionRef, Counters, DOORBELL_ABORT, DTYPES, Fault, Form, Item, KV, Kept,
    KvTables, Lane, Launch as Record, MAX_DEPTH, MAX_GROUP, MAX_POOLS, NONE, PAGE, Pool, PoolSet,
    Ref, SIZES, STATUS, Table, VALUES,
};
use crate::proxy::page::Page;
use crate::proxy::running::Running;
use crate::proxy::wired::{CREDITS, FLAGS, Place, RECV, REGIONS, SEND, Wired, regions};
use crate::proxy::{self, Op, Proxy};
use crate::{Backend, Options, flatten};
use std::ffi::c_int;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tessera_engine_comm::{Error as CommError, Link as Comm};
use tessera_gpukit::Error as GpuError;
use tessera_gpukit::backend::Backend as Gpu;
use tessera_gpukit::device::{Device, DeviceBuffer, KernelStats};
use tessera_gpukit::launch::{Apart, KernelArgs, Launch};

/// Arenas the launcher fills: control, doorbell, then the link's.
const CONTROL: u32 = 0;
const DOORBELL: u32 = 1;
const LINK: u32 = 2;
/// The first arena a plan's items name.
pub const USER: u32 = LINK + REGIONS as u32;
/// A first grid's pools start at this ordinal.
pub(crate) const BASE: u64 = 1;

const CLOCK: &str = include_str!("clock.kernel");

unsafe extern "C" {
    fn getpagesize() -> c_int;
}

/// One pool: shape, member ranks, and bytes per slot.
#[derive(Clone, Debug)]
pub struct PoolPlan {
    pub depth: u32,
    pub one_way: bool,
    pub group: Vec<u32>,
    pub send_bytes: u32,
    pub recv_bytes: u32,
}

/// One lane: its operation, `dtype`, pool and all-to-all block.
#[derive(Clone, Copy, Debug)]
pub struct LanePlan {
    pub op: Op,
    pub elem: u32,
    pub pool: u32,
    pub block: u32,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub pools: Vec<PoolPlan>,
    pub lanes: Vec<LanePlan>,
    /// Items whose refs name arenas from `USER` on.
    pub items: Vec<Item>,
    /// Packed weight forms, indexed by `Item::form`.
    pub forms: Vec<Form>,
    /// Kept-row tables for the KV convention.
    pub kept: Vec<Kept>,
    /// Synthetic page entries kept in the control arena.
    pub pages: Vec<u32>,
    /// Device addresses of those arenas, in order.
    pub arenas: Vec<u64>,
    pub threads: u32,
    /// Bytes of dynamic shared memory per block.
    pub shared: u32,
    /// Grids that run on the device at once.
    pub share: u32,
}

impl Plan {
    fn proxy_pools(&self, rank: u32) -> Vec<proxy::Pool> {
        let pool = |p: &PoolPlan| proxy::Pool {
            depth: p.depth,
            one_way: p.one_way,
            group: p.group.len() as u32,
            me: p.group.iter().position(|&r| r == rank).unwrap_or(0) as u32,
        };
        self.pools.iter().map(pool).collect()
    }

    fn bytes(&self) -> Vec<(usize, usize)> {
        let bytes = |p: &PoolPlan| (p.send_bytes as usize, p.recv_bytes as usize);
        self.pools.iter().map(bytes).collect()
    }

    /// The region sizes for the rank's link.
    pub fn regions(&self) -> [usize; REGIONS] {
        regions(&self.proxy_pools(0), &self.bytes())
    }
}

/// A `status_code` from `core.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Status(pub u32);

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match STATUS.get(self.0 as usize) {
            Some(name) => f.write_str(&name.to_lowercase().replace('_', " ")),
            None => write!(f, "status {}", self.0),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Comm(#[from] CommError),
    #[error("proxy: {0}")]
    Proxy(#[from] proxy::Error),
    #[error(transparent)]
    Kernel(#[from] crate::Error),
    #[error("plan: {0}")]
    Plan(String),
    #[error("the kernel stopped: {code} in pool {pool} at ordinal {ordinal}")]
    Fault {
        code: Status,
        pool: u32,
        ordinal: u64,
    },
}

fn plan(what: impl Into<String>) -> Error {
    Error::Plan(what.into())
}

/// The bytes of plain `repr(C)` values.
fn bytes<T: Copy>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

/// The control arena, built on the host.
#[derive(Default)]
struct Block(Vec<u8>);

impl Block {
    fn put<T: Copy>(&mut self, values: &[T]) -> u64 {
        let at = self.0.len().next_multiple_of(16);
        self.0.resize(at, 0);
        self.0.extend_from_slice(bytes(values));
        at as u64
    }
}

fn zeroed<T>() -> T {
    // Only for the plain integer structs of `layout`.
    unsafe { std::mem::zeroed() }
}

/// Launch record and fault paired for one upload.
#[repr(C)]
#[derive(Clone, Copy)]
struct LaunchSlot {
    launch: Record,
    pad: u64,
    status: Fault,
}

/// Where each record sits in the control arena.
pub(crate) struct Places {
    pub launch: u64,
    pub status: u64,
    pub items: u64,
    pub forms: u64,
    pub kept: u64,
    pub pages: u64,
    pub counters: u64,
}

/// Builds the host control arena bytes and record offsets.
pub(crate) fn build_control(
    plan: &Plan,
    pool_places: &[Place],
    [rank, ranks, base]: [u64; 3],
    arenas: [u64; ARENAS],
) -> (Vec<u8>, Places) {
    let (rank, ranks) = (rank as u32, ranks as u32);
    let mut block = Block::default();
    let table_at = block.put(&[zeroed::<Table>()]);
    let launch = block.put(&[zeroed::<LaunchSlot>()]);
    let status = launch + std::mem::offset_of!(LaunchSlot, status) as u64;
    let items = block.put(&plan.items);
    let (forms_at, forms_ref) = if plan.forms.is_empty() {
        (0, Ref::NONE)
    } else {
        let at = block.put(&plan.forms);
        (at, Ref::at(CONTROL, at))
    };
    let pages_at = if plan.pages.is_empty() {
        0
    } else {
        block.put(&plan.pages)
    };
    let mut conventions = [ConventionRef {
        at: Ref::NONE,
        version: 0,
        bytes: 0,
    }; 16];
    let kept_at = if plan.kept.is_empty() {
        0
    } else {
        let mut tables = plan.kept.clone();
        if !plan.pages.is_empty() {
            for k in &mut tables {
                if k.pages.arena == NONE {
                    k.pages = Ref::at(CONTROL, pages_at);
                }
            }
        }
        let at = block.put(&tables);
        let kv = KvTables {
            count: tables.len() as u32,
            unused: 0,
            tables: Ref::at(CONTROL, at),
        };
        let kv_at = block.put(&[kv]);
        conventions[KV as usize] = ConventionRef {
            at: Ref::at(CONTROL, kv_at),
            version: crate::EDITION,
            bytes: size_of::<KvTables>() as u32,
        };
        at
    };
    let lanes: Vec<Lane> = plan.lanes.iter().map(|l| lane(plan, l, rank)).collect();
    let lanes_at = block.put(&lanes);
    let mut set: PoolSet = zeroed();
    set.count = plan.pools.len() as u32;
    for (p, (pool, &place)) in plan.pools.iter().zip(pool_places).enumerate() {
        set.pools[p] = pool_record(pool, p, place);
    }
    let mut counters: Counters = zeroed();
    counters.next = [base; MAX_POOLS];
    // The first bump must differ from the page's zero.
    counters.writes = 1;
    let counted = block.put(&[counters]);
    set.state = Ref::at(CONTROL, counted);
    let pools = block.put(&[set]);
    let barrier = block.put(&[zeroed::<Barrier>()]);
    let table = Table {
        edition: crate::EDITION,
        rank,
        ranks,
        items_len: plan.items.len() as u32,
        forms_len: plan.forms.len() as u32,
        lanes_len: lanes.len() as u32,
        values_len: VALUES as u32,
        unused: 0,
        arenas,
        now: Ref::at(CONTROL, launch),
        items: Ref::at(CONTROL, items),
        forms: forms_ref,
        conventions,
        lanes: Ref::at(CONTROL, lanes_at),
        pools: Ref::at(CONTROL, pools),
        doorbell: Ref::at(DOORBELL, 0),
        barrier: Ref::at(CONTROL, barrier),
        status: Ref::at(CONTROL, status),
        marks: Ref::NONE,
    };
    block.0[table_at as usize..][..size_of::<Table>()].copy_from_slice(bytes(&[table]));
    let places = Places {
        launch,
        status,
        items,
        forms: forms_at,
        kept: kept_at,
        pages: pages_at,
        counters: counted,
    };
    (block.0, places)
}

pub(crate) fn backend<D: Device>() -> Result<Backend, Error> {
    match D::BACKEND {
        Gpu::Cuda => Ok(Backend::Cuda),
        Gpu::Rocm => Ok(Backend::Hip),
        Gpu::Metal => Err(plan("Metal has no persistent grids")),
    }
}

/// The device's clock now, from a one-thread kernel.
fn clock<D: Device + Launch<Kernel = <D as Device>::Kernel>>(ctx: &D) -> Result<u64, Error> {
    let options = Options {
        backend: backend::<D>()?,
        remote_kv: false,
        marks: false,
    };
    let flat = flatten("clock.kernel", CLOCK, &|_| None, &options)?;
    let kernel = ctx.compile(&flat.text, "tessera_clock")?;
    let out = DeviceBuffer::alloc(ctx, 8)?;
    let mut args = KernelArgs::new();
    args.device_ptr(out.device_ptr());
    ctx.launch(&kernel, (1, 1, 1), (1, 1, 1), 0, &mut args)?;
    ctx.finish()?;
    let mut word = [0u8; 8];
    out.read_bytes(&mut word)?;
    Ok(u64::from_ne_bytes(word))
}

/// A persistent kernel, its control arena and its proxy.
pub struct Resident<D: Device, L> {
    ctx: D,
    kernel: D::Kernel,
    control: DeviceBuffer<D>,
    places: Places,
    page: u64,
    registered: Vec<*mut u8>,
    running: Option<Running<L>>,
    /// A stopped proxy's link, kept until its pages unregister.
    parked: Option<Wired<L>>,
    grid: u32,
    threads: u32,
    shared: u32,
    /// Resident blocks per multiprocessor.
    fit: u32,
    /// The device clock and the host instant it matches.
    clock: (u64, Instant),
    seq: u64,
    broken: bool,
    /// The table's item, form, kept and synthetic page counts.
    items: usize,
    forms: usize,
    kept: usize,
    pages: usize,
    /// A grid started apart, until `join`.
    apart: Option<Box<dyn Apart>>,
}

impl<D, L> Resident<D, L>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Comm + 'static,
{
    /// Compiles `text`, lays out `plan` and starts the proxy.
    pub fn open(ctx: &D, text: &str, entry: &str, link: L, plan: &Plan) -> Result<Self, Error> {
        let kernel = ctx.compile(text, entry)?;
        Self::compiled(ctx, kernel, entry, link, plan, BASE)
    }

    /// The same, compiled, with pools from ordinal `base`.
    pub fn compiled(
        ctx: &D,
        kernel: <D as Device>::Kernel,
        entry: &str,
        link: L,
        plan: &Plan,
        base: u64,
    ) -> Result<Self, Error> {
        let rank = link.rank();
        check(plan, rank)?;
        let fit = ctx.resident(&kernel, plan.threads, plan.shared)?;
        if fit == 0 {
            let threads = plan.threads;
            return Err(self::plan(format!(
                "{entry} cannot stay resident with {threads} threads"
            )));
        }
        let units = ctx.platform().units as u32;
        let grid = (units * fit / plan.share.max(1)).max(1);
        let ranks = link.ranks();
        let pools = plan.proxy_pools(rank);
        let groups = plan.pools.iter().map(|p| p.group.clone()).collect();
        let mut wired = Wired::new(link, &pools, &plan.bytes(), groups)?;
        let (page, device_page) = ctx.pin(PAGE)?;
        unsafe { std::ptr::write_bytes(page as *mut u8, 0, PAGE) };
        let mut made = Resident {
            ctx: ctx.clone(),
            kernel,
            control: DeviceBuffer::alloc(ctx, 16)?,
            places: Places {
                launch: 0,
                status: 0,
                items: 0,
                forms: 0,
                kept: 0,
                pages: 0,
                counters: 0,
            },
            page,
            registered: Vec::new(),
            running: None,
            parked: None,
            grid,
            threads: plan.threads,
            shared: plan.shared,
            fit,
            clock: (0, Instant::now()),
            seq: 0,
            broken: false,
            items: 0,
            forms: 0,
            kept: 0,
            pages: 0,
            apart: None,
        };
        let mut arenas = [0u64; ARENAS];
        arenas[DOORBELL as usize] = device_page;
        let mapped = made.map(wired.link(), plan)?;
        arenas[LINK as usize..][..REGIONS].copy_from_slice(&mapped);
        arenas[USER as usize..][..plan.arenas.len()].copy_from_slice(&plan.arenas);
        let mut page_words = unsafe { Page::new(page as *mut u64, &mut wired)? };
        let proxy_lanes = plan.lanes.iter().map(|l| proxy::Lane {
            op: l.op,
            pool: l.pool,
            width: width(l.elem),
            block: l.block,
        });
        let proxy = Proxy::new(&mut page_words, pools, proxy_lanes.collect(), base)?;
        made.lay_out(&wired, plan, [rank.into(), ranks.into(), base], arenas)?;
        made.clock = (clock(ctx)?, Instant::now());
        made.running = Some(Running::spawn(proxy, page_words, wired));
        Ok(made)
    }

    /// Device addresses of the link's regions. Registers unpinned pages.
    fn map(&mut self, link: &mut L, plan: &Plan) -> Result<[u64; REGIONS], Error> {
        let page = unsafe { getpagesize() } as usize;
        let mut found = Vec::with_capacity(REGIONS);
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (region, need) in plan.regions().into_iter().enumerate() {
            let (host, device, len) = link
                .memory(region)
                .ok_or_else(|| self::plan(format!("the link has no fixed region {region}")))?;
            if len < need {
                return Err(self::plan(format!(
                    "region {region} holds {len} of {need} bytes"
                )));
            }
            let host = host as usize;
            if device.is_none() {
                spans.push((host / page * page, (host + len).next_multiple_of(page)));
            }
            found.push((host, device));
        }
        // Regions may share pages, so each page registers once.
        spans.sort_unstable();
        let mut merged: Vec<(usize, usize, u64)> = Vec::new();
        for (start, end) in spans {
            match merged.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => merged.push((start, end, 0)),
            }
        }
        for (start, end, base) in &mut merged {
            *base = self.ctx.register(*start as *mut u8, *end - *start)?;
            self.registered.push(*start as *mut u8);
        }
        let mut out = [0u64; REGIONS];
        for (slot, (host, device)) in out.iter_mut().zip(found) {
            let span = merged.iter().find(|m| (m.0..m.1).contains(&host));
            *slot = match (device, span) {
                (Some(device), _) => device,
                (None, Some(&(start, _, base))) => base + (host - start) as u64,
                (None, None) => unreachable!("every unpinned region has a span"),
            };
        }
        Ok(out)
    }

    /// Writes the table and the records it points to.
    fn lay_out(
        &mut self,
        wired: &Wired<L>,
        plan: &Plan,
        ids: [u64; 3],
        arenas: [u64; ARENAS],
    ) -> Result<(), Error> {
        let mut pool_places = Vec::with_capacity(plan.pools.len());
        for p in 0..plan.pools.len() {
            pool_places.push(wired.place(p).expect("a place per pool"));
        }
        let (mut raw, places) = build_control(plan, &pool_places, ids, arenas);
        let control = DeviceBuffer::alloc(&self.ctx, raw.len())?;
        let at = std::mem::offset_of!(Table, arenas) + CONTROL as usize * 8;
        raw[at..at + 8].copy_from_slice(&control.device_ptr().to_ne_bytes());
        control.write_bytes(&raw)?;
        self.control = control;
        self.places = places;
        self.items = plan.items.len();
        self.forms = plan.forms.len();
        self.kept = plan.kept.len();
        self.pages = plan.pages.len();
        Ok(())
    }

    /// Ref to synthetic page entries in control memory, or `Ref::NONE`.
    pub fn pages_ref(&self) -> Ref {
        match self.pages {
            0 => Ref::NONE,
            _ => Ref::at(CONTROL, self.places.pages),
        }
    }

    /// The next free ordinal, the largest over the pools.
    pub fn next(&self) -> Result<u64, Error> {
        let mut next = [0u64; MAX_POOLS];
        let raw =
            unsafe { std::slice::from_raw_parts_mut(next.as_mut_ptr().cast(), 8 * MAX_POOLS) };
        self.ctx
            .download(self.table() + self.places.counters, raw)?;
        Ok(next.into_iter().max().unwrap_or(BASE))
    }

    /// Replaces items, forms, kept tables and pages in place.
    pub fn refill(
        &mut self,
        items: &[Item],
        forms: &[Form],
        kept: &[Kept],
        pages: &[u32],
    ) -> Result<(), Error> {
        let counts = [
            ("items", items.len(), self.items),
            ("forms", forms.len(), self.forms),
            ("kept", kept.len(), self.kept),
            ("pages", pages.len(), self.pages),
        ];
        for (what, got, laid) in counts {
            if got != laid {
                return Err(plan(format!("{got} {what}, laid out for {laid}")));
            }
        }
        let base = self.table();
        if !items.is_empty() {
            self.ctx.upload(base + self.places.items, bytes(items))?;
        }
        if !forms.is_empty() {
            self.ctx.upload(base + self.places.forms, bytes(forms))?;
        }
        if !kept.is_empty() {
            self.ctx.upload(base + self.places.kept, bytes(kept))?;
        }
        if !pages.is_empty() {
            self.ctx.upload(base + self.places.pages, bytes(pages))?;
        }
        Ok(())
    }

    /// Blocks in the grid.
    pub fn grid(&self) -> u32 {
        self.grid
    }

    /// Threads per block in the grid.
    pub fn threads(&self) -> u32 {
        self.threads
    }

    /// Resident blocks per multiprocessor.
    pub fn fit(&self) -> u32 {
        self.fit
    }

    /// The compiled kernel handle.
    pub fn kernel(&self) -> &<D as Device>::Kernel {
        &self.kernel
    }

    /// Static register and memory counts for this kernel.
    pub fn stats(&self) -> Option<KernelStats> {
        self.ctx.stats(&self.kernel)
    }

    /// One launch with `values`. Waits for the grid to end.
    pub fn launch(&mut self, values: &[u32], timeout: Duration) -> Result<(), Error> {
        let deadline = self.deadline(timeout);
        self.record(values, deadline)?;
        let mut args = KernelArgs::new();
        args.device_ptr(self.table());
        let (grid, threads) = ((self.grid, 1, 1), (self.threads, 1, 1));
        self.ctx
            .launch(&self.kernel, grid, threads, self.shared, &mut args)?;
        self.ctx.finish()?;
        self.status()?;
        self.broken = false;
        Ok(())
    }

    /// Starts the grid on its own stream, with no deadline.
    pub fn start(&mut self, values: &[u32]) -> Result<(), Error> {
        if self.apart.is_some() {
            return Err(plan("the grid already runs"));
        }
        self.record(values, 0)?;
        let mut args = KernelArgs::new();
        args.device_ptr(self.table());
        let (grid, threads) = ((self.grid, 1, 1), (self.threads, 1, 1));
        let apart = self
            .ctx
            .launch_apart(&self.kernel, grid, threads, &mut args)?;
        self.apart = Some(apart);
        self.broken = false;
        Ok(())
    }

    /// Waits for a started grid, then reads its fault.
    pub fn join(&mut self) -> Result<(), Error> {
        if let Some(apart) = self.apart.take() {
            apart.wait()?;
        }
        self.status()
    }

    /// The control table's device address, the kernel's argument.
    pub fn table(&self) -> u64 {
        self.control.device_ptr()
    }

    /// The device clock `timeout` from now.
    pub fn deadline(&self, timeout: Duration) -> u64 {
        let ahead = (self.clock.1.elapsed() + timeout).as_secs_f64() * self.ctx.ticks();
        self.clock.0 + ahead as u64
    }

    /// Writes the launch record and clears the fault.
    fn record(&mut self, values: &[u32], deadline: u64) -> Result<(), Error> {
        if self.broken {
            return Err(plan("an earlier launch failed"));
        }
        if values.len() > VALUES {
            return Err(plan(format!("{} launch values", values.len())));
        }
        self.broken = true;
        self.alive()?;
        self.seq += 1;
        let mut slot = LaunchSlot {
            launch: Record {
                seq: self.seq,
                deadline,
                values: [0; VALUES],
            },
            pad: 0,
            status: zeroed(),
        };
        slot.launch.values[..values.len()].copy_from_slice(values);
        let at = self.table() + self.places.launch;
        self.ctx.upload(at, bytes(&[slot]))?;
        Ok(())
    }

    /// The proxy's error, or the fault the grid recorded.
    pub fn status(&mut self) -> Result<(), Error> {
        self.alive()?;
        let mut fault = [zeroed::<Fault>()];
        let raw = unsafe {
            std::slice::from_raw_parts_mut(fault.as_mut_ptr().cast(), size_of::<Fault>())
        };
        self.ctx.download(self.table() + self.places.status, raw)?;
        let [fault] = fault;
        if fault.code != 0 {
            self.broken = true;
            return Err(Error::Fault {
                code: Status(fault.code),
                pool: fault.pool,
                ordinal: fault.ordinal,
            });
        }
        Ok(())
    }

    /// The proxy's error, if its thread stopped.
    pub fn alive(&mut self) -> Result<(), Error> {
        if !self.running.as_ref().is_none_or(Running::failed) {
            return Ok(());
        }
        let Some((wired, ended)) = self.running.take().map(Running::stop) else {
            return Err(plan("the proxy is gone"));
        };
        self.parked = wired;
        Err(ended
            .err()
            .map_or_else(|| plan("the proxy is gone"), Error::from))
    }

    /// Stops the proxy and gives the link back.
    pub fn close(mut self) -> Result<L, Error> {
        let running = self
            .running
            .take()
            .ok_or_else(|| plan("the proxy is gone"))?;
        let (wired, ended) = running.stop();
        let wired = wired.ok_or_else(|| plan("the proxy thread panicked"));
        ended?;
        Ok(wired?.into_link())
    }
}

impl<D: Device, L> Drop for Resident<D, L> {
    fn drop(&mut self) {
        // A grid still running sees the abort word and ends.
        if let Some(apart) = self.apart.take() {
            let abort = (self.page as usize + DOORBELL_ABORT) as *const AtomicU64;
            unsafe { &*abort }.store(1, Ordering::Release);
            let _ = apart.wait();
        }
        // The proxy reads the page and regions until it stops.
        if let Some(running) = self.running.take() {
            self.parked = running.stop().0;
        }
        // Unregister before `parked` drops and its link unmaps.
        for host in self.registered.drain(..) {
            self.ctx.unregister(host);
        }
        self.ctx.unpin(self.page);
    }
}

fn check(plan: &Plan, rank: u32) -> Result<(), Error> {
    if plan.pools.len() > MAX_POOLS {
        return Err(self::plan(format!("{} pools", plan.pools.len())));
    }
    if plan.arenas.len() > ARENAS - USER as usize {
        return Err(self::plan(format!("{} arenas", plan.arenas.len())));
    }
    if let Some(l) = plan.lanes.iter().find(|l| l.elem as usize >= DTYPES.len()) {
        return Err(self::plan(format!("a lane of dtype {}", l.elem)));
    }
    for (p, pool) in plan.pools.iter().enumerate() {
        if !pool.group.contains(&rank) {
            return Err(self::plan(format!("rank {rank} is not in pool {p}")));
        }
    }
    Ok(())
}

/// Bytes per element of a `dtype`. `check` refused unknown ones.
fn width(elem: u32) -> u32 {
    DTYPES[elem as usize].1
}

fn lane(plan: &Plan, l: &LanePlan, rank: u32) -> Lane {
    let group = &plan.pools[l.pool as usize].group;
    Lane {
        what: l.op as u32,
        elem: l.elem,
        group_size: group.len() as u32,
        group_rank: group.iter().position(|&r| r == rank).unwrap_or(0) as u32,
        block: l.block,
        pool: l.pool,
        unused: [0; 2],
    }
}

fn pool_record(pool: &PoolPlan, p: usize, place: Place) -> Pool {
    let region = |r: usize, at: usize| Ref::at(LINK + r as u32, at as u64);
    let [send, recv, flags, credits] = [SEND, RECV, FLAGS, CREDITS];
    let mut record: Pool = zeroed();
    record.depth = pool.depth;
    record.one_way = u32::from(pool.one_way);
    record.send_bytes = pool.send_bytes;
    record.recv_bytes = pool.recv_bytes;
    for slot in 0..pool.depth as usize {
        record.send[slot] = region(send, place.send + slot * place.send_bytes);
        for k in 0..MAX_GROUP {
            let at = place.recv + (slot * MAX_GROUP + k) * place.recv_bytes;
            record.recv[slot * MAX_GROUP + k] = region(recv, at);
        }
    }
    record.arrived = region(flags, place.flags);
    record.credit = region(credits, place.credits);
    let sizes = SIZES + p * MAX_DEPTH * MAX_GROUP * 4;
    record.sizes = Ref::at(DOORBELL, sizes as u64);
    record
}
