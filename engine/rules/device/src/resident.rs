//! One persistent grid per rank, with its proxy beside it.

use crate::layout::{
    ARENAS, Barrier, ConventionRef, Counters, DTYPES, Fault, Item, Lane, Launch as Record,
    MAX_DEPTH, MAX_GROUP, MAX_POOLS, PAGE, Pool, PoolSet, Ref, SIZES, STATUS, Table, VALUES,
};
use crate::proxy::page::Page;
use crate::proxy::running::Running;
use crate::proxy::wired::{CREDITS, FLAGS, Place, RECV, REGIONS, SEND, Wired, regions};
use crate::proxy::{self, Op, Proxy};
use crate::{Backend, Options, flatten};
use std::ffi::c_int;
use std::fmt;
use std::time::{Duration, Instant};
use tessera_engine_comm::{Error as CommError, Link as Comm};
use tessera_gpukit::Error as GpuError;
use tessera_gpukit::backend::Backend as Gpu;
use tessera_gpukit::device::{Device, DeviceBuffer};
use tessera_gpukit::launch::{KernelArgs, Launch};

/// Arenas the launcher fills: control, doorbell, then the link's.
const CONTROL: u32 = 0;
const DOORBELL: u32 = 1;
const LINK: u32 = 2;
/// The first arena a plan's items name.
pub const USER: u32 = LINK + REGIONS as u32;
/// Every pool starts at this ordinal.
const BASE: u64 = 1;

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
    /// Device addresses of those arenas, in order.
    pub arenas: Vec<u64>,
    pub threads: u32,
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

/// Where each record sits in the control arena.
struct Places {
    launch: u64,
    status: u64,
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
    /// The device clock and the host instant it matches.
    clock: (u64, Instant),
    seq: u64,
    broken: bool,
}

impl<D, L> Resident<D, L>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Comm + 'static,
{
    /// Compiles `text`, lays out `plan` and starts the proxy.
    pub fn open(ctx: &D, text: &str, entry: &str, link: L, plan: &Plan) -> Result<Self, Error> {
        let rank = link.rank();
        check(plan, rank)?;
        let kernel = ctx.compile(text, entry)?;
        let fit = ctx.resident(&kernel, plan.threads, 0)?;
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
            },
            page,
            registered: Vec::new(),
            running: None,
            parked: None,
            grid,
            threads: plan.threads,
            clock: (0, Instant::now()),
            seq: 0,
            broken: false,
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
        let proxy = Proxy::new(&mut page_words, pools, proxy_lanes.collect(), BASE)?;
        made.lay_out(&wired, plan, [rank, ranks], arenas)?;
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
        [rank, ranks]: [u32; 2],
        arenas: [u64; ARENAS],
    ) -> Result<(), Error> {
        let mut block = Block::default();
        let table_at = block.put(&[zeroed::<Table>()]);
        let launch = block.put(&[zeroed::<Record>()]);
        let items = block.put(&plan.items);
        let lanes: Vec<Lane> = plan.lanes.iter().map(|l| lane(plan, l, rank)).collect();
        let lanes_at = block.put(&lanes);
        let mut set: PoolSet = zeroed();
        set.count = plan.pools.len() as u32;
        for (p, pool) in plan.pools.iter().enumerate() {
            let place = wired.place(p).expect("a place per pool");
            set.pools[p] = pool_record(pool, p, place);
        }
        let mut counters: Counters = zeroed();
        counters.next = [BASE; MAX_POOLS];
        // The first bump must differ from the page's zero.
        counters.writes = 1;
        set.state = Ref::at(CONTROL, block.put(&[counters]));
        let pools = block.put(&[set]);
        let barrier = block.put(&[zeroed::<Barrier>()]);
        let status = block.put(&[zeroed::<Fault>()]);
        let none = ConventionRef {
            at: Ref::NONE,
            version: 0,
            bytes: 0,
        };
        let table = Table {
            edition: crate::EDITION,
            rank,
            ranks,
            items_len: plan.items.len() as u32,
            forms_len: 0,
            lanes_len: lanes.len() as u32,
            values_len: VALUES as u32,
            unused: 0,
            arenas,
            now: Ref::at(CONTROL, launch),
            items: Ref::at(CONTROL, items),
            forms: Ref::NONE,
            conventions: [none; 16],
            lanes: Ref::at(CONTROL, lanes_at),
            pools: Ref::at(CONTROL, pools),
            doorbell: Ref::at(DOORBELL, 0),
            barrier: Ref::at(CONTROL, barrier),
            status: Ref::at(CONTROL, status),
            marks: Ref::NONE,
        };
        let control = DeviceBuffer::alloc(&self.ctx, block.0.len())?;
        let mut table = table;
        table.arenas[CONTROL as usize] = control.device_ptr();
        block.0[table_at as usize..][..size_of::<Table>()].copy_from_slice(bytes(&[table]));
        control.write_bytes(&block.0)?;
        self.control = control;
        self.places = Places { launch, status };
        Ok(())
    }

    /// Blocks in the grid.
    pub fn grid(&self) -> u32 {
        self.grid
    }

    /// One launch with `values`. Waits for the grid to end.
    pub fn launch(&mut self, values: &[u32], timeout: Duration) -> Result<(), Error> {
        if self.broken {
            return Err(plan("an earlier launch failed"));
        }
        if values.len() > VALUES {
            return Err(plan(format!("{} launch values", values.len())));
        }
        self.broken = true;
        self.alive()?;
        self.seq += 1;
        let ahead = (self.clock.1.elapsed() + timeout).as_secs_f64() * self.ctx.ticks();
        let mut record = Record {
            seq: self.seq,
            deadline: self.clock.0 + ahead as u64,
            values: [0; VALUES],
        };
        record.values[..values.len()].copy_from_slice(values);
        let base = self.control.device_ptr();
        self.ctx
            .upload(base + self.places.launch, bytes(&[record]))?;
        self.ctx
            .upload(base + self.places.status, bytes(&[zeroed::<Fault>()]))?;
        let mut args = KernelArgs::new();
        args.device_ptr(base);
        let (grid, threads) = ((self.grid, 1, 1), (self.threads, 1, 1));
        self.ctx.launch(&self.kernel, grid, threads, 0, &mut args)?;
        self.ctx.finish()?;
        self.alive()?;
        let mut fault = [zeroed::<Fault>()];
        let raw = unsafe {
            std::slice::from_raw_parts_mut(fault.as_mut_ptr().cast(), size_of::<Fault>())
        };
        self.ctx.download(base + self.places.status, raw)?;
        let [fault] = fault;
        if fault.code != 0 {
            return Err(Error::Fault {
                code: Status(fault.code),
                pool: fault.pool,
                ordinal: fault.ordinal,
            });
        }
        self.broken = false;
        Ok(())
    }

    /// The proxy's error, if its thread stopped.
    fn alive(&mut self) -> Result<(), Error> {
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
