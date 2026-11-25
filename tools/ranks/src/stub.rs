//! A three-phase all-reduce in one persistent grid per rank.

use crate::Args;
use std::error::Error;
use std::time::{Duration, Instant};
use tessera_device::layout::{Item, NONE, Ref};
use tessera_device::p2p::P2p;
use tessera_device::proxy::Op;
use tessera_device::resident::{LanePlan, Plan, PoolPlan, Resident, USER};
use tessera_device::{Options, flatten};
use tessera_engine_comm::{Link, ShmLink, Transport, VerbsLink};
use tessera_frontend_comm::{CAPACITY, Sums, meet, shared, shm};
use tessera_gpukit::backend::OnGpu;
use tessera_gpukit::device::{Device, DeviceBuffer, cached_context};
use tessera_gpukit::launch::Launch;
use tessera_graph::DType;
use tessera_run::Sum;

const STUB: &str = include_str!("all_reduce_stub.cu");
/// Floats per part, and threads per block.
const ELEMENTS: u32 = 4096;
const THREADS: u32 = 256;
const TIMEOUT: Duration = Duration::from_secs(30);
/// Under GB10 MPS, two 48-block grids ran one after another.
const MPS: u32 = 4;

type Outcome = Result<(), Box<dyn Error>>;

/// Runs `rounds` sums on the GPU the settings pick.
pub fn run(args: &Args, rounds: u32) -> Outcome {
    let settings = tessera_frontend_settings::read();
    if args.sums == Sums::P2p {
        tessera_run::select(settings.device.unwrap_or(args.rank as usize));
    }
    let stub = Stub(args, rounds);
    tessera_gpukit::backend::dispatch(settings.gpu().filter(|b| b.persistent()), stub)
        .unwrap_or_else(|_| Err("the stub needs a CUDA or ROCm GPU".into()))
}

struct Stub<'a>(&'a Args, u32);

impl OnGpu for Stub<'_> {
    type Out = Outcome;

    fn on<D: Device + Launch<Kernel = <D as Device>::Kernel>>(self) -> Outcome {
        let ctx = cached_context::<D>()?;
        match self.0.sums {
            Sums::P2p => p2p_stub(ctx, self.0, self.1),
            _ => on(ctx, self.0, self.1),
        }
    }
}

/// Runs `rounds` direct P2P sums of `args.size` floats.
fn p2p_stub<D>(ctx: &D, args: &Args, rounds: u32) -> Outcome
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
{
    let (rank, ranks, n) = (args.rank, args.ranks as u32, args.size as usize);
    let mut boot: Box<dyn Transport> = match &args.shm {
        Some(path) => Box::new(shm(path, rank, ranks, CAPACITY)?),
        None => Box::new(meet(rank, &args.addrs, args.listen)?),
    };
    let wire = args.sum_wire.unwrap_or(DType::F32);
    let bf16 = wire == DType::Bf16;
    let mut p2p = P2p::on(ctx, &mut *boot, &args.shape)?.with_schedule(args.sched);
    let close = |got: f32, want: f32| {
        if bf16 {
            (got - want).abs() <= want.abs() * 0.02 + 1e-2
        } else {
            got == want
        }
    };
    let bytes = n * 4;
    let from = DeviceBuffer::alloc(ctx, bytes)?;
    let to = DeviceBuffer::alloc(ctx, bytes)?;
    let mine: Vec<f32> = (0..n)
        .map(|i| ((rank + 1) * (i as u32 % 97)) as f32)
        .collect();
    from.write_f32(&mine)?;
    to.zero()?;
    for _ in 0..5 {
        p2p.sum(from.device_ptr(), to.device_ptr(), n, wire)?;
    }
    p2p.check()?;
    boot.exchange(&[])?;
    let start = Instant::now();
    for _ in 0..rounds {
        p2p.sum(from.device_ptr(), to.device_ptr(), n, wire)?;
    }
    p2p.check()?;
    let elapsed = start.elapsed().as_secs_f64();
    let mut got = vec![0f32; n];
    to.read_f32(&mut got)?;
    let triangle = (ranks * (ranks + 1) / 2) as f32;
    // Every rank must hold the same bits.
    let bits: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
    let sums = boot.exchange(&bits)?;
    let mut wrong = sums.iter().filter(|s| **s != sums[0]).count();
    for (i, &v) in got.iter().enumerate() {
        let want = (i % 97) as f32 * triangle;
        wrong += usize::from(!close(v, want));
    }
    // Verify in-place all-reduce.
    p2p.sum(to.device_ptr(), to.device_ptr(), n, wire)?;
    p2p.check()?;
    to.read_f32(&mut got)?;
    for (i, &v) in got.iter().enumerate() {
        let want = (i % 97) as f32 * triangle * ranks as f32;
        wrong += usize::from(!close(v, want));
    }
    // Verify CUDA/HIP graph capture and repeated replay.
    if ctx.graphs() && p2p.capturable() {
        let graph = ctx.capture(|| -> Result<(), Box<dyn Error>> {
            p2p.sum(from.device_ptr(), to.device_ptr(), n, wire)?;
            p2p.sum(to.device_ptr(), to.device_ptr(), n, wire)?;
            Ok(())
        })?;
        for _ in 0..4 {
            ctx.replay(&graph)?;
        }
        p2p.check()?;
        to.read_f32(&mut got)?;
        for (i, &v) in got.iter().enumerate() {
            let want = (i % 97) as f32 * triangle * ranks as f32;
            wrong += usize::from(!close(v, want));
        }
    }
    // Verify multi-row P2P all-gather (direct and graphed).
    let rows = if n >= 4 && n.is_multiple_of(4) { 4 } else { 1 };
    let total = n * ranks as usize;
    let g_out = DeviceBuffer::alloc(ctx, total * 4)?;
    g_out.zero()?;
    let g_ok = p2p.gather(from.device_ptr(), g_out.device_ptr(), n, total, rows)?;
    if g_ok {
        p2p.check()?;
        let mut g_got = vec![0f32; total];
        g_out.read_f32(&mut g_got)?;
        let local_w = n / rows;
        let cols = total / rows;
        for r in 0..rows {
            for k in 0..ranks as usize {
                for c in 0..local_w {
                    let src_i = r * local_w + c;
                    let want = ((k as u32 + 1) * (src_i as u32 % 97)) as f32;
                    let got_v = g_got[r * cols + k * local_w + c];
                    wrong += usize::from(got_v != want);
                }
            }
        }
        if ctx.graphs() && p2p.capturable() {
            g_out.zero()?;
            let graph = ctx.capture(|| -> Result<(), Box<dyn Error>> {
                p2p.sum(from.device_ptr(), to.device_ptr(), n, wire)?;
                let ok = p2p.gather(from.device_ptr(), g_out.device_ptr(), n, total, rows)?;
                if !ok {
                    return Err("p2p gather declined inside graph".into());
                }
                Ok(())
            })?;
            for _ in 0..4 {
                ctx.replay(&graph)?;
            }
            p2p.check()?;
            g_out.read_f32(&mut g_got)?;
            for r in 0..rows {
                for k in 0..ranks as usize {
                    for c in 0..local_w {
                        let src_i = r * local_w + c;
                        let want = ((k as u32 + 1) * (src_i as u32 % 97)) as f32;
                        let got_v = g_got[r * cols + k * local_w + c];
                        wrong += usize::from(got_v != want);
                    }
                }
            }
        }
    }
    let us = elapsed * 1e6 / f64::from(rounds);
    let gbs = (bytes as f64 * f64::from(ranks - 1)) / (us * 1e3);
    eprintln!(
        "rank {rank}: p2p {rounds} sums of {n} floats ({bytes} B, {:?}), {wrong} wrong; \
         {us:.2} us/sum, {gbs:.2} GB/s out",
        args.sched
    );
    drop(p2p);
    boot.exchange(&[])?;
    match wrong {
        0 => Ok(()),
        _ => Err(format!("{wrong} sums are wrong").into()),
    }
}

fn on<D>(ctx: &D, args: &Args, rounds: u32) -> Outcome
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
{
    let (rank, ranks) = (args.rank, args.ranks as u32);
    let bytes = ELEMENTS * 4;
    let out = DeviceBuffer::alloc(ctx, (rounds * bytes) as usize)?;
    let item = Item {
        at: Ref::at(USER, 0),
        kind: 1,
        elem: 0,
        dims: 2,
        shape: [rounds, ELEMENTS, 1, 1],
        share: [0; 4],
        form: NONE,
        forms: 0,
        unused: 0,
    };
    let plan = Plan {
        pools: vec![PoolPlan {
            depth: 2,
            one_way: false,
            group: (0..ranks).collect(),
            send_bytes: bytes,
            recv_bytes: bytes,
        }],
        lanes: vec![LanePlan {
            op: Op::Sum,
            elem: 0,
            pool: 0,
            block: 0,
        }],
        items: vec![item],
        forms: Vec::new(),
        kept: Vec::new(),
        pages: Vec::new(),
        arenas: vec![out.device_ptr()],
        threads: THREADS,
        shared: 0,
        share: if args.shm.is_some() { ranks * MPS } else { 1 },
    };
    let options = Options {
        backend: D::BACKEND,
        remote_kv: false,
        marks: false,
    };
    let text = flatten("all_reduce_stub.cu", STUB, &|_| None, &options)?.text;
    let regions = plan.regions();
    let sums = match &args.shm {
        Some(path) => {
            let file = shared(path, rank, ranks, ShmLink::room(ranks, &regions))?;
            let link = ShmLink::over(rank, ranks, file, &regions)?;
            timed(ctx, &text, link, &plan, rounds)?
        }
        None => {
            let fabric = args
                .fabric
                .as_ref()
                .ok_or("the stub over sockets needs --verbs")?;
            let mut boot = meet(rank, &args.addrs, args.listen)?;
            let pinning = tessera_run::pinning(&tessera_frontend_settings::read());
            let link = VerbsLink::connect(
                rank,
                ranks,
                fabric,
                &regions,
                bytes as usize,
                &mut boot,
                pinning,
            )?;
            let timing = timed(ctx, &text, link, &plan, rounds)?;
            // Ranks keep their links until every peer is done.
            boot.exchange(&[])?;
            timing
        }
    };
    let mut got = vec![0f32; (rounds * ELEMENTS) as usize];
    out.read_f32(&mut got)?;
    let triangle = (ranks * (ranks + 1) / 2) as f32;
    let mut wrong = 0;
    for (at, v) in got.iter().enumerate() {
        let (round, i) = (at as u32 / ELEMENTS, at as u32 % ELEMENTS);
        let want = (i % 97) as f32 * triangle + (ranks * round) as f32;
        wrong += usize::from(*v != want);
    }
    let [one, all] = sums.map(|d| d.as_secs_f64() * 1e6);
    let trip = (all - one) / f64::from(rounds.max(2) - 1);
    eprintln!(
        "rank {rank}: {rounds} sums of {ELEMENTS} floats, {wrong} wrong; \
         one launch {one:.0} us, {all:.0} us in all, {trip:.1} us per round trip"
    );
    match wrong {
        0 => Ok(()),
        _ => Err(format!("{wrong} sums are wrong").into()),
    }
}

/// Launch times for one round, then for `rounds`.
fn timed<D, L>(
    ctx: &D,
    text: &str,
    link: L,
    plan: &Plan,
    rounds: u32,
) -> Result<[Duration; 2], Box<dyn Error>>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Link + 'static,
{
    let rank = link.rank();
    let mut grid = Resident::open(ctx, text, "all_reduce_stub", link, plan)?;
    eprintln!("rank {rank}: {} blocks of {THREADS} threads", grid.grid());
    grid.launch(&[ELEMENTS, 1], TIMEOUT)?;
    let mut times = [Duration::ZERO; 2];
    for (time, count) in times.iter_mut().zip([1, rounds]) {
        let start = Instant::now();
        grid.launch(&[ELEMENTS, count], TIMEOUT)?;
        *time = start.elapsed();
    }
    grid.close()?;
    Ok(times)
}
