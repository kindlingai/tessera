//! A three-phase all-reduce in one persistent grid per rank.

use crate::Args;
use std::error::Error;
use std::time::{Duration, Instant};
use tessera_device::layout::{Item, NONE, Ref};
use tessera_device::proxy::Op;
use tessera_device::resident::{LanePlan, Plan, PoolPlan, Resident, USER};
use tessera_device::{Backend, Options, flatten};
use tessera_engine_comm::{Link, ShmLink, Transport, VerbsLink};
use tessera_frontend_comm::{meet, shared};
use tessera_gpukit::device::{Device, DeviceBuffer};
use tessera_gpukit::launch::Launch;

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
    match settings.gpu() {
        #[cfg(feature = "cuda")]
        Some(tessera_gpukit::backend::Backend::Cuda) => {
            let ctx = tessera_gpukit::cuda::cached_context()?;
            on(ctx, Backend::Cuda, args, rounds)
        }
        #[cfg(feature = "rocm")]
        Some(tessera_gpukit::backend::Backend::Rocm) => {
            let ctx = tessera_gpukit::rocm::cached_context()?;
            on(ctx, Backend::Hip, args, rounds)
        }
        _ => Err("the stub needs a CUDA or ROCm GPU".into()),
    }
}

fn on<D>(ctx: &D, backend: Backend, args: &Args, rounds: u32) -> Outcome
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
        backend,
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
