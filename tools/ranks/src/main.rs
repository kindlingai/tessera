//! One model rank, sharded over TCP, against the fixture.

// Linking these registers their lowering rules.
use tessera_rules as _;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use tessera_bundle::Weights;
use tessera_device::p2p::Schedule;
use tessera_device::sums::{self, Sum};
use tessera_engine_comm::{Fabric, ShmLink, Transport};
use tessera_frontend_comm::{
    CAPACITY, Sums, meet, p2p, ports, shared, shm, sum_wire_flag, sums_flag,
};
use tessera_graph::DimName::{Context, Seqs};
use tessera_graph::{DType, shard_aligned};
use tessera_interp::Tensor;
use tessera_lower::{Cluster, Device, Knobs, Pick, ZoneKind, lower};
use tessera_models::preset;
use tessera_run::{Rank, Run, sum_wire};

const USAGE: &str = "tessera-ranks --rank N (--addrs HOST:PORT,... | --shm PATH --ranks N)
  [--model NAME] [--fixture DIR] [--listen ADDR] [--backend host|gpu] [--parts 3,1]
  [--form q8_0] [--verbs DEV[:GID],... | --ring PREVDEV,.../NEXTDEV,...] [--repeat N]
  [--stub ROUNDS] [--size FLOATS] [--sched auto|fanout|rounds|butterfly] [--shape KERNEL:KNOB=N]... [--sums host|grid|p2p]
  [--sum-wire f32|bf16|f16] [--recipe DIR] [--require-favored]
  --verbs and --ring move the sums over RDMA, one lane per device, after the
  ranks meet over the sockets or the file. --sums grid runs them in a
  persistent grid per rank, over that fabric or the file. --sums p2p runs
  direct GPU-to-GPU SM-store sums over IPC. --sum-wire sets the dtype
  each rank's part crosses as; by default F32, or BF16 for --sums grid.
  The fixture holds model.safetensors, and tokens.txt with logits.f32,
  at its root or under cases/prompt. Its default is ~/.cache/tessera/references/<model>.
  --stub runs ROUNDS sums in one persistent grid per rank instead of a model.
  --recipe runs a small model, such as qwen3 or glm53, with the recipe's
  resident grid per rank, over --shm or --verbs, and rank 0 checks it
  against one engine alone.";

struct Args {
    rank: u32,
    addrs: Vec<SocketAddr>,
    /// A mapped file joining the ranks, instead of sockets.
    shm: Option<PathBuf>,
    ranks: usize,
    model: String,
    fixture: Option<PathBuf>,
    listen: Option<SocketAddr>,
    host: bool,
    parts: Vec<u32>,
    /// A form every weight takes, instead of the file's.
    form: Option<String>,
    /// Forward passes to run. The last one is reported.
    repeat: usize,
    /// RDMA ports, after the ranks meet.
    fabric: Option<Fabric>,
    /// Sums for the stub, instead of a model.
    stub: Option<u32>,
    /// Floats per stub sum.
    size: u32,
    /// P2P schedule override for stub runs.
    sched: Schedule,
    shape: Knobs,
    /// How the model's sums run across ranks.
    sums: Sums,
    /// The dtype each part crosses as. `None`: the transport's.
    sum_wire: Option<DType>,
    /// A recipe whose grid runs the small model.
    recipe: Option<PathBuf>,
    /// Fail lowering unless the favored rule covers a step.
    require_favored: bool,
}

fn sched_flag(value: &str) -> Option<Schedule> {
    match value {
        "auto" => Some(Schedule::Auto),
        "fanout" => Some(Schedule::Fanout),
        "rounds" => Some(Schedule::Rounds),
        "butterfly" => Some(Schedule::Butterfly),
        _ => None,
    }
}

fn parse(mut given: impl Iterator<Item = String>) -> Option<Args> {
    let mut args = Args {
        rank: 0,
        addrs: Vec::new(),
        shm: None,
        ranks: 0,
        model: "qwen3".into(),
        fixture: None,
        listen: None,
        host: false,
        parts: Vec::new(),
        form: None,
        repeat: 1,
        fabric: None,
        stub: None,
        size: 4096,
        sched: Schedule::Auto,
        shape: Knobs::default(),
        sums: Sums::Host,
        sum_wire: None,
        recipe: None,
        require_favored: false,
    };
    while let Some(flag) = given.next() {
        if flag == "--require-favored" {
            args.require_favored = true;
            continue;
        }
        let value = given.next()?;
        match flag.as_str() {
            "--rank" => args.rank = value.parse().ok()?,
            "--addrs" => {
                let addrs = value.split(',').map(|a| a.parse().ok());
                args.addrs = addrs.collect::<Option<_>>()?;
            }
            "--shm" => args.shm = Some(value.into()),
            "--ranks" => args.ranks = value.parse().ok()?,
            "--model" => args.model = value,
            "--fixture" => args.fixture = Some(value.into()),
            "--listen" => args.listen = Some(value.parse().ok()?),
            "--backend" => args.host = value == "host" || value == "cpu",
            "--form" => args.form = Some(value),
            "--repeat" => args.repeat = value.parse().ok().filter(|n| *n > 0)?,
            "--sums" => args.sums = sums_flag(&value)?,
            "--sum-wire" => args.sum_wire = Some(sum_wire_flag(&value)?),
            "--recipe" => args.recipe = Some(value.into()),
            "--stub" => args.stub = Some(value.parse().ok().filter(|n| *n > 0)?),
            "--size" => args.size = value.parse().ok().filter(|n| *n > 0)?,
            "--sched" => args.sched = sched_flag(&value)?,
            "--shape" => args.shape.parse(&value).ok()?,
            "--verbs" => args.fabric = Some(Fabric::Mesh(ports(&value).ok()?)),
            "--ring" => {
                let (prev, next) = value.split_once('/')?;
                let (prev, next) = (ports(prev).ok()?, ports(next).ok()?);
                args.fabric = Some(Fabric::Ring { prev, next });
            }
            "--parts" => {
                let parts = value.split(',').map(|p| p.parse().ok());
                args.parts = parts.collect::<Option<_>>()?;
            }
            _ => return None,
        }
    }
    if args.shm.is_none() {
        args.ranks = args.addrs.len();
    }
    ((args.rank as usize) < args.ranks).then_some(args)
}

mod resident;
#[cfg(any(feature = "cuda", feature = "rocm"))]
mod stub;

/// The persistent-grid stub, where a GPU backend is built in.
fn stub(args: &Args, rounds: u32) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    return stub::run(args, rounds);
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    return Err(format!(
        "--stub {rounds} for rank {} needs a cuda or rocm build",
        args.rank
    )
    .into());
}

#[cfg(test)]
mod eg2_test;
#[cfg(test)]
mod mla_test;

fn main() -> ExitCode {
    let Some(args) = parse(std::env::args().skip(1)) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let ran = match (args.stub, &args.recipe) {
        (Some(rounds), _) => stub(&args, rounds),
        (None, Some(dir)) => resident::run(&args, dir),
        (None, None) => run(&args),
    };
    match ran {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tessera-ranks: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `ranks` linked zones on the host or the GPU.
fn device(
    settings: &tessera_lower::Settings,
    pins: tessera_lower::Pins,
    ranks: usize,
    host: bool,
) -> Device {
    let kind = match host {
        true => ZoneKind::Cpu,
        false => ZoneKind::Gpu,
    };
    let device = Device::guess(settings)
        .pinned(pins)
        .zoned(Cluster::mesh(ranks, kind));
    device.ranked((0..ranks as u32).collect())
}

type Loaded = (tessera_graph::Graph, Weights, tessera_lower::Pins);

/// One sequence's graph for `n` tokens, its weights and forms.
fn model(name: &str, fixture: &Path, n: u64) -> Result<Loaded, Box<dyn std::error::Error>> {
    let g = match name {
        "deepseek" | "glm53" => {
            let r = tessera_bundle::mla::release(fixture, 1, n, 3).map_err(|e| e.to_string())?;
            let (graph, pins) = (r.graph.clone(), r.pins.clone());
            return Ok((graph, Weights::Streamed(Box::new(r)), pins));
        }
        "eg2" => {
            let e = tessera_bundle::encoder::encoder(
                &fixture.join("exlr8"),
                [1, n],
                false,
                &Default::default(),
            )
            .map_err(|e| e.to_string())?;
            let pins = tessera_lower::Pins::default();
            return Ok((e.graph, Weights::Held(e.weights), pins));
        }
        _ => {
            let forward = preset::named(name)?.forward;
            forward.ok_or_else(|| format!("`{name}` has no forward graph"))?(n)
        }
    };
    let file = tessera_checkpoint::File::open(&fixture.join("model.safetensors"))?;
    let (weights, pins) = tessera_bundle::pull(&g, &file, 0, &[]).map_err(|e| e.to_string())?;
    Ok((g, Weights::Held(weights), pins))
}

type Joined = (Box<dyn Transport>, Option<Box<dyn Sum>>);

/// The sums' transport: RDMA when ports are given.
fn join(
    args: &Args,
    settings: &tessera_lower::Settings,
) -> Result<Joined, Box<dyn std::error::Error>> {
    let mut met: Box<dyn Transport> = match &args.shm {
        Some(path) => Box::new(shm(path, args.rank, args.ranks as u32, CAPACITY)?),
        None => Box::new(meet(args.rank, &args.addrs, args.listen)?),
    };
    if args.sums == Sums::P2p {
        let sum = p2p(&mut *met, settings, &args.shape)?;
        return Ok((met, Some(sum)));
    }
    if args.sums == Sums::Grid {
        let grid = match (&args.shm, &args.fabric) {
            (Some(path), _) => {
                let (rank, ranks) = (args.rank, args.ranks as u32);
                let regions = sums::regions(ranks);
                let room = ShmLink::room(ranks, &regions);
                let file = shared(&path.with_extension("grid"), rank, ranks, room)?;
                sums::open(settings.gpu(), ShmLink::over(rank, ranks, file, &regions)?)?
            }
            (None, Some(fabric)) => tessera_frontend_comm::grid(&mut *met, fabric, settings)?,
            (None, None) => return Err("--sums grid needs --shm or --verbs".into()),
        };
        return Ok((met, Some(grid)));
    }
    let [sums, _] = tessera_frontend_comm::join(met, args.fabric.as_ref())?;
    Ok((sums, None))
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut settings = tessera_frontend_settings::read();
    if args.host {
        settings.backend = Pick::Cpu;
    }
    let fixture = match &args.fixture {
        Some(dir) => dir.clone(),
        None => tessera_testkit::reference(&args.model)
            .ok_or_else(|| format!("no fixture: {} has no reference here", args.model))?,
    };
    let case = ["", "cases/prompt", "cases/words"]
        .map(|at| fixture.join(at))
        .into_iter()
        .find(|dir| dir.join("tokens.txt").is_file())
        .ok_or("no tokens.txt in the fixture")?;
    let text = std::fs::read_to_string(case.join("tokens.txt"))?;
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .ok_or("no tokens")?;
    let tokens: Vec<u32> = line
        .split_whitespace()
        .map(|t| t.parse())
        .collect::<Result<_, _>>()?;
    let n = tokens.len();
    let ranks = args.ranks;
    let (g, weights, mut pins) = model(&args.model, &fixture, n as u64)?;
    if let Some(form) = &args.form {
        let form = tessera_lower::form::named(form).ok_or_else(|| format!("no form `{form}`"))?;
        pins = tessera_lower::Pins::all(form);
    }
    let parts = match args.parts.is_empty() {
        true => tessera_graph::Parts::even(ranks),
        false => tessera_graph::Parts(args.parts.clone()),
    };
    let tessera_graph::Sharded {
        graph: sharded,
        whole,
    } = shard_aligned(&g, &parts, &pins.shares(&g))?;
    for w in &whole {
        let path = &g.weights[w.weight as usize].path;
        eprintln!("rank {}: {path:?} stays whole: {}", args.rank, w.why);
    }
    let split = sharded
        .weights
        .iter()
        .filter(|w| w.ty.placement != tessera_graph::Placement::Replicated);
    eprintln!(
        "rank {}: {} of {} weights split over {ranks} ranks",
        args.rank,
        split.count(),
        g.weights.len()
    );

    let device = device(&settings, pins, ranks, args.host).as_rank(args.rank);
    let device = device.uneven(parts.0);
    let sizes = [(Seqs, 1), (Context, n as u64)];
    let (transport, grid) = join(args, &settings)?;
    let sharded = sharded.wired(sum_wire(args.sum_wire, grid.as_deref()));
    let start = Instant::now();
    let plan = lower(&sharded, 0, &sizes, &device)?;
    let sums = plan
        .steps
        .iter()
        .filter(|s| s.kernel.name() == "all_reduce")
        .count();
    let held: usize = plan.weights.iter().map(|l| l.slot.bytes()).sum();
    eprintln!(
        "rank {}: {} steps, {sums} sums, {:.1} MB of weights, lowered in {:.1} s",
        args.rank,
        plan.steps.len(),
        held as f64 / 1e6,
        start.elapsed().as_secs_f64()
    );

    let mut rank = Rank::open(plan, &settings, transport)?.with_grid(grid)?;
    let sums = match rank.on_device() {
        true => "on the device",
        false => "on the host",
    };
    eprintln!("rank {}: on {}, sums {sums}", args.rank, rank.device());
    let start = Instant::now();
    let count = g.weights.len() as u32;
    let shares: Vec<_> = (0..count).map(|at| rank.plan().share(at)).collect();
    let shares = |at: u32| shares[at as usize];
    let loading = weights.each_share(shares, |at, w| rank.load_weight(at, w));
    loading.map_err(|e| e.to_string())?;
    eprintln!(
        "rank {}: loaded in {:.1} s",
        args.rank,
        start.elapsed().as_secs_f64()
    );

    let ids = Tensor::u32(vec![1, n], tokens);
    let positions = Tensor::u32(vec![1, n], (0..n as u32).collect());
    let feeds = [ids.view(), positions.view()];
    for _ in 1..args.repeat {
        rank.run(&feeds)?;
    }
    rank.profiling(settings.profile.is_some());
    let start = Instant::now();
    let out = rank.run(&feeds)?;
    let seconds = start.elapsed().as_secs_f64();
    if let Some(profile) = rank.take_profile() {
        eprintln!("rank {}: last forward\n{}", args.rank, profile.report(1));
    }
    let logits = out[0].floats()?;
    // An embedding is one short row.
    let vocab = (logits.len() / n).max(1);
    let top = |row: &[f32]| {
        (0..row.len())
            .max_by(|a, b| row[*a].total_cmp(&row[*b]))
            .expect("a vocab")
    };
    let ours: Vec<usize> = logits.chunks(vocab).map(top).collect();
    eprintln!(
        "rank {}: forward of {n} tokens in {seconds:.3} s, top ids {ours:?}",
        args.rank
    );
    if let Ok(bytes) = std::fs::read(case.join("embedding.f32")) {
        let (words, _) = bytes.as_chunks::<4>();
        let want: Vec<f32> = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let cos = tessera_testkit::cosine_f32(logits, &want);
        eprintln!("rank {}: embedding cos {cos:.5}", args.rank);
    }
    if let Ok(bytes) = std::fs::read(case.join("logits.f32")) {
        let (words, _) = bytes.as_chunks::<4>();
        let want: Vec<f32> = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let theirs: Vec<usize> = want.chunks(vocab).map(top).collect();
        let agree = ours.iter().zip(&theirs).filter(|(a, b)| a == b).count();
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let off = logits
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;
        eprintln!(
            "rank {}: top token agrees on {agree} of {n} rows, logits off {off:.2e}",
            args.rank
        );
    }
    Ok(())
}
