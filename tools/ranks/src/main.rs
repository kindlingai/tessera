//! One rank of Qwen3-0.6B, sharded over TCP, against the fixture.

// Linking these registers their lowering rules.
use sensai_rules as _;

use sensai_engine_comm::{Fabric, Transport};
use sensai_frontend_comm::{CAPACITY, meet, ports, shm};
use sensai_graph::DimName::{Context, Seqs};
use sensai_graph::shard;
use sensai_interp::Tensor;
use sensai_lower::{Cluster, Device, Pick, Weight, ZoneKind, lower};
use sensai_models::decoder::{Config, Moe, Swiglu, program};
use sensai_run::{Rank, Run};
use sensai_trace::Norm;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "tessera-ranks --rank N (--addrs HOST:PORT,... | --shm PATH --ranks N)
  [--model qwen3|granite|eg2|deepseek|glm53] [--fixture DIR] [--listen ADDR] [--backend host|gpu] [--parts 3,1]
  [--form q8_0] [--verbs DEV[:GID],... | --ring PREVDEV,.../NEXTDEV,...] [--repeat N]
  --verbs and --ring move the sums over RDMA, one lane per device, after the
  ranks meet over the sockets or the file.
  The fixture holds model.safetensors, and tokens.txt with logits.f32,
  at its root or under cases/prompt. Its default is ~/.cache/tessera/references/<model>.";

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
    };
    while let Some(flag) = given.next() {
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

#[cfg(test)]
mod eg2_test;
#[cfg(test)]
mod mla_test;

fn main() -> ExitCode {
    let Some(args) = parse(std::env::args().skip(1)) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tessera-ranks: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `ranks` linked zones on the host or the GPU.
fn device(
    settings: &sensai_lower::Settings,
    pins: sensai_lower::Pins,
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

/// A model's weights: held, or read as they load.
enum Weights {
    Held(Vec<Weight>),
    Streamed(Box<sensai_bundle::mla::Release>),
}

impl Weights {
    /// Each weight in graph order, one held at a time.
    fn each(
        &self,
        mut take: impl FnMut(u32, &Weight) -> Result<(), sensai_run::Error>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Weights::Held(all) => {
                for (at, w) in all.iter().enumerate() {
                    take(at as u32, w)?;
                }
            }
            Weights::Streamed(release) => release
                .each(|at, w| Ok(take(at, &w)?))
                .map_err(|e| e.to_string())?,
        }
        Ok(())
    }
}

type Loaded = (sensai_graph::Graph, Weights, sensai_lower::Pins);

/// One sequence's graph for `n` tokens, its weights and forms.
fn model(name: &str, fixture: &Path, n: u64) -> Result<Loaded, Box<dyn std::error::Error>> {
    let g = match name {
        "qwen3" => program::<Norm, Swiglu>(&Config::qwen3(1, n)),
        "granite" => program::<(), Moe>(&Config::granite(1, n)),
        "deepseek" | "glm53" => {
            let r = sensai_bundle::mla::release(fixture, 1, n, 3).map_err(|e| e.to_string())?;
            let (graph, pins) = (r.graph.clone(), r.pins.clone());
            return Ok((graph, Weights::Streamed(Box::new(r)), pins));
        }
        "eg2" => {
            let e = sensai_bundle::encoder::encoder(&fixture.join("exlr8"), [1, n], false)
                .map_err(|e| e.to_string())?;
            let pins = sensai_lower::Pins::default();
            return Ok((e.graph, Weights::Held(e.weights), pins));
        }
        _ => return Err("a model this tool knows".into()),
    };
    let file = sensai_checkpoint::File::open(&fixture.join("model.safetensors"))?;
    let (weights, pins) = sensai_bundle::pull(&g, &file, 0, &[]).map_err(|e| e.to_string())?;
    Ok((g, Weights::Held(weights), pins))
}

/// The sums' transport: RDMA when ports are given.
fn join(
    args: &Args,
    settings: &sensai_lower::Settings,
) -> Result<Box<dyn Transport>, Box<dyn std::error::Error>> {
    let met: Box<dyn Transport> = match &args.shm {
        Some(path) => Box::new(shm(path, args.rank, args.ranks as u32, CAPACITY)?),
        None => Box::new(meet(args.rank, &args.addrs, args.listen)?),
    };
    let [sums, _] = sensai_frontend_comm::join(met, args.fabric.as_ref(), settings)?;
    Ok(sums)
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut settings = sensai_frontend_settings::read();
    if args.host {
        settings.backend = Pick::Cpu;
    }
    let fixture = match &args.fixture {
        Some(dir) => dir.clone(),
        None => sensai_testkit::reference(&args.model)
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
    let parts = match args.parts.is_empty() {
        true => sensai_graph::Parts::even(ranks),
        false => sensai_graph::Parts(args.parts.clone()),
    };
    let sensai_graph::Sharded {
        graph: sharded,
        whole,
    } = shard(&g, &parts)?;
    for w in &whole {
        let path = &g.weights[w.weight as usize].path;
        eprintln!("rank {}: {path:?} stays whole: {}", args.rank, w.why);
    }
    let split = sharded
        .weights
        .iter()
        .filter(|w| w.ty.placement != sensai_graph::Placement::Replicated);
    eprintln!(
        "rank {}: {} of {} weights split over {ranks} ranks",
        args.rank,
        split.count(),
        g.weights.len()
    );

    if let Some(form) = &args.form {
        let form = sensai_lower::form::named(form).ok_or_else(|| format!("no form `{form}`"))?;
        pins = sensai_lower::Pins::all(form);
    }
    let device = device(&settings, pins, ranks, args.host).as_rank(args.rank);
    let device = device.uneven(parts.0);
    let sizes = [(Seqs, 1), (Context, n as u64)];
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

    let transport = join(args, &settings)?;
    let mut rank = Rank::open(plan, &settings, transport)?;
    let sums = match rank.on_device() {
        true => "on the device",
        false => "on the host",
    };
    eprintln!("rank {}: on {}, sums {sums}", args.rank, rank.device());
    let start = Instant::now();
    // Each rank cuts its part from the packed bytes.
    weights.each(|at, w| rank.load_weight(at, w))?;
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
    let start = Instant::now();
    let out = rank.run(&feeds)?;
    let seconds = start.elapsed().as_secs_f64();
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
        let cos = sensai_testkit::cosine_f32(logits, &want);
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
