//! A recipe's resident grid over ranks, against one engine alone.

use crate::Args;
use std::error::Error;
use std::path::Path;
use std::time::Instant;
use tessera_device::whole;
use tessera_engine::wasm::Module;
use tessera_engine::{Engine, Place, Row};
use tessera_engine_comm::{Link, ShmLink, Transport, VerbsLink};
use tessera_frontend_comm::{CAPACITY, join, meet, shared, shm};
use tessera_graph::Graph;
use tessera_lower::{Recipe, Settings, Weight};
use tessera_models::decoder::{Swiglu, stepper};
use tessera_testkit::{Noise, weights};
use tessera_trace::Norm;

type Outcome = Result<(), Box<dyn Error>>;
/// Each launch's ids, and seconds per launch.
type Timed = Result<(Vec<Vec<u32>>, f64), Box<dyn Error>>;

const CONTEXT: usize = 64;
/// Under GB10 MPS, two large grids ran in turn.
const MPS: u32 = 16;
/// Decode steps after the prompt, timed.
const STEPS: u32 = 32;

fn graph() -> Graph {
    stepper::<Norm, Swiglu>(&tessera_models::small::qwen3())
}

/// The recipe's choices, with its module's rules.
fn choices(dir: &Path) -> Result<Recipe, Box<dyn Error>> {
    let recipe = tessera_recipe::Recipe::read(dir)?;
    let bytes = recipe.module_bytes(Some(Path::new("target/recipes")))?;
    let module = Module::open(&bytes.ok_or("the recipe has no module")?)?;
    let mut lower = recipe.lower.unwrap_or_default();
    lower.rules.extend(module.rules);
    Ok(lower)
}

/// A faux-weight engine: alone, or a rank over two transports.
fn engine(
    lower: &Recipe,
    settings: &Settings,
    ranked: Option<[Box<dyn Transport>; 2]>,
) -> Result<Engine, Box<dyn Error>> {
    let g = graph();
    let stored: Vec<Weight> = weights(&g, &mut Noise(3))
        .into_iter()
        .map(Weight::Floats)
        .collect();
    let mut engine = match ranked {
        Some([sums, control]) => {
            let place = Place::mesh(sums.ranks() as usize, Vec::new(), true);
            Engine::open_ranked(g, CONTEXT, (lower, settings), place, sums, control)?
        }
        None => Engine::open(g, CONTEXT, lower, settings)?,
    };
    engine.load(&stored)?;
    Ok(engine)
}

fn row(slot: u32, token: u32, position: u32) -> Row {
    Row {
        slot,
        token,
        position,
    }
}

/// A 20-row prompt, then single-row steps in turn.
fn launches() -> Vec<Vec<Row>> {
    let prompt = (0..20).map(|p| row(0, (p * 7 + 3) % 64, p)).collect();
    let steps = (0..STEPS).map(|p| vec![row(0, (p * 5 + 1) % 64, 20 + p)]);
    std::iter::once(prompt).chain(steps).collect()
}

/// The grid's link over the file or the fabric.
fn grid(
    args: &Args,
    met: &mut dyn Transport,
    settings: &Settings,
) -> Result<Box<dyn tessera_engine::Grid>, Box<dyn Error>> {
    let (rank, ranks) = (args.rank, args.ranks as u32);
    let regions = whole::regions(ranks);
    let gpu = settings.gpu();
    match (&args.shm, &args.fabric) {
        (Some(path), _) => {
            let room = ShmLink::room(ranks, &regions);
            let file = shared(&path.with_extension("grid"), rank, ranks, room)?;
            let link = ShmLink::over(rank, ranks, file, &regions)?;
            Ok(whole::open(gpu, link, ranks * MPS)?)
        }
        (None, Some(fabric)) => {
            let pinning = tessera_engine::pinning(settings);
            let relay = whole::RELAY;
            let link = VerbsLink::connect(rank, ranks, fabric, &regions, relay, met, pinning)?;
            eprintln!("rank {rank}: grid over {} ranks by RDMA", link.ranks());
            Ok(whole::open(gpu, link, 1)?)
        }
        (None, None) => Err("a resident grid needs --shm or --verbs".into()),
    }
}

/// Seconds per launch of each, from the second launch on.
fn timed(engine: &mut Engine, launches: &[Vec<Row>]) -> Timed {
    let mut ids = Vec::with_capacity(launches.len());
    let mut start = Instant::now();
    for (i, rows) in launches.iter().enumerate() {
        if i == 1 {
            start = Instant::now();
        }
        ids.push(engine.step(rows)?);
    }
    let steps = launches.len().saturating_sub(1).max(1);
    Ok((ids, start.elapsed().as_secs_f64() / steps as f64))
}

/// Runs this rank. Rank 0 also runs the graph alone.
pub fn run(args: &Args, dir: &Path) -> Outcome {
    let settings = tessera_frontend_settings::read();
    let lower = choices(dir)?;
    let (rank, ranks) = (args.rank, args.ranks as u32);
    let mut met: Box<dyn Transport> = match &args.shm {
        Some(path) => Box::new(shm(path, rank, ranks, CAPACITY)?),
        None => Box::new(meet(rank, &args.addrs, args.listen)?),
    };
    let grid = grid(args, &mut *met, &settings)?;
    let mut ranked = engine(&lower, &settings, Some(join(met, None)?))?;
    ranked.persist(grid)?;
    if rank != 0 {
        ranked.follow()?;
        eprintln!("rank {rank}: followed");
        return Ok(());
    }
    let launches = launches();
    let (got, grid_s) = timed(&mut ranked, &launches)?;
    ranked.dismiss()?;
    let mut alone = engine(&Recipe::default(), &settings, None)?;
    let (want, alone_s) = timed(&mut alone, &launches)?;
    let same = got.iter().zip(&want).filter(|(a, b)| a == b).count();
    for (i, (a, b)) in got
        .iter()
        .zip(&want)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
    {
        eprintln!("rank 0: launch {i}: grid {a:?}, alone {b:?}");
    }
    let kernels = ranked.kernels();
    eprintln!("rank 0: kernels {kernels:?}");
    eprintln!(
        "rank 0: {same} of {} launches agree, last ids {:?} and {:?}",
        launches.len(),
        got.last(),
        want.last()
    );
    eprintln!(
        "rank 0: {:.3} ms per step in the grid over {ranks} ranks, {:.3} ms alone",
        grid_s * 1e3,
        alone_s * 1e3
    );
    match same == launches.len() {
        true => Ok(()),
        false => Err("the ranks' ids differ from the engine alone".into()),
    }
}
