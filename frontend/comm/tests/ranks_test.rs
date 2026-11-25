//! Ranks on threads, joined over handles the front end opened.

// Linking these registers their lowering rules.
use tessera_rules as _;

use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use tessera_engine::comm::Transport;
use tessera_frontend_comm::{shm, sockets_within};
use tessera_frontend_settings::read as settings;
use tessera_graph::{DType, Graph, Numerics, PathSeg, shard, sums_ranks};
use tessera_interp::{Tensor, run};
use tessera_lower::{Cluster, Device, Pick, Plan, Recipe, Settings, View, lower};
use tessera_models::decoder::{Config, Swiglu, program};
use tessera_run::{Rank, Run};
use tessera_testkit::{Noise, TCP_TIMEOUT, fill, max_abs_diff, stage, tcp_bind, tcp_pair};
use tessera_trace::DimName::{self, Context, Seqs};
use tessera_trace::{Norm, TensorType};

const SIZES: [(DimName, u64); 2] = [(Seqs, 1), (Context, 8)];

/// A tensor's parts, which may cross to another thread.
type Plain = (Vec<usize>, Vec<u8>, tessera_mem::DType);

fn plain(t: &Tensor) -> Plain {
    (t.shape.clone(), t.bytes().to_vec(), t.dtype())
}

fn rebuilt((shape, bytes, dtype): &Plain) -> Tensor {
    Tensor::from_bytes(shape.clone(), *dtype, bytes)
}

fn carried(tensors: &[Tensor]) -> Vec<Plain> {
    tensors.iter().map(plain).collect()
}

/// `n` listeners on the loopback, with their addresses.
fn listening(n: usize) -> (Vec<TcpListener>, Vec<SocketAddr>) {
    tcp_pair().unwrap();
    let listeners: Vec<TcpListener> = (0..n).map(|_| tcp_bind().unwrap()).collect();
    let addrs = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    (listeners, addrs)
}

/// The toy graph with faux weights and inputs.
fn toy() -> (Graph, Vec<Tensor>, Vec<Tensor>) {
    let g = program::<Norm, Swiglu>(&Config::toy(2, 8));
    let mut noise = Noise(5);
    let mut draw = |ty: &TensorType| fill(&g, ty, &SIZES, &mut noise).unwrap();
    let weights = g.weights.iter().map(|w| draw(&w.ty)).collect();
    let inputs = g.entries[0].inputs.iter().map(|p| draw(&p.ty)).collect();
    (g, weights, inputs)
}

/// The sharded graph planned for each of two host ranks.
fn planned(g: &Graph, host: &Settings) -> [Plan; 2] {
    let device = Device::guess(host)
        .zoned(Cluster::two("a", "b"))
        .ranked(vec![0, 1]);
    let sharded = shard(g, &tessera_graph::Parts::even(2))
        .unwrap()
        .graph
        .wired(tessera_graph::DType::F32);
    [0, 1].map(|r| lower(&sharded, 0, &SIZES, &device.clone().as_rank(r)).unwrap())
}

/// Wires come from model pins, then the recipe, then setup.
#[test]
fn recipe_wires() {
    let load = |name: &str, text: &str| Recipe::load(&stage(name, text.as_bytes())).unwrap();
    assert_eq!(
        load("old.json", r#"{"barred": []}"#).sums,
        Default::default()
    );
    let text = r#"{"sums": {"all": "Bf16", "named": [["*.down", "F32"]]}}"#;
    let recipe = load("wires.json", text);
    let mut g = shard(&toy().0, &tessera_graph::Parts::even(2))
        .unwrap()
        .graph;
    let sums: Vec<usize> = (0..g.nodes.len())
        .filter(|&i| sums_ranks(&g.nodes[i].op))
        .collect();
    // The model pins the first sum.
    g.nodes[sums[0]].numerics = Some(Numerics::pinned(DType::F32));
    let g = recipe.sums.pin(g).wired(DType::F16);
    let mut down = 0;
    for &at in &sums[1..] {
        let mut id = g.nodes[at].args[0];
        while !matches!(g.node(id).op, tessera_graph::Op::Matmul) {
            id = g.node(id).args[0];
        }
        let tessera_graph::Op::Weight(w) = g.node(g.node(id).args[1]).op else {
            panic!("a weight product");
        };
        let is_down = PathSeg::dotted(&g.weights[w as usize].path).ends_with(".down");
        down += usize::from(is_down);
        let want = if is_down { DType::F32 } else { DType::Bf16 };
        assert_eq!(g.nodes[at].numerics.unwrap().wire, Some(want));
    }
    assert_eq!(g.nodes[sums[0]].numerics.unwrap().wire, Some(DType::F32));
    tessera_graph::verify(&g).unwrap();
    assert!(
        down > 0 && down < sums.len() - 1,
        "{down} of {}",
        sums.len()
    );
}

/// One rank of `plan` over `transport`, run once.
fn rank<T: Transport + 'static>(
    transport: T,
    plan: &Plan,
    host: &Settings,
    weights: &[Plain],
    inputs: &[Plain],
) -> Vec<f32> {
    let mut rank = Rank::open(plan.clone(), host, transport).unwrap();
    for (at, w) in weights.iter().enumerate() {
        rank.load(at as u32, rebuilt(w).view()).unwrap();
    }
    let inputs: Vec<Tensor> = inputs.iter().map(rebuilt).collect();
    let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
    // A launch in two halves sums the same.
    rank.feed(&feeds).unwrap();
    let steps = plan.steps.len();
    rank.submit(0..steps / 2).unwrap();
    rank.submit(steps / 2..steps).unwrap();
    let out = rank.read(plan.outputs[0].slot).unwrap();
    out.floats().unwrap().to_vec()
}

/// Both ranks on threads, over the transports `open` gives.
fn both<T: Transport + 'static>(open: impl Fn(u32) -> T + Sync) {
    let (g, weights, inputs) = toy();
    let host = Settings {
        backend: Pick::Host,
        ..settings()
    };
    let plans = planned(&g, &host);
    let (weights_plain, inputs_plain) = (carried(&weights), carried(&inputs));
    let outs: Vec<Vec<f32>> = std::thread::scope(|s| {
        let runs: Vec<_> = (0..2u32)
            .map(|r| {
                let (plan, host, open) = (&plans[r as usize], &host, &open);
                let (weights, inputs) = (&weights_plain, &inputs_plain);
                s.spawn(move || rank(open(r), plan, host, weights, inputs))
            })
            .collect();
        runs.into_iter().map(|t| t.join().unwrap()).collect()
    });
    assert_eq!(outs[0], outs[1]);
    let want = run(&g, &weights, 0, &inputs).unwrap();
    let want = want[0].floats().unwrap();
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let diff = max_abs_diff(&outs[0], want) / scale;
    assert!(diff < 1e-3, "{diff}");
}

/// Ranks on threads over TCP, against one rank.
#[test]
fn two_sockets() {
    let (listeners, addrs) = listening(2);
    let listeners: Vec<_> = listeners.into_iter().map(std::sync::Mutex::new).collect();
    let abort = AtomicBool::new(false);
    both(|r| {
        let listener = listeners[r as usize].lock().unwrap().try_clone().unwrap();
        match sockets_within(r, listener, &addrs, TCP_TIMEOUT, Some(&abort)) {
            Ok(s) => s,
            Err(e) => {
                abort.store(true, Ordering::Release);
                panic!("rank {r} failed to join: {e}");
            }
        }
    });
}

/// Ranks on threads over one mapped file, against one rank.
#[test]
fn two_shm() {
    let path: PathBuf = tessera_testkit::scratch(&format!("ranks-{}", std::process::id()));
    both(|r| shm(&path, r, 2, 1 << 16).unwrap());
    let _ = std::fs::remove_file(&path);
}
