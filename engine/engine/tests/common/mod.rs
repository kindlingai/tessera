//! Shared helpers for speculative decoding integration tests.

use std::time::Duration;
use tessera_engine::comm::Sockets;
use tessera_engine::{Engine, Event, Place, Recipe};
use tessera_graph::Graph;
use tessera_interp::Tensor;
use tessera_lower::{Settings, Weight};
use tessera_testkit::{Noise, tcp_pair, tcp_timeouts, weights};

pub fn collect_ids(events: &[Event]) -> Vec<u32> {
    let mut out = Vec::new();
    for e in events {
        if let Event::Token(id) = e {
            out.push(*id);
        }
    }
    out
}

pub fn float_weights(g: &Graph, seed: u64) -> Vec<Weight> {
    weights(g, &mut Noise(seed))
        .into_iter()
        .map(Weight::Floats)
        .collect()
}

pub fn open_target(g: &Graph, w: &[Tensor], ctx: usize, s: &Settings) -> Engine {
    let mut engine = Engine::open(g.clone(), ctx, &Recipe::default(), s).unwrap();
    let stored: Vec<Weight> = w.iter().cloned().map(Weight::Floats).collect();
    engine.load(&stored).unwrap();
    engine
}

pub fn loopback_links() -> [[Sockets; 2]; 2] {
    let wait = Duration::from_secs(20);
    std::array::from_fn(|_| {
        let (a, b) = tcp_pair().unwrap();
        tcp_timeouts(&a, wait).unwrap();
        tcp_timeouts(&b, wait).unwrap();
        [
            Sockets::over(0, vec![None, Some(a)]).unwrap(),
            Sockets::over(1, vec![Some(b), None]).unwrap(),
        ]
    })
}

pub fn open_ranked_target<C: tessera_engine::comm::Transport + 'static>(
    g: Graph,
    ctx: usize,
    settings: &Settings,
    place: Place,
    sums: Box<dyn tessera_engine::comm::Transport>,
    control: C,
    w: &[Weight],
) -> Engine {
    let mut engine = Engine::open_ranked(
        g,
        ctx,
        (&Recipe::default(), settings),
        place,
        sums,
        Box::new(control),
    )
    .unwrap();
    engine.load(w).unwrap();
    engine
}
