//! Two engines as ranks against one engine alone.

use sensai_engine::comm::{Sockets, Transport};
use sensai_engine::{Engine, Place, Row};
use sensai_graph::{Graph, Parts};
use sensai_lower::{Cluster, Pick, Recipe, Settings, Weight};
use sensai_models::decoder::{Swiglu, stepper};
use sensai_models::{mla, small};
use sensai_testkit::{Noise, weights};
use sensai_trace::Norm;
use std::net::{TcpListener, TcpStream};

/// Both ends of one loopback stream, as two ranks.
fn pair() -> [Sockets; 2] {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let caller = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (answer, _) = listener.accept().unwrap();
    [
        Sockets::over(0, vec![None, Some(caller)]).unwrap(),
        Sockets::over(1, vec![Some(answer), None]).unwrap(),
    ]
}

/// A faux-weight engine, a rank over `links` if any.
fn engine(graph: fn() -> Graph, links: Option<[Sockets; 2]>) -> Engine {
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let (g, recipe) = (graph(), Recipe::default());
    let stored = weights(&g, &mut Noise(3)).into_iter().map(Weight::Floats);
    let stored: Vec<Weight> = stored.collect();
    let mut engine = match links {
        Some([sums, control]) => {
            let place = Place::new(Cluster::two("a", "b"), Parts::even(2));
            let control: Box<dyn Transport> = Box::new(control);
            Engine::open_ranked(g, 16, (&recipe, &settings), place, sums, control).unwrap()
        }
        None => Engine::open(g, 16, &recipe, &settings).unwrap(),
    };
    engine.load(&stored).unwrap();
    engine
}

fn row(slot: u32, token: u32, position: u32) -> Row {
    Row {
        slot,
        token,
        position,
    }
}

/// Rank 0 leads, rank 1 follows on a thread.
fn agree(graph: fn() -> Graph) {
    let ([sums, others], [control, told]) = (pair(), pair());
    let follower = std::thread::spawn(move || {
        engine(graph, Some([others, told])).follow().unwrap();
    });
    let mut lead = engine(graph, Some([sums, control]));
    let mut alone = engine(graph, None);
    let launches = [
        vec![row(1, 5, 0), row(1, 7, 1), row(0, 9, 0)],
        vec![row(0, 4, 1), row(1, 3, 2)],
    ];
    for rows in &launches {
        assert_eq!(lead.step(rows).unwrap(), alone.step(rows).unwrap());
        assert_eq!(lead.last(rows).unwrap(), alone.last(rows).unwrap());
    }
    assert!(lead.kernels().contains_key("all_reduce"));
    lead.dismiss().unwrap();
    follower.join().unwrap();
}

#[test]
fn ranks_match_one() {
    agree(|| stepper::<Norm, Swiglu>(&small::qwen3()));
    agree(|| mla::stepper::<mla::RotatedMoe>(&small::glm53()));
}
