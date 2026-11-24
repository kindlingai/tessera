//! Two engines as ranks against one engine alone.

use std::net::{TcpListener, TcpStream};
use tessera_engine::comm::{Sockets, Transport};
use tessera_engine::{Batch, Engine, Event, Fair, Place, Request, Row};
use tessera_graph::{Graph, Parts};
use tessera_lower::{Cluster, Pick, Recipe, Settings, Weight, kept};
use tessera_models::decoder::{Config, Swiglu, stepper};
use tessera_models::mla;
use tessera_run::Profile;
use tessera_testkit::{Noise, weights};
use tessera_trace::Norm;

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
fn engine(graph: fn() -> Graph, links: Option<[Sockets; 2]>, moe_tp: bool) -> Engine {
    opened(graph, links, &Recipe::default(), 16, moe_tp)
}

/// As `engine`, with a recipe and a context.
fn opened(
    graph: fn() -> Graph,
    links: Option<[Sockets; 2]>,
    recipe: &Recipe,
    context: usize,
    moe_tp: bool,
) -> Engine {
    let settings = Settings {
        backend: Pick::Host,
        profile: Some(1),
        ..Settings::default()
    };
    let g = graph();
    let stored = weights(&g, &mut Noise(3)).into_iter().map(Weight::Floats);
    let stored: Vec<Weight> = stored.collect();
    let mut engine = match links {
        Some([sums, control]) => {
            let place = Place::new(Cluster::two("a", "b"), Parts::even(2)).with_moe_tp(moe_tp);
            let control: Box<dyn Transport> = Box::new(control);
            Engine::open_ranked(g, context, (recipe, &settings), place, sums, control).unwrap()
        }
        None => Engine::open(g, context, recipe, &settings).unwrap(),
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
fn agree(graph: fn() -> Graph, moe_tp: bool) {
    let ([sums, others], [control, told]) = (pair(), pair());
    let follower = std::thread::spawn(move || {
        engine(graph, Some([others, told]), moe_tp)
            .follow()
            .unwrap();
    });
    let mut lead = engine(graph, Some([sums, control]), moe_tp);
    let mut alone = engine(graph, None, false);
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
    let ranks = lead.ranks().unwrap();
    let reduced = |p: &Option<Profile>| {
        p.as_ref()
            .is_some_and(|p| p.spent.contains_key("all_reduce: wait"))
    };
    assert!(ranks.profiles.len() == 2 && ranks.profiles.iter().all(reduced));
}

fn wide_glm53() -> Graph {
    let mut c = mla::Config::glm53_small();
    c.stack.ffn = 512;
    mla::stepper::<mla::RotatedMoe>(&c)
}

#[test]
fn ranks_match_one() {
    agree(|| stepper::<Norm, Swiglu>(&Config::qwen3_small()), false);
    agree(
        || mla::stepper::<mla::RotatedMoe>(&mla::Config::glm53_small()),
        false,
    );
    agree(wide_glm53, true);
}

/// Answers to a prompt, then two that continue it.
fn continued(engine: &mut Engine) -> Vec<Vec<u32>> {
    let mut batch = Batch::new(engine, Box::new(Fair));
    batch.reuse();
    batch.budget(1);
    let base: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 32).collect();
    let later = [
        [&base[..18], &[1, 2]].concat(),
        [&base[..17], &[5]].concat(),
    ];
    let mut answers = Vec::new();
    for round in [vec![base.clone()], later.to_vec()] {
        let mut taken = Vec::new();
        for tokens in round {
            let (events, receiver) = std::sync::mpsc::channel();
            let request = Request {
                tokens,
                max_tokens: 3,
                ..Request::default()
            };
            batch.submit(request, Box::new(move |e| events.send(e).is_ok()));
            taken.push(receiver);
        }
        while batch.step().unwrap() {}
        for r in taken {
            let id = |e: Event| match e {
                Event::Token(id) => Some(id),
                _ => None,
            };
            answers.push(r.try_iter().filter_map(id).collect());
        }
    }
    answers
}

#[test]
fn ranks_share_pages() {
    let recipe = Recipe {
        kept: &kept::PAGED16,
        ..Recipe::default()
    };
    let graph = || stepper::<Norm, Swiglu>(&Config::toy(2, 64));
    let ([sums, others], [control, told]) = (pair(), pair());
    let theirs = recipe.clone();
    let follower = std::thread::spawn(move || {
        opened(graph, Some([others, told]), &theirs, 64, false)
            .follow()
            .unwrap();
    });
    let mut lead = opened(graph, Some([sums, control]), &recipe, 64, false);
    let mut alone = opened(graph, None, &recipe, 64, false);
    assert_eq!(continued(&mut lead), continued(&mut alone));
    lead.dismiss().unwrap();
    follower.join().unwrap();
}
