//! Two engines as ranks against one engine alone.

mod common;

use common::{collect_ids, float_weights, host_settings, loopback_links, req, row, two_place};
use tessera_engine::comm::{Sockets, Transport};
use tessera_engine::{Batch, Engine, Fair};
use tessera_graph::Graph;
use tessera_lower::{Recipe, Settings, kept};
use tessera_models::decoder::{Config, Swiglu, stepper};
use tessera_models::mla;
use tessera_run::Profile;
use tessera_trace::Norm;

/// A faux-weight engine, a rank over `links` if any.
fn opened(
    graph: fn() -> Graph,
    links: Option<[Sockets; 2]>,
    recipe: &Recipe,
    context: usize,
    moe_tp: bool,
) -> Engine {
    let settings = Settings {
        profile: Some(1),
        ..host_settings()
    };
    let g = graph();
    let stored = float_weights(&g, 3);
    let mut engine = match links {
        Some([sums, control]) => {
            let control: Box<dyn Transport> = Box::new(control);
            Engine::open_ranked(
                g,
                context,
                (recipe, &settings),
                two_place(moe_tp),
                sums,
                control,
            )
            .unwrap()
        }
        None => Engine::open(g, context, recipe, &settings).unwrap(),
    };
    engine.load(&stored).unwrap();
    engine
}

/// Rank 0 leads, rank 1 follows on a thread.
fn agree(graph: fn() -> Graph, moe_tp: bool) {
    let [[s0, s1], [c0, c1]] = loopback_links();
    let follower = std::thread::spawn(move || {
        opened(graph, Some([s1, c1]), &Recipe::default(), 16, moe_tp)
            .follow()
            .unwrap();
    });
    let mut lead = opened(graph, Some([s0, c0]), &Recipe::default(), 16, moe_tp);
    let mut alone = opened(graph, None, &Recipe::default(), 16, false);
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
            batch.submit(req(&tokens, 3), Box::new(move |e| events.send(e).is_ok()));
            taken.push(receiver);
        }
        while batch.step().unwrap() {}
        for r in taken {
            answers.push(collect_ids(&r.try_iter().collect::<Vec<_>>()));
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
    let [[s0, s1], [c0, c1]] = loopback_links();
    let theirs = recipe.clone();
    let follower = std::thread::spawn(move || {
        opened(graph, Some([s1, c1]), &theirs, 64, false)
            .follow()
            .unwrap();
    });
    let mut lead = opened(graph, Some([s0, c0]), &recipe, 64, false);
    let mut alone = opened(graph, None, &recipe, 64, false);
    assert_eq!(continued(&mut lead), continued(&mut alone));
    lead.dismiss().unwrap();
    follower.join().unwrap();
}
