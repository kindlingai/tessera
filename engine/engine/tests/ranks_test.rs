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

/// Runs split-batch two-rank prefill against one unsplit rank.
fn agree_split(graph: fn() -> Graph, moe_tp: bool) {
    let [[s0, s1], [c0, c1]] = loopback_links();
    let follower = std::thread::spawn(move || {
        let mut e = opened(graph, Some([s1, c1]), &Recipe::default(), 16, moe_tp);
        e.set_split_rows(4);
        e.follow().unwrap();
    });
    let mut lead = opened(graph, Some([s0, c0]), &Recipe::default(), 16, moe_tp);
    lead.set_split_rows(4);
    let mut alone = opened(graph, None, &Recipe::default(), 16, false);
    let launches = [
        vec![row(0, 5, 0), row(0, 7, 1), row(0, 9, 2), row(0, 2, 3)],
        vec![row(1, 3, 0), row(1, 8, 1), row(0, 4, 4), row(0, 6, 5)],
    ];
    for rows in &launches {
        assert_eq!(lead.step(rows).unwrap(), alone.step(rows).unwrap());
        assert_eq!(lead.last(rows).unwrap(), alone.last(rows).unwrap());
    }
    let kernels = lead.kernels();
    assert!(kernels.contains_key("all_reduce"), "missing all_reduce");
    assert!(
        kernels.get("slice_u32").copied().unwrap_or(0) >= 2,
        "expected split_batch plan, got kernels: {kernels:?}"
    );
    lead.dismiss().unwrap();
    follower.join().unwrap();
}

#[test]
fn split_batch_matches_one() {
    agree_split(|| stepper::<Norm, Swiglu>(&Config::qwen3_small()), false);
    agree_split(
        || mla::stepper::<mla::RotatedMoe>(&mla::Config::glm53_small()),
        false,
    );
    agree_split(wide_glm53, true);
}

#[test]
fn split_batch_overlaps_glm53() {
    use tessera_graph::{DimName, HALF_ROWS, Parts, shard, split_batch, trim};
    use tessera_lower::{Cluster, Device, lower};
    use tessera_run::testing::host;

    let base = trim(&wide_glm53(), 0).unwrap();
    let parts = Parts::even(2);
    let mut sharded = shard(&base, &parts).unwrap().graph;
    for node in &mut sharded.nodes {
        if let tessera_graph::Op::Row(r) = &node.op
            && r.0.contains(&tessera_graph::Stage::AllReduce)
        {
            node.numerics.get_or_insert_default().wire = Some(tessera_graph::DType::F32);
        }
    }
    let split = split_batch(&sharded, 0).unwrap();
    let device = Device::guess(&host())
        .zoned(Cluster::two("a", "b"))
        .ranked(vec![0, 1])
        .as_rank(0);
    let sizes = [
        (DimName::Seqs, 2),
        (DimName::Rows, 8),
        (HALF_ROWS, 4),
        (DimName::Context, 16),
        (DimName::Picks, 1),
    ];
    let plan = lower(&split, 0, &sizes, &device).unwrap();
    let comm_steps: Vec<usize> = plan
        .steps
        .iter()
        .enumerate()
        .filter(|(_, s)| s.lane == 1)
        .map(|(i, _)| i)
        .collect();
    assert!(
        comm_steps.len() >= 4,
        "expected >= 4 lane-1 comm steps, got {}",
        comm_steps.len()
    );
    // Count lane-1 comms overlapped with independent lane-0 compute.
    let mut overlapped = 0usize;
    for &ci in &comm_steps {
        let out = plan.steps[ci].write;
        let reader = plan.steps[ci + 1..]
            .iter()
            .position(|s| {
                s.reads
                    .iter()
                    .any(|r| r.buffer == out.buffer && r.offset == out.offset)
            })
            .map(|d| ci + 1 + d);
        let hidden = reader
            .map(|ri| {
                plan.steps[ci + 1..ri]
                    .iter()
                    .filter(|s| s.lane == 0)
                    .count()
            })
            .unwrap_or(0);
        if hidden >= 4 {
            overlapped += 1;
        }
    }
    // Every body comm except the final half-B join overlaps compute.
    assert!(
        overlapped >= 9,
        "expected >= 9 overlapped body comms, got {overlapped} of {}",
        comm_steps.len()
    );
}
