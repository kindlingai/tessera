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

/// Runs two-rank prefill (`owned` or `split_batch`) against one rank.
fn agree_mode(graph: fn() -> Graph, moe_tp: bool, owned: bool) {
    let [[s0, s1], [c0, c1]] = loopback_links();
    let tune = move |e: &mut Engine| {
        if owned {
            e.set_own_rows(4);
        } else {
            e.set_split_rows(4);
        }
    };
    let follower = std::thread::spawn(move || {
        let mut e = opened(graph, Some([s1, c1]), &Recipe::default(), 16, moe_tp);
        tune(&mut e);
        e.follow().unwrap();
    });
    let mut lead = opened(graph, Some([s0, c0]), &Recipe::default(), 16, moe_tp);
    tune(&mut lead);
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
    if owned {
        assert!(
            kernels.contains_key("reduce_scatter"),
            "missing RS: {kernels:?}"
        );
        assert!(
            kernels.contains_key("gather_rows"),
            "missing GR: {kernels:?}"
        );
        // Odd batches and 1-row decode use all_reduce.
        let odd = [
            row(0, 1, 6),
            row(0, 2, 7),
            row(0, 3, 8),
            row(0, 4, 9),
            row(0, 5, 10),
        ];
        assert_eq!(lead.step(&odd).unwrap(), alone.step(&odd).unwrap());
        assert!(lead.kernels().contains_key("all_reduce"));
        let decode = [row(0, 1, 11)];
        assert_eq!(lead.step(&decode).unwrap(), alone.step(&decode).unwrap());
        assert!(lead.kernels().contains_key("all_reduce"));
    } else {
        assert!(kernels.contains_key("all_reduce"), "missing all_reduce");
        assert!(
            kernels.get("slice_u32").copied().unwrap_or(0) >= 2,
            "expected split_batch plan, got kernels: {kernels:?}"
        );
    }
    lead.dismiss().unwrap();
    follower.join().unwrap();
}

#[test]
fn split_batch_matches_one() {
    agree_mode(
        || stepper::<Norm, Swiglu>(&Config::qwen3_small()),
        false,
        false,
    );
    agree_mode(
        || mla::stepper::<mla::RotatedMoe>(&mla::Config::glm53_small()),
        false,
        false,
    );
    agree_mode(wide_glm53, true, false);
}

#[test]
fn own_rows_matches_one() {
    agree_mode(
        || stepper::<Norm, Swiglu>(&Config::qwen3_small()),
        false,
        true,
    );
    agree_mode(
        || mla::stepper::<mla::RotatedMoe>(&mla::Config::glm53_small()),
        false,
        true,
    );
    agree_mode(wide_glm53, true, true);
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

#[test]
fn ready_waits_for_ranks() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    let recipe = Recipe::default();
    let graph = || stepper::<Norm, Swiglu>(&Config::toy(2, 16));
    let [[sums, others], [control, told]] = loopback_links();
    let done = Arc::new(AtomicBool::new(false));
    let theirs = Arc::clone(&done);
    let r = recipe.clone();
    let follower = std::thread::spawn(move || {
        let mut eng = opened(graph, Some([others, told]), &r, 16, false);
        std::thread::sleep(Duration::from_millis(80));
        theirs.store(true, Ordering::Release);
        eng.ready().unwrap();
        eng.follow().unwrap();
    });
    let mut lead = opened(graph, Some([sums, control]), &recipe, 16, false);
    lead.ready().unwrap();
    assert!(done.load(Ordering::Acquire));
    lead.dismiss().unwrap();
    follower.join().unwrap();

    // Follower drop fails the ready barrier instead of hanging.
    let [[s2, _], [c2, t2]] = loopback_links();
    drop(t2);
    let mut lead = opened(graph, Some([s2, c2]), &recipe, 16, false);
    assert!(lead.ready().is_err());
}

fn open_ranked_pair(settings: Settings) -> (Engine, std::thread::JoinHandle<()>) {
    let graph = || stepper::<Norm, Swiglu>(&Config::toy(1, 16));
    let [[s0, s1], [c0, c1]] = loopback_links();
    let f_settings = settings.clone();
    let follower = std::thread::spawn(move || {
        let g = graph();
        let stored = float_weights(&g, 3);
        let control: Box<dyn Transport> = Box::new(c1);
        let mut eng = Engine::open_ranked(
            g,
            16,
            (&Recipe::default(), &f_settings),
            two_place(false),
            s1,
            control,
        )
        .unwrap();
        eng.load(&stored).unwrap();
        eng.follow().unwrap();
    });
    let g = graph();
    let stored = float_weights(&g, 3);
    let control: Box<dyn Transport> = Box::new(c0);
    let mut lead = Engine::open_ranked(
        g,
        16,
        (&Recipe::default(), &settings),
        two_place(false),
        s0,
        control,
    )
    .unwrap();
    lead.load(&stored).unwrap();
    (lead, follower)
}

#[test]
fn ranked_profile_survives_park() {
    use tessera_engine::{Asks, Event, Sink};

    let asks = Asks::default();
    let (send, got) = std::sync::mpsc::channel();
    let noop = || -> Sink { Box::new(|_| true) };
    let (mut eng1, f1) = open_ranked_pair(host_settings());
    eng1.profile_into(asks.clone(), Box::new(move |r| send.send(r).unwrap()));
    let mut b1 = Batch::new(&mut eng1, Box::new(Fair));
    let ask = asks.hold();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    b1.submit(
        req(&[3, 9, 4], 4),
        Box::new(move |ev| {
            if matches!(ev, Event::Done(_)) {
                let _ = done_tx.send(());
            }
            true
        }),
    );
    assert!(b1.step().unwrap());
    assert!(b1.busy() && got.try_recv().is_err());
    let parked = b1.park().unwrap();
    // Ranked park flushes pre-park segment across all ranks.
    let seg1 = got.try_recv().unwrap();
    assert_eq!(seg1.launches, 1);
    assert_eq!(seg1.profiles.len(), 2);
    assert!(seg1.profiles.iter().all(Option::is_some));
    eng1.dismiss().unwrap();
    f1.join().unwrap();
    drop(eng1);

    // Resuming onto default-settings ranked engine keeps asks and sink.
    let (mut eng2, f2) = open_ranked_pair(host_settings());
    let mut b2 = Batch::resumed(&mut eng2, Box::new(Fair), parked);
    while b2.step().unwrap() {}
    done_rx.try_recv().unwrap();
    assert!(got.try_recv().is_err());
    drop(ask);
    assert!(!b2.step().unwrap());
    let seg2 = got.try_recv().unwrap();
    assert_eq!(seg2.launches, 3);
    assert_eq!(seg2.profiles.len(), 2);
    assert!(seg2.profiles.iter().all(Option::is_some));

    // Resuming onto asks-configured ranked engine still restores original sink.
    let ask2 = asks.hold();
    b2.submit(req(&[3, 9, 4], 4), noop());
    assert!(b2.step().unwrap());
    let parked2 = b2.park().unwrap();
    assert_eq!(got.try_recv().unwrap().launches, 1);
    eng2.dismiss().unwrap();
    f2.join().unwrap();
    drop(eng2);

    let with_asks = Settings {
        asks: Some(asks.clone()),
        ..host_settings()
    };
    let (mut eng3, f3) = open_ranked_pair(with_asks);
    let mut b3 = Batch::resumed(&mut eng3, Box::new(Fair), parked2);
    while b3.step().unwrap() {}
    assert!(got.try_recv().is_err());
    drop(ask2);
    assert!(!b3.step().unwrap());
    let seg3 = got.try_recv().unwrap();
    assert_eq!(seg3.launches, 3);
    assert_eq!(seg3.profiles.len(), 2);
    assert!(seg3.profiles.iter().all(Option::is_some));
    eng3.dismiss().unwrap();
    f3.join().unwrap();
}
