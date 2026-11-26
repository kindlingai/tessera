//! Helpers for multi-slot DFlash batching tests.

mod batch_test;

use super::*;

fn toy_pair(ctx: usize) -> (Graph, Graph) {
    let mut cfg = Config::toy(2, ctx as u64);
    cfg.tied = false;
    cfg.stack.taps = vec![1, 2];
    let spec = target_spec(cfg.stack.vocab, cfg.stack.hidden, cfg.stack.layers, false);
    let d_cfg = dflash::Config::toy(&spec, 4, 0, 2, ctx as u64);
    (
        stepper::<Norm, Swiglu>(&cfg),
        dflash::stepper::<Norm>(&d_cfg),
    )
}

fn paged_recipe(paged: bool) -> Recipe {
    let mut recipe = Recipe::default();
    if paged {
        recipe.kept = &tessera_lower::kept::PAGED16;
    }
    recipe
}

fn run_solo_refs(
    t_graph: &Graph,
    d_graph: &Graph,
    tw: &[Tensor],
    dw: &[Tensor],
    ctx: usize,
    settings: &Settings,
    prompts: &[(&[u32], usize)],
) -> Vec<Vec<u32>> {
    let mut te = open_target(t_graph, tw, ctx, settings);
    let mut dr = open_drafter(d_graph, dw, ctx, 4, 0, settings);
    let mut solo = Speculate::new(&mut te, &mut dr)
        .unwrap()
        .with_prefill_chunk(3);
    prompts
        .iter()
        .map(|&(p, max)| collect_ids(&spec_events(|e| solo.generate(req(p, max), e)).0))
        .collect()
}

fn run_batch_reqs(te: &mut Engine, dr: &mut Drafter, prompts: &[(&[u32], usize)]) -> Vec<Vec<u32>> {
    let mut batch = BatchDFlash::new(te, dr).unwrap().with_prefill_chunk(3);
    let got: Vec<Arc<Mutex<Vec<Event>>>> = prompts
        .iter()
        .map(|&(p, max)| {
            let buf = Arc::new(Mutex::new(Vec::new()));
            let sink = buf.clone();
            batch.submit(
                req(p, max),
                Box::new(move |e| {
                    sink.lock().unwrap().push(e);
                    true
                }),
            );
            buf
        })
        .collect();
    while batch.step().unwrap() {}
    got.iter()
        .map(|b| collect_ids(&b.lock().unwrap()))
        .collect()
}

fn run_reuse_seq(
    batch: &mut BatchDFlash<'_>,
    prompts: &[(&[u32], usize)],
) -> (Vec<Vec<u32>>, Vec<usize>) {
    let (mut ids, mut kept) = (Vec::new(), Vec::new());
    for &(p, max) in prompts {
        let (tx, rx) = std::sync::mpsc::channel();
        batch.submit(req(p, max), Box::new(move |e| tx.send(e).is_ok()));
        while batch.step().unwrap() {}
        let evs: Vec<_> = rx.try_iter().collect();
        kept.push(kept_count(&evs));
        ids.push(collect_ids(&evs));
    }
    (ids, kept)
}

/// Runs `lead` on rank 0 with rank 1 following `BatchDFlash`.
fn with_tp_batch<R>(
    (t_graph, d_graph): (&Graph, &Graph),
    ctx: usize,
    seeds: (u64, u64),
    chunk: Option<usize>,
    reuse: bool,
    lead: impl FnOnce(&mut BatchDFlash<'_>) -> R,
) -> R {
    let settings = host_settings();
    let [[s0, s1], [c0, c1]] = loopback_links();
    let (tg_f, dg_f, s_f) = (t_graph.clone(), d_graph.clone(), settings.clone());
    let follower = std::thread::spawn(move || {
        let (mut te, mut dr) = open_tp_dflash(
            (&tg_f, seeds.0),
            (&dg_f, seeds.1),
            ctx,
            false,
            &s_f,
            (Box::new(s1), c1),
        );
        let mut batch = BatchDFlash::new(&mut te, &mut dr).unwrap();
        if let Some(c) = chunk {
            batch = batch.with_prefill_chunk(c);
        }
        if reuse {
            batch.reuse();
        }
        batch.follow().unwrap();
    });
    let (mut te, mut dr) = open_tp_dflash(
        (t_graph, seeds.0),
        (d_graph, seeds.1),
        ctx,
        false,
        &settings,
        (Box::new(s0), c0),
    );
    let out = {
        let mut batch = BatchDFlash::new(&mut te, &mut dr).unwrap();
        if let Some(c) = chunk {
            batch = batch.with_prefill_chunk(c);
        }
        if reuse {
            batch.reuse();
        }
        lead(&mut batch)
    };
    te.dismiss().unwrap();
    follower.join().unwrap();
    out
}
