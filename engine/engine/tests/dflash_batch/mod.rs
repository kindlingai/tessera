//! Helpers for multi-slot DFlash batching tests.

mod batch_test;

use super::*;

fn toy_pair(ctx: usize) -> (Graph, Graph) {
    let mut cfg = Config::toy(2, ctx as u64);
    cfg.tied = false;
    cfg.stack.taps = vec![1, 2];
    let spec = TargetSpec {
        vocab: cfg.stack.vocab,
        hidden: cfg.stack.hidden,
        layers: cfg.stack.layers,
        tied: false,
    };
    let d_cfg = dflash::Config::toy(&spec, 4, 0, 2, ctx as u64);
    (
        stepper::<Norm, Swiglu>(&cfg),
        dflash::stepper::<Norm>(&d_cfg),
    )
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
        .map(|&(p, max)| {
            let mut evs = Vec::new();
            let req = Request {
                tokens: p.to_vec(),
                max_tokens: max,
                ..Request::default()
            };
            solo.generate(req, &mut |e| {
                evs.push(e);
                true
            })
            .unwrap();
            collect_ids(&evs)
        })
        .collect()
}

fn run_batch_reqs(te: &mut Engine, dr: &mut Drafter, prompts: &[(&[u32], usize)]) -> Vec<Vec<u32>> {
    let mut batch = BatchDFlash::new(te, dr).unwrap().with_prefill_chunk(3);
    let got: Vec<Arc<Mutex<Vec<Event>>>> = prompts
        .iter()
        .map(|&(p, max)| {
            let buf = Arc::new(Mutex::new(Vec::new()));
            let sink = buf.clone();
            let req = Request {
                tokens: p.to_vec(),
                max_tokens: max,
                ..Request::default()
            };
            batch.submit(
                req,
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

fn open_tp_dflash(
    t_graph: &Graph,
    d_graph: &Graph,
    ctx: usize,
    settings: &Settings,
    sum: Box<dyn Transport>,
    ctrl: Sockets,
    seeds: (u64, u64),
) -> (Engine, Drafter) {
    let [ts, ds] = Drafter::share_transport(sum);
    let place = || Place::new(Cluster::two("a", "b"), Parts::even(2));
    let te = open_ranked_target(
        t_graph.clone(),
        ctx,
        settings,
        place(),
        ts,
        ctrl,
        &float_weights(t_graph, seeds.0),
    );
    let mut dr = Drafter::open_ranked(
        d_graph.clone(),
        ctx,
        4,
        0,
        (&Recipe::default(), settings),
        place(),
        ds,
        None,
    )
    .unwrap();
    dr.load(&float_weights(d_graph, seeds.1)).unwrap();
    (te, dr)
}

fn run_reuse_seq(
    batch: &mut BatchDFlash<'_>,
    prompts: &[(&[u32], usize)],
) -> (Vec<Vec<u32>>, Vec<usize>) {
    let (mut ids, mut kept) = (Vec::new(), Vec::new());
    for &(p, max) in prompts {
        let evs = Arc::new(Mutex::new(Vec::new()));
        let s = evs.clone();
        batch.submit(
            Request {
                tokens: p.to_vec(),
                max_tokens: max,
                ..Request::default()
            },
            Box::new(move |e| {
                s.lock().unwrap().push(e);
                true
            }),
        );
        while batch.step().unwrap() {}
        let evs = evs.lock().unwrap();
        kept.push(
            evs.iter()
                .find_map(|e| match *e {
                    Event::Kept(n) => Some(n),
                    _ => None,
                })
                .unwrap_or(0),
        );
        ids.push(collect_ids(&evs));
    }
    (ids, kept)
}
