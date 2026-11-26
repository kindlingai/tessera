#![allow(dead_code)]
//! Shared helpers for engine and speculative decoding integration tests.

use std::time::Duration;
use tessera_engine::comm::{Sockets, Transport};
use tessera_engine::{
    Drafter, Engine, Error, Event, MtpDrafter, Place, Reason, Recipe, Request, Row, SpecStats,
};
use tessera_graph::DimName::{Context, Rows, Segments, Seqs};
use tessera_graph::{DimName, Graph, Parts, PathSeg};
use tessera_interp::{Tensor, run, run_ranks};
use tessera_lower::{Cache, Cluster, Device, Pick, Settings, Wants, Weight};
use tessera_testkit::{Noise, tcp_pair, tcp_timeouts, weights};

pub fn row(slot: u32, token: u32, position: u32) -> Row {
    Row {
        slot,
        token,
        position,
    }
}

pub fn step_feeds(rows: &[Row]) -> Vec<Tensor> {
    let col = |of: fn(&Row) -> u32| Tensor::u32(vec![rows.len()], rows.iter().map(of).collect());
    vec![col(|r| r.token), col(|r| r.position), col(|r| r.slot)]
}

pub fn host_settings() -> Settings {
    Settings {
        backend: Pick::Host,
        ..Settings::default()
    }
}

pub fn req(tokens: &[u32], max_tokens: usize) -> Request {
    Request {
        tokens: tokens.to_vec(),
        max_tokens,
        ..Request::default()
    }
}

pub fn collect_ids(events: &[Event]) -> Vec<u32> {
    let mut out = Vec::new();
    for e in events {
        if let Event::Token(id) = e {
            out.push(*id);
        }
    }
    out
}

pub fn kept_count(events: &[Event]) -> usize {
    events
        .iter()
        .find_map(|e| match *e {
            Event::Kept(n) => Some(n),
            _ => None,
        })
        .unwrap_or(0)
}

pub fn gen_events(engine: &mut Engine, request: Request) -> Vec<Event> {
    let mut out = Vec::new();
    engine.generate(request, &mut |e| out.push(e)).unwrap();
    out
}

pub fn gen_ids(engine: &mut Engine, request: Request) -> Vec<u32> {
    collect_ids(&gen_events(engine, request))
}

pub fn spec_events(
    run: impl FnOnce(&mut dyn FnMut(Event) -> bool) -> Result<SpecStats, Error>,
) -> (Vec<Event>, SpecStats) {
    let mut out = Vec::new();
    let stats = run(&mut |e| {
        out.push(e);
        true
    })
    .unwrap();
    (out, stats)
}

pub fn float_weights(g: &Graph, seed: u64) -> Vec<Weight> {
    weights(g, &mut Noise(seed))
        .into_iter()
        .map(Weight::Floats)
        .collect()
}

pub fn zero_last(w: &mut [Tensor]) {
    let last = w.last_mut().unwrap();
    *last = Tensor::f32(last.shape.clone(), vec![0.0; last.len()]);
}

pub fn zero_weight(g: &Graph, w: &mut [Tensor], field: &str) {
    let at = g
        .weights
        .iter()
        .position(|wt| matches!(wt.path.last(), Some(PathSeg::Field(f)) if f == field))
        .unwrap();
    w[at] = Tensor::f32(w[at].shape.clone(), vec![0.0; w[at].len()]);
}

pub fn open_engine(g: &Graph, w: &[Tensor], ctx: usize, recipe: &Recipe, s: &Settings) -> Engine {
    let mut engine = Engine::open(g.clone(), ctx, recipe, s).unwrap();
    let stored: Vec<Weight> = w.iter().cloned().map(Weight::Floats).collect();
    engine.load(&stored).unwrap();
    engine
}

pub fn open_target(g: &Graph, w: &[Tensor], ctx: usize, s: &Settings) -> Engine {
    open_engine(g, w, ctx, &Recipe::default(), s)
}

pub fn open_drafter(
    g: &Graph,
    w: &[Tensor],
    ctx: usize,
    block: usize,
    mask: u32,
    s: &Settings,
) -> Drafter {
    let mut drafter = Drafter::open(g.clone(), ctx, block, mask, &Recipe::default(), s).unwrap();
    let stored: Vec<Weight> = w.iter().cloned().map(Weight::Floats).collect();
    drafter.load(&stored).unwrap();
    drafter
}

pub fn open_mtp(g: &Graph, w: &[Tensor], ctx: usize, s: &Settings) -> MtpDrafter {
    let mut drafter = MtpDrafter::open(g.clone(), ctx, &Recipe::default(), s).unwrap();
    let stored: Vec<Weight> = w.iter().cloned().map(Weight::Floats).collect();
    drafter.load(&stored).unwrap();
    drafter
}

pub fn two_place(moe_tp: bool) -> Place {
    Place::new(Cluster::two("a", "b"), Parts::even(2)).with_moe_tp(moe_tp)
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

pub fn open_ranked_target<C: Transport + 'static>(
    g: Graph,
    ctx: usize,
    settings: &Settings,
    place: Place,
    (sums, control): (Box<dyn Transport>, C),
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

pub fn dim_max(g: &Graph, name: DimName) -> usize {
    g.dims.iter().find(|d| d.name == name).unwrap().size.max() as usize
}

/// Verifies sharded head argmax and tied-head tie-breaking match `g`.
pub fn check_sharded_head(
    g: &Graph,
    sharded: &Graph,
    w: &[Tensor],
    feeds: &[Tensor],
) -> (Vec<Tensor>, Vec<Tensor>) {
    let want = run(g, w, 0, feeds).unwrap();
    let got = run_ranks(sharded, w, 0, feeds, 2).unwrap();
    assert_eq!(got[0].indices().unwrap(), want[0].indices().unwrap());
    let mut tied = w.to_vec();
    zero_weight(g, &mut tied, "head");
    let want_tie = run(g, &tied, 0, feeds).unwrap();
    let got_tie = run_ranks(sharded, &tied, 0, feeds, 2).unwrap();
    assert_eq!(
        got_tie[0].indices().unwrap(),
        want_tie[0].indices().unwrap()
    );
    (want, got)
}

fn open_ranked_dflash(
    g: &Graph,
    seed: u64,
    ctx: usize,
    causal: bool,
    settings: &Settings,
    sums: Box<dyn Transport>,
) -> Drafter {
    let mut dr = Drafter::open_ranked(
        g.clone(),
        ctx,
        4,
        0,
        (&Recipe::default(), settings),
        two_place(false),
        sums,
        None,
    )
    .unwrap();
    dr.set_causal(causal);
    dr.load(&float_weights(g, seed)).unwrap();
    dr
}

fn open_ranked_mtp(
    g: &Graph,
    seed: u64,
    ctx: usize,
    moe_tp: bool,
    settings: &Settings,
    sums: Box<dyn Transport>,
) -> MtpDrafter {
    let mut dr = MtpDrafter::open_ranked(
        g.clone(),
        ctx,
        (&Recipe::default(), settings),
        two_place(moe_tp),
        sums,
    )
    .unwrap();
    dr.load(&float_weights(g, seed)).unwrap();
    dr
}

pub fn open_tp_dflash<C: Transport + 'static>(
    (t_graph, t_seed): (&Graph, u64),
    (d_graph, d_seed): (&Graph, u64),
    ctx: usize,
    causal: bool,
    settings: &Settings,
    (sums, control): (Box<dyn Transport>, C),
) -> (Engine, Drafter) {
    let [ts, ds] = Drafter::share_transport(sums);
    let te = open_ranked_target(
        t_graph.clone(),
        ctx,
        settings,
        two_place(false),
        (ts, control),
        &float_weights(t_graph, t_seed),
    );
    let dr = open_ranked_dflash(d_graph, d_seed, ctx, causal, settings, ds);
    (te, dr)
}

pub fn open_tp_mtp<C: Transport + 'static>(
    (t_graph, t_seed): (&Graph, u64),
    (m_graph, m_seed): (&Graph, u64),
    ctx: usize,
    moe_tp: bool,
    settings: &Settings,
    (sums, control): (Box<dyn Transport>, C),
) -> (Engine, MtpDrafter) {
    let [ts, ms] = Drafter::share_transport(sums);
    let te = open_ranked_target(
        t_graph.clone(),
        ctx,
        settings,
        two_place(moe_tp),
        (ts, control),
        &float_weights(t_graph, t_seed),
    );
    let dr = open_ranked_mtp(m_graph, m_seed, ctx, moe_tp, settings, ms);
    (te, dr)
}

pub fn open_tp_dual<C: Transport + 'static>(
    (t_graph, d_graph, m_graph): (&Graph, &Graph, &Graph),
    (t_seed, d_seed, m_seed): (u64, u64, u64),
    ctx: usize,
    settings: &Settings,
    (sums, control): (Box<dyn Transport>, C),
) -> (Engine, Drafter, MtpDrafter) {
    let [ts, ds, ms] = Drafter::share_transports(sums);
    let te = open_ranked_target(
        t_graph.clone(),
        ctx,
        settings,
        two_place(false),
        (ts, control),
        &float_weights(t_graph, t_seed),
    );
    let dr = open_ranked_dflash(d_graph, d_seed, ctx, false, settings, ds);
    let mtp = open_ranked_mtp(m_graph, m_seed, ctx, false, settings, ms);
    (te, dr, mtp)
}

/// Returns `all_gather` output lengths for a 2-rank sharded plan.
pub fn plan_gathers(g: &Graph, parts: &Parts, rows: u64, wants: Wants) -> Vec<usize> {
    let dev = Device::guess(&host_settings())
        .zoned(Cluster::two("a", "b"))
        .ranked(vec![0, 1])
        .uneven(parts.0.clone())
        .as_rank(0);
    let mut cache = Cache::default();
    let dims = [(Seqs, 1), (Rows, rows), (Context, rows), (Segments, 1)];
    let (plan, _) = cache
        .wanting(g, 0, wants, &dims, &[Context, Segments], &dev)
        .unwrap();
    plan.steps
        .iter()
        .filter(|s| s.kernel.name() == "all_gather")
        .map(|s| s.write.len)
        .collect()
}

/// Checks length, stop-token, and context-tail agreement against `baseline`.
pub fn check_spec(
    baseline: &mut Engine,
    mut run: impl FnMut(Request) -> (Vec<Event>, SpecStats),
    prompt: &[u32],
    [max_tokens, stop_at, overlong]: [usize; 3],
) -> SpecStats {
    let want = gen_ids(baseline, req(prompt, max_tokens));
    let (got, stats) = run(req(prompt, max_tokens));
    assert_eq!(collect_ids(&got), want);
    assert_eq!(got.last(), Some(&Event::Done(Reason::Length)));
    assert!(stats.steps > 0);
    let stop_req = || Request {
        stop: vec![want[stop_at]],
        ..req(prompt, max_tokens)
    };
    let (stop_got, _) = run(stop_req());
    assert_eq!(collect_ids(&stop_got), gen_ids(baseline, stop_req()));
    assert_eq!(stop_got.last(), Some(&Event::Done(Reason::Stop)));
    let (ctx_got, _) = run(req(prompt, overlong));
    assert_eq!(
        collect_ids(&ctx_got),
        gen_ids(baseline, req(prompt, overlong))
    );
    assert_eq!(ctx_got.last(), Some(&Event::Done(Reason::Context)));
    stats
}
