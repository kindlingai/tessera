//! Unit tests for ranked speculative plan selection.

use super::decide_plan;
use crate::{Engine, Error, Place, Recipe, Row};
use std::time::Duration;
use tessera_engine_comm::Sockets;
use tessera_graph::{Graph, Parts, Role};
use tessera_lower::{Cache, Cluster, Error as Le, Scoped, Settings, Wants, Weight};
use tessera_models::{mla, mtp};
use tessera_run::Error as Re;
use tessera_testkit::{Noise, tcp_pair, tcp_timeouts, weights};

type P = Result<usize, Error>;
fn socks() -> [Sockets; 2] {
    let (a, b) = tcp_pair().unwrap();
    tcp_timeouts(&a, Duration::from_secs(5)).unwrap();
    tcp_timeouts(&b, Duration::from_secs(5)).unwrap();
    let s0 = Sockets::over(0, vec![None, Some(a)]).unwrap();
    let s1 = Sockets::over(1, vec![Some(b), None]).unwrap();
    [s0, s1]
}
fn vote(r0: (P, P), r1: (P, P)) -> (Result<bool, Error>, Result<bool, Error>) {
    let [mut c0, mut c1] = socks();
    let t = std::thread::spawn(move || decide_plan(Some(&mut c1), r1.0, r1.1));
    (decide_plan(Some(&mut c0), r0.0, r0.1), t.join().unwrap())
}

#[test]
fn ranked_plan_votes() {
    let u = || Err(Error::Lower(Le::Unbound));
    let is_u = |r: Result<bool, Error>| matches!(r, Err(Error::Lower(Le::Unbound)));
    let is_d = |r: Result<bool, Error>| matches!(r, Err(Error::Run(Re::OutputsDisagree)));
    for (r0, r1, w) in [
        ((Ok(10), Ok(15)), (Ok(10), u()), false),
        ((Ok(10), Ok(15)), (u(), Ok(15)), true),
        ((Ok(20), Ok(15)), (Ok(10), u()), false),
        ((Ok(10), Ok(15)), (Ok(20), Ok(15)), true),
    ] {
        let (a, b) = vote(r0, r1);
        assert_eq!((a.unwrap(), b.unwrap()), (w, w));
    }
    let (a, b) = vote((Ok(10), u()), (u(), Ok(15)));
    assert!(is_u(a) && is_u(b));
    let (a, b) = vote((Ok(10), Ok(15)), (u(), u()));
    assert!(is_d(a) && is_u(b));
}

fn open_rank(g: Graph, s: Sockets, c: Sockets) -> Engine {
    let w: Vec<_> = weights(&g, &mut Noise(7))
        .into_iter()
        .map(Weight::Floats)
        .collect();
    let (r, cfg) = (Recipe::default(), Settings::default());
    let p = Place::new(Cluster::two("a", "b"), Parts::even(2));
    let mut e = Engine::open_ranked(g, 16, (&r, &cfg), p, Box::new(s), Box::new(c)).unwrap();
    e.load(&w).unwrap();
    e
}

#[test]
fn ranked_taps_plans() {
    let mut cfg = mla::Config::deepseek_small();
    (cfg.stack.max_seqs, cfg.stack.max_context) = (1, 16);
    cfg.stack.taps = vec![mtp::tap_layer(&cfg)];
    let g = mla::stepper::<mla::Moe>(&cfg);
    let r = |i| Row {
        slot: 0,
        token: i + 1,
        position: i,
    };
    let (rows, [s0, s1], [c0, c1]) = (vec![r(0), r(1), r(2), r(3)], socks(), socks());
    let (g1, r1) = (g.clone(), rows.clone());
    let t = std::thread::spawn(move || {
        let mut e = open_rank(g1, s1, c1);
        let mut sc = Scoped::default();
        (sc.at, sc.barred) = ("tail".into(), vec!["argmax".into()]);
        let mut rec = Recipe::default();
        rec.scoped.push(sc);
        (e.device, e.plans) = (e.device.clone().with(&rec), Cache::default());
        let both = Wants::roles(&e.graph, e.full, &[Role::Tokens, Role::Taps]);
        assert!(e.steps(&r1, &[3], both).is_err());
        let b = e.step_taps_only(&r1).unwrap();
        e.fill_local(&r1).unwrap();
        let out = b.into_host(e.runner.as_ref(), e.taps.unwrap()).unwrap();
        let mut bad = Recipe::default();
        (bad.favor, bad.require_favored) = (Some("x".into()), true);
        (e.device, e.plans) = (e.device.clone().with(&bad), Cache::default());
        (out, e.step_taps_only(&r1).err())
    });
    let mut e0 = open_rank(g, s0, c0);
    let b0 = e0.step_taps_only(&rows).unwrap();
    e0.fill_local(&rows).unwrap();
    let out0 = b0.into_host(e0.runner.as_ref(), e0.taps.unwrap()).unwrap();
    let (err0, (out1, err1)) = (e0.step_taps_only(&rows).err(), t.join().unwrap());
    assert!(out0 == out1 && matches!(err0, Some(Error::Run(Re::OutputsDisagree))));
    assert!(matches!(err1, Some(Error::Lower(Le::FavoredUnused(_)))));
}
