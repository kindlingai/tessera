use crate::ranks::Ranks;
use std::net::TcpListener;
use tessera_engine::decoder::{Swiglu, stepper};
use tessera_engine::{Asks, Engine, Norm, Pick, Recipe, Row, Settings, Weight};
use tessera_models::decoder;
use tessera_testkit::{Noise, tcp_bind, tcp_pair, weights};

/// A faux-weight engine as `rank` of `peers`, or alone.
fn engine(place: Option<(u32, &str, TcpListener)>, asks: Option<Asks>) -> Engine {
    let mut ranks = Ranks::default();
    if let Some((rank, peers, listener)) = place {
        ranks.flag("--rank", &rank.to_string()).unwrap();
        ranks.flag("--peers", peers).unwrap();
        ranks.listen(listener);
    }
    let settings = Settings {
        backend: Pick::Host,
        asks,
        ..Settings::default()
    };
    let g = stepper::<Norm, Swiglu>(&decoder::Config::qwen3_small());
    let stored = weights(&g, &mut Noise(3)).into_iter().map(Weight::Floats);
    let stored: Vec<Weight> = stored.collect();
    let links = ranks.join(&settings, &Default::default()).unwrap();
    let at = (&Recipe::default(), &settings);
    let mut engine = ranks.open(links, g, 16, at).unwrap();
    engine.load(&stored).unwrap();
    engine
}

#[test]
fn ranks_match() {
    tcp_pair().unwrap();
    let [mine, theirs] = [(); 2].map(|()| tcp_bind().unwrap());
    let addr = |l: &TcpListener| l.local_addr().unwrap().to_string();
    let peers = format!("{},{}", addr(&mine), addr(&theirs));
    let their_peers = peers.clone();
    let follower = std::thread::spawn(move || {
        engine(Some((1, &their_peers, theirs)), None)
            .follow()
            .unwrap();
    });
    let asks = Asks::default();
    let mut lead = engine(Some((0, &peers, mine)), Some(asks.clone()));
    let mut alone = engine(None, None);
    let row = |slot, token, position| Row {
        slot,
        token,
        position,
    };
    for rows in [vec![row(0, 5, 0), row(1, 7, 0)], vec![row(0, 4, 1)]] {
        assert_eq!(lead.step(&rows).unwrap(), alone.step(&rows).unwrap());
    }
    assert!(lead.kernels().contains_key("all_reduce"));
    // An ask times both ranks until it drops.
    let ask = asks.hold();
    lead.settle().unwrap();
    lead.step(&[row(0, 3, 2)]).unwrap();
    drop(ask);
    lead.settle().unwrap();
    let ranks = lead.ranks().unwrap();
    assert_eq!(ranks.launches, 1);
    assert!(ranks.profiles.iter().all(Option::is_some));
    lead.dismiss().unwrap();
    follower.join().unwrap();

    let mut bad = Ranks::default();
    assert!(bad.flag("--peers", "nowhere").is_none());
    assert!(bad.flag("--rank", "1").is_some() && bad.follows());
}

#[test]
fn sizes_ranks_offline() {
    let guard = tessera_engine::sizer::sizing();
    let mut ranks = Ranks::default();
    ranks.flag("--rank", "1").unwrap();
    ranks.flag("--peers", "127.0.0.1:1,127.0.0.2:1").unwrap();
    ranks.flag("--sums", "arx").unwrap();
    assert!(!ranks.follows());
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let links = ranks.join(&settings, &Default::default()).unwrap();
    let g = stepper::<Norm, Swiglu>(&decoder::Config::qwen3_small());
    let at = (&Recipe::default(), &settings);
    let _engine = ranks.open(links, g, 16, at).unwrap();
    assert!(guard.sized().weights > 0);
}
