use crate::ranks::Ranks;
use tessera_engine::decoder::{Swiglu, stepper};
use tessera_engine::{Engine, Norm, Pick, Recipe, Row, Settings, Weight};
use tessera_models::decoder;
use tessera_testkit::{Noise, tcp_bind, tcp_pair, weights};

/// A faux-weight engine as `rank` of `peers`, or alone.
fn engine(place: Option<(u32, &str)>) -> Engine {
    let mut ranks = Ranks::default();
    if let Some((rank, peers)) = place {
        ranks.flag("--rank", &rank.to_string()).unwrap();
        ranks.flag("--peers", peers).unwrap();
    }
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let g = stepper::<Norm, Swiglu>(&decoder::Config::qwen3_small());
    let stored = weights(&g, &mut Noise(3)).into_iter().map(Weight::Floats);
    let stored: Vec<Weight> = stored.collect();
    let links = ranks.join(&settings).unwrap();
    let at = (&Recipe::default(), &settings);
    let mut engine = ranks.open(links, g, 16, at).unwrap();
    engine.load(&stored).unwrap();
    engine
}

#[test]
fn ranks_match() {
    tcp_pair().unwrap();
    let free = || tcp_bind().unwrap().local_addr().unwrap().to_string();
    let peers = format!("{},{}", free(), free());
    let theirs = peers.clone();
    let follower = std::thread::spawn(move || {
        engine(Some((1, &theirs))).follow().unwrap();
    });
    let mut lead = engine(Some((0, &peers)));
    let mut alone = engine(None);
    let row = |slot, token, position| Row {
        slot,
        token,
        position,
    };
    for rows in [vec![row(0, 5, 0), row(1, 7, 0)], vec![row(0, 4, 1)]] {
        assert_eq!(lead.step(&rows).unwrap(), alone.step(&rows).unwrap());
    }
    assert!(lead.kernels().contains_key("all_reduce"));
    lead.dismiss().unwrap();
    follower.join().unwrap();

    let mut bad = Ranks::default();
    assert!(bad.flag("--peers", "nowhere").is_none());
    assert!(bad.flag("--rank", "1").is_some() && bad.follows());
}
