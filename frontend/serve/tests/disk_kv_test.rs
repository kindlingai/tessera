//! Integration test for `--kv-store` host-RAM and SSD KV caching.

use std::time::Duration;
use tessera_engine::comm::{Sockets, Transport};
use tessera_engine::decoder::{Swiglu, stepper};
use tessera_engine::kv::{Keep, KvConfig, StoreKind, TierSpec};
use tessera_engine::{
    Batch, Engine, Event, Fair, Norm, Pick, Place, Recipe, Request, Settings, Weight,
};
use tessera_models::decoder;
use tessera_serve::start::Options;
use tessera_testkit::{Noise, tcp_pair, tcp_timeouts, weights};

fn open_disk_eng(links: Option<[Sockets; 2]>, recipe: &Recipe) -> Engine {
    let settings = Settings {
        backend: Pick::Host,
        ..Settings::default()
    };
    let g = stepper::<Norm, Swiglu>(&decoder::Config::toy(2, 128));
    let stored: Vec<Weight> = weights(&g, &mut Noise(3))
        .into_iter()
        .map(Weight::Floats)
        .collect();
    let mut eng = match links {
        Some([sums, ctrl]) => {
            let place = Place::mesh(2, Vec::new(), false);
            let c: Box<dyn Transport> = Box::new(ctrl);
            Engine::open_ranked(g, 128, (recipe, &settings), place, sums, c).unwrap()
        }
        None => Engine::open(g, 128, recipe, &settings).unwrap(),
    };
    eng.load(&stored).unwrap();
    eng
}

#[test]
fn disk_kv_restores() {
    let mut opts = Options::default();
    assert!(opts.flag("--kv-store", "ram:,size=4G".into()).is_some());
    assert!(
        opts.flag("--kv-store", "ssd:/tmp/kv,size=8G,persist=on".into())
            .is_some()
    );
    assert!(opts.flag("--kv-keep", "sessions".into()).is_some());
    assert!(opts.flag("--kv-system-after", "2".into()).is_some());
    assert!(opts.flag("--kv-store", "ssd:/tmp/kv".into()).is_none());
    assert!(opts.flag("--kv-keep", "bogus".into()).is_none());
    assert!(opts.flag("--kv-cache-dir", "/tmp/kv".into()).is_none());

    let pid = std::process::id();
    let (d, d_tp) = (
        tessera_testkit::scratch(&format!("{pid}-dk")),
        tessera_testkit::scratch(&format!("{pid}-dktp")),
    );
    let recipe = Recipe {
        kept: Engine::paged_layout(),
        ..Recipe::default()
    };
    let p1: Vec<u32> = (0..70).map(|i| (i * 5 + 1) % 32).collect();
    let p2: Vec<u32> = p1.iter().copied().chain([4, 9, 13]).collect();
    let ids = |e: &[Event]| -> Vec<u32> {
        e.iter()
            .filter_map(|x| match x {
                Event::Token(id) => Some(*id),
                _ => None,
            })
            .collect()
    };
    let mut want = Vec::new();
    let req = Request {
        tokens: p2.clone(),
        max_tokens: 4,
        ..Request::default()
    };
    open_disk_eng(None, &recipe)
        .generate(req, &mut |e| want.push(e))
        .unwrap();
    let run_one = |eng: &mut Engine, tokens: Vec<u32>| -> Vec<Event> {
        let mut batch = Batch::new(eng, Box::new(Fair));
        batch.reuse();
        let (tx, rx) = std::sync::mpsc::channel();
        let req = Request {
            tokens,
            max_tokens: 4,
            ..Request::default()
        };
        batch.submit(req, Box::new(move |e| tx.send(e).is_ok()));
        while batch.step().unwrap() {}
        rx.try_iter().collect()
    };
    let ssd_cfg = |dir: &std::path::Path| KvConfig {
        tiers: vec![TierSpec {
            name: "ssd".into(),
            store: StoreKind::Ssd {
                path: dir.to_path_buf(),
                bytes: 1 << 20,
                persist: true,
            },
            keep: Some(Keep::All),
            spare_ram: None,
        }],
        ..KvConfig::default()
    };
    {
        let mut eng = open_disk_eng(None, &recipe);
        eng.set_kv_store(&ssd_cfg(&d)).unwrap();
        let _ = run_one(&mut eng, p1.clone());
        assert_eq!(eng.kv_store_stats().unwrap().saves, 1);
    }
    let mut eng = open_disk_eng(None, &recipe);
    eng.set_kv_store(&ssd_cfg(&d)).unwrap();
    let evs = run_one(&mut eng, p2.clone());
    assert_eq!(evs.first(), Some(&Event::Kept(64)));
    assert_eq!(ids(&evs), ids(&want));
    let st = eng.kv_store_stats().unwrap();
    assert_eq!((st.hits, st.saves), (1, 0));

    // Host-RAM tier restores after VRAM eviction without touching disk.
    let mut ram_recipe = recipe.clone();
    ram_recipe.share_kept(128);
    let mut ram_eng = open_disk_eng(None, &ram_recipe);
    ram_eng
        .set_kv_store(&KvConfig::default_ram(1 << 20))
        .unwrap();
    let _ = run_one(&mut ram_eng, p1.clone());
    assert_eq!(ram_eng.kv_store_stats().unwrap().saves, 1);
    let filler: Vec<u32> = (0..70).map(|i| (i * 7 + 3) % 32).collect();
    let _ = run_one(&mut ram_eng, filler);
    let ram_evs = run_one(&mut ram_eng, p2.clone());
    assert_eq!(ram_evs.first(), Some(&Event::Kept(64)));
    assert_eq!(ids(&ram_evs), ids(&want));
    assert_eq!(ram_eng.kv_store_stats().unwrap().hits, 1);

    let make_links = || -> [Sockets; 2] {
        let (a, b) = tcp_pair().unwrap();
        for s in [&a, &b] {
            tcp_timeouts(s, Duration::from_secs(20)).unwrap();
        }
        [
            Sockets::over(0, vec![None, Some(a)]).unwrap(),
            Sockets::over(1, vec![Some(b), None]).unwrap(),
        ]
    };
    let run_tp = |tokens: Vec<u32>| -> Vec<Event> {
        let ([sums, others], [control, told]) = (make_links(), make_links());
        let (r1, fd) = (recipe.clone(), d_tp.clone());
        let f = std::thread::spawn(move || {
            let mut e = open_disk_eng(Some([others, told]), &r1);
            e.set_kv_store(&ssd_cfg(&fd)).unwrap();
            e.follow().unwrap();
        });
        let mut lead = open_disk_eng(Some([sums, control]), &recipe);
        lead.set_kv_store(&ssd_cfg(&d_tp)).unwrap();
        let evs = run_one(&mut lead, tokens);
        lead.dismiss().unwrap();
        f.join().unwrap();
        evs
    };
    let _ = run_tp(p1);
    let evs = run_tp(p2);
    assert_eq!(evs.first(), Some(&Event::Kept(64)));
    assert_eq!(ids(&evs), ids(&want));
    for dir in [d, d_tp] {
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
