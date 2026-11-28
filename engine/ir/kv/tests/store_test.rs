use std::path::PathBuf;
use tessera_kv::{Keep, KvConfig, PageStore, RamStore, StoreKind, TieredStore, page_keys};

#[test]
fn direct_load_keys() {
    let toks = [1u32, 2, 3, 4, 5, 6, 7, 8, 9];
    assert_eq!(
        page_keys([11, 29], &toks, 1)[0].0,
        [0x20852c94059f33a7, 0x08cbcf7a9a80d7d4]
    );
    let k3: Vec<[u64; 2]> = page_keys([11, 29], &toks, 3)
        .into_iter()
        .map(|k| k.0)
        .collect();
    assert_eq!(
        k3,
        vec![
            [0xd2b3f162ebe97c67, 0x621df6b2e953a359],
            [0xd13436d96f2d7c32, 0x42ef9e526411e139],
            [0xce85b8bac416e7b8, 0xca6f0307427291c2],
        ]
    );
    let k4: Vec<[u64; 2]> = page_keys([11, 29], &toks, 4)
        .into_iter()
        .map(|k| k.0)
        .collect();
    assert_eq!(
        k4,
        vec![
            [0x1828dadddf0044e1, 0x51565db3b4066eef],
            [0xfb714ee916f5a96f, 0xc5b029aeab7c4fc1],
        ]
    );

    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("tessera-kv-direct-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let ks = page_keys([11, 29], &toks, 3);
    let mut store = PageStore::open_mode(&dir, 32, false).unwrap();
    assert!(store.save(ks[0], &[5u8; 16]).unwrap());
    assert!(store.save(ks[1], &[6u8; 16]).unwrap());
    // Repeated hits trigger LRU compaction while keeping ks[0] hottest.
    let mut dst = [0u8; 16];
    for _ in 0..100 {
        assert!(store.load(ks[0], &mut dst).unwrap());
    }
    assert_eq!(dst, [5u8; 16]);
    assert!(store.save(ks[2], &[7u8; 16]).unwrap());
    assert!(store.has(ks[0]) && !store.has(ks[1]) && store.has(ks[2]));
    // Corrupted oversized or truncated files miss and get dropped.
    let hex = ks[0].hex();
    let path = dir.join(&hex[..2]).join(format!("{hex}.kv"));
    std::fs::write(&path, [9u8; 20]).unwrap();
    assert!(!store.load(ks[0], &mut dst).unwrap());
    assert!(!store.has(ks[0]));
    let hex2 = ks[2].hex();
    let path2 = dir.join(&hex2[..2]).join(format!("{hex2}.kv"));
    std::fs::write(&path2, [9u8; 8]).unwrap();
    assert!(!store.load(ks[2], &mut dst).unwrap());
    assert!(!store.has(ks[2]));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pages_persist_and_evict() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("tessera-kv-store-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let seed = [11u64, 29];
    let p1 = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let p2 = [1u32, 2, 3, 4, 9, 10, 11, 12];
    let k1 = page_keys(seed, &p1, 4);
    let k2 = page_keys(seed, &p2, 4);
    assert_eq!(k1.len(), 2);
    assert_eq!(k1[0], k2[0]);
    assert_ne!(k1[1], k2[1]);
    assert_ne!(page_keys([11, 30], &p1, 4)[0], k1[0]);

    let mut store = PageStore::open(&dir, 32).unwrap();
    assert!(store.save(k1[0], &[1u8; 16]).unwrap());
    // Saving an identical page skips disk writes.
    assert!(!store.save(k1[0], &[1u8; 16]).unwrap());
    assert!(store.save(k1[1], &[2u8; 16]).unwrap());
    assert_eq!(store.stats().saves, 2);
    assert_eq!(store.prefix(&k1), 2);
    assert_eq!(store.prefix(&k2), 1);

    // Touching k1[0] keeps it hot when k2[1] evicts k1[1].
    let mut got = [0u8; 16];
    assert!(store.load(k1[0], &mut got).unwrap());
    assert_eq!(got, [1u8; 16]);
    assert!(store.save(k2[1], &[3u8; 16]).unwrap());
    assert_eq!(store.stats().evictions, 1);
    assert!(store.has(k1[0]));
    assert!(!store.has(k1[1]));
    assert!(store.has(k2[1]));
    drop(store);

    // Reopening recovers existing pages from disk.
    let mut reopened = PageStore::open(&dir, 32).unwrap();
    assert_eq!(reopened.prefix(&k2), 2);
    let mut tail = [0u8; 16];
    assert!(reopened.load(k2[1], &mut tail).unwrap());
    assert_eq!(tail, [3u8; 16]);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn parses_kv_specs() {
    let inline = "cache:spare,kind=spare_ram,floor=2G;ram:,size=16G,keep=all;ssd:/tmp/kv,size=4G,persist=on,keep=system,cache=spare;remote:auto,size=8G,group=g1,cap=16G,weight=1000";
    let cfg = KvConfig::parse(inline).unwrap();
    assert_eq!(cfg.tiers.len(), 3);
    assert_eq!(cfg.tiers[0].store, StoreKind::Ram { bytes: 16 << 30 });
    assert_eq!(cfg.tiers[1].spare_ram, Some(2 << 30));
    assert_eq!(cfg.tier_keep(&cfg.tiers[1]), Keep::System);
    assert!(matches!(cfg.tiers[2].store, StoreKind::Remote { .. }));
    assert!(KvConfig::parse("off").unwrap().disabled);
    assert!(KvConfig::parse("ssd:/tmp/x").is_err());
    assert!(KvConfig::parse("ram:,size=1G,persist=on").is_err());
    assert!(KvConfig::parse("remote:h:7,size=8G,group=g,cap=4G").is_err());

    let y = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("tessera-kv-spec-{}.yaml", std::process::id()));
    std::fs::write(
        &y,
        "caches:\n  sp: {kind: spare_ram, floor: 1G}\ntiers:\n  - keep: all\n    store:\n      ram: {size_per_rank: 2G}\n  - keep: sessions\n    cache: sp\n    store:\n      ssd: {path: /tmp/kv2, size_per_rank: 4G, persist: true}\n",
    )
    .unwrap();
    let ycfg = KvConfig::parse(&format!("@{}", y.display())).unwrap();
    assert_eq!(ycfg.tiers.len(), 2);
    assert_eq!(ycfg.tiers[0].store, StoreKind::Ram { bytes: 2 << 30 });
    assert_eq!(ycfg.tiers[1].spare_ram, Some(1 << 30));
    let _ = std::fs::remove_file(&y);
}

#[test]
fn tiered_ram_ssd_cascade() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("tessera-kv-tier-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let mut ram = RamStore::new(32);
    let keys = page_keys([3, 7], &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 4);
    assert!(ram.save(keys[0], &[1u8; 16]).is_empty());
    assert!(ram.save(keys[1], &[2u8; 16]).is_empty());
    let ev = ram.save(keys[2], &[3u8; 16]);
    assert_eq!(ev.len(), 1);
    assert_eq!(ev[0].0, keys[0]);

    let cfg = KvConfig {
        tiers: vec![
            tessera_kv::TierSpec {
                name: "ram".into(),
                store: StoreKind::Ram { bytes: 16 },
                keep: Some(Keep::All),
                spare_ram: None,
            },
            tessera_kv::TierSpec {
                name: "ssd".into(),
                store: StoreKind::Ssd {
                    path: dir.clone(),
                    bytes: 64,
                    persist: false,
                },
                keep: Some(Keep::All),
                spare_ram: None,
            },
        ],
        ..KvConfig::default()
    };
    let mut tiered = TieredStore::open(&cfg, 0, 1).unwrap();
    assert!(tiered.save(keys[0], &[10u8; 16]).unwrap());
    assert!(tiered.save(keys[1], &[20u8; 16]).unwrap());
    // keys[0] evicted from RAM and cascaded to SSD.
    let mut got = [0u8; 16];
    assert!(tiered.load(keys[0], &mut got).unwrap());
    assert_eq!(got, [10u8; 16]);
    // persist=false clears SSD on reopen.
    drop(tiered);
    let reopened = TieredStore::open(&cfg, 0, 1).unwrap();
    assert!(!reopened.has(keys[0]));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn spare_ram_caches() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("tessera-kv-spare-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let keys = page_keys([5, 9], &[1, 2, 3, 4], 4);
    let payload = [7u8; 64];
    let make_cfg = |floor: u64| KvConfig {
        tiers: vec![tessera_kv::TierSpec {
            name: "ssd".into(),
            store: StoreKind::Ssd {
                path: dir.clone(),
                bytes: 256,
                persist: false,
            },
            keep: Some(Keep::All),
            spare_ram: Some(floor),
        }],
        ..KvConfig::default()
    };

    // A 1-byte floor caches pages in spare RAM.
    let mut cached = TieredStore::open(&make_cfg(1), 0, 1).unwrap();
    assert!(cached.save(keys[0], &payload).unwrap());
    let mut got = [0u8; 64];
    assert!(cached.load(keys[0], &mut got).unwrap());
    assert_eq!(got, payload);
    assert_eq!(cached.stats().hits, 0);
    drop(cached);

    // An unreachable floor bypasses spare RAM for disk.
    let mut uncached = TieredStore::open(&make_cfg(u64::MAX), 0, 1).unwrap();
    assert!(uncached.save(keys[0], &payload).unwrap());
    let mut got2 = [0u8; 64];
    assert!(uncached.load(keys[0], &mut got2).unwrap());
    assert_eq!(got2, payload);
    assert_eq!(uncached.stats().hits, 1);
    std::fs::remove_dir_all(&dir).unwrap();
}
