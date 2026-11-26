use std::path::PathBuf;
use tessera_kv::{Keep, KvConfig, PageStore, RamStore, StoreKind, TieredStore, page_keys};

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
