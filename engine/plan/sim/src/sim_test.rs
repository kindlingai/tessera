use crate::known::{generic, known, m3_pro};
use crate::nvidia::gb10;
use crate::{Chip, Fit, Limit, Pool, Tag};
use tessera_work::{Load, Ops};

/// The Radeon 8060S of a Ryzen AI Max, through ROCm.
fn strix_halo() -> Pool {
    let fit = Fit {
        floor: 7.4e-6,
        spawn: 6e-12,
        step: 1.0e-8,
        curve: 2.4e-8,
        rate: 2.0e12,
        wait: 3.35e-7,
        near: 4.3e-8,
        cache: 4e6,
        cached: 3.24e11,
        read: 2.23e11,
        scan: 1.19e11,
        write: 2.04e11,
        tile: 1.5e13,
        ..Fit::default()
    };
    fit.pool("rocm Radeon 8060S")
}

/// One-instruction e4m3 costs a chip fewer ops.
#[test]
fn priced_per_chip() {
    let trip = Ops {
        f16: 1.0,
        e4m3: 1.0,
        e2m1: 1.0,
        ..Ops::default()
    };
    let (native, plain) = (trip.priced(true), trip.priced(false));
    assert_eq!((native.f16, native.e4m3, native.e2m1), (0.0, 0.0, 0.0));
    assert!(native.int < plain.int && native.float < plain.float);
    assert_eq!((plain.int, plain.float), (13.0, 5.0));
}

/// A stream of `bytes`, one float per thread per trip.
fn stream(bytes: f64, threads: usize) -> Load {
    let trip = Ops {
        loads: 1.0,
        fresh: 1.0,
        ..Ops::default()
    };
    Load {
        threads,
        group: 256,
        trips: bytes / 4.0 / threads as f64,
        trip,
        reads: vec![bytes],
        flight: 4.0,
        ..Load::default()
    }
}

#[test]
fn gb10_streams_and_stalls() {
    let chip = gb10();
    // Every resident warp reading: the bus is the limit.
    let wide = chip.estimate(&stream(512e6, 48 * 1536));
    assert_eq!(wide.limit, Limit::Read);
    assert!((wide.seconds - 512e6 / 255e9).abs() < 1e-4, "{wide:?}");
    // One group per multiprocessor keeps too little in flight.
    let thin = chip.estimate(&stream(512e6, 48 * 32));
    let flow = 48.0 * 32.0 * 4.0 / 433e-9;
    assert!((thin.seconds - 512e6 / flow).abs() < 1e-4, "{thin:?}");
    assert!(thin.seconds > 4.0 * wide.seconds);
    // A buffer that fits the cache never waits on memory.
    assert_eq!(chip.estimate(&stream(8e6, 48 * 1536)).limit, Limit::Cache);
    let constants = chip.constants();
    let measured = constants.iter().filter(|(_, c)| c.tag == Tag::Measured);
    assert!(measured.count() > constants.len() / 2);
}

fn fit() -> Fit {
    Fit {
        floor: 3e-6,
        spawn: 1e-10,
        step: 20e-9,
        curve: 60e-9,
        rate: 400e9,
        wait: 300e-9,
        near: 40e-9,
        cache: 8e6,
        cached: 250e9,
        read: 125e9,
        scan: 100e9,
        write: 110e9,
        ..Fit::default()
    }
}

fn pool() -> Pool {
    fit().pool("a test pool")
}

#[test]
fn pool_limits_follow() {
    let chip = pool();
    let float = Ops {
        float: 1.0,
        ..Ops::default()
    };
    let spin = |threads| Load {
        threads,
        group: 256,
        trips: 4096.0,
        trip: float,
        ..Load::default()
    };
    // A few threads wait on their own chain of ops.
    let few = chip.estimate(&spin(256));
    assert_eq!(few.limit, Limit::Chain);
    assert!((few.seconds - (3e-6 + 256.0 * 1e-10 + 4096.0 * 20e-9)).abs() < 1e-9);
    // A million threads wait on the whole chip.
    let many = chip.estimate(&spin(1 << 20));
    assert_eq!(many.limit, Limit::Compute);
    assert_eq!(chip.estimate(&stream(512e6, 1 << 20)).limit, Limit::Read);
    // Few threads wait on each load: 500,000 waits apiece.
    let starved = chip.estimate(&stream(512e6, 256)).seconds;
    assert!((starved - 500e3 * 300e-9).abs() < 1e-3, "{starved}");
    let empty = chip.estimate(&Load {
        threads: 1,
        ..Load::default()
    });
    assert_eq!(empty.limit, Limit::Start);
}

#[test]
fn known_chips_stream() {
    for (chip, rate) in [(m3_pro(), 1.28e11), (strix_halo(), 2.23e11)] {
        let wide = chip.estimate(&stream(512e6, 1 << 16));
        assert_eq!(wide.limit, Limit::Read, "{}", chip.name);
        assert!((wide.seconds - 512e6 / rate).abs() < 2e-4, "{wide:?}");
        // One thread along its own stretch waits on every line.
        let mut alone = stream(512e6, 1);
        (alone.trip.fresh, alone.flight) = (1.0 / 16.0, 64.0);
        assert_eq!(chip.estimate(&alone).limit, Limit::Chain);
    }
    assert_eq!(known("cuda NVIDIA GB10").unwrap().name(), gb10().name);
    assert_eq!(known("metal Apple M3 Pro").unwrap().name(), m3_pro().name);
    assert!(known("rocm HIP device 0").is_none());
    let gpu = generic("cuda", "cuda Test", 188, 1.79e12, 48 * 1024).unwrap();
    assert_eq!(gpu.name(), "cuda Test");
    let wide = gpu.estimate(&stream(512e6, 188 * 1536));
    assert_eq!(wide.limit, Limit::Read);
    assert!(wide.seconds < gb10().estimate(&stream(512e6, 48 * 1536)).seconds);
    assert!(generic("rocm", "rocm Test", 40, 2.5e11, 64 * 1024).is_some());
    assert!(generic("cuda", "cuda Zero", 0, 1e12, 48 * 1024).is_none());
    assert!(generic("cuda", "cuda Zero", 48, 0.0, 48 * 1024).is_none());
    assert!(generic("cpu", "cpu", 8, 1e11, 0).is_none());
}

#[test]
fn clusters_pick_links() {
    use crate::{Cluster, Link, Zone, ZoneKind};
    let mut c = Cluster::default();
    assert_eq!((c.zones.len(), c.kind(0)), (1, ZoneKind::Gpu));
    let host = c.add(Zone {
        name: "host".into(),
        kind: ZoneKind::Cpu,
        parent: None,
    });
    let link = |latency, rate| Link {
        from: 0,
        to: host,
        latency,
        rate,
        tag: Tag::Assumed,
    };
    c.links.push(link(1e-5, 1e10));
    c.links.push(link(1e-3, 1e12));
    assert_eq!(c.transfer(0, 0, 1e9), Some(0.0));
    // Small moves take the quick link, large ones the wide.
    assert_eq!(c.transfer(0, host, 1e4), Some(1e-5 + 1e-6));
    assert_eq!(c.transfer(0, host, 1e9), Some(1e-3 + 1e-3));
    assert_eq!(c.transfer(host, 0, 1.0), None, "links are one way");
}

#[test]
fn ring_rounds() {
    use crate::{Cluster, Link, Zone, ZoneKind};
    let mut c = Cluster::default();
    let gpu = |name: &str| Zone {
        name: name.into(),
        kind: ZoneKind::Gpu,
        parent: None,
    };
    let (b, d) = (c.add(gpu("b")), c.add(gpu("d")));
    // One slow link of 10 us. Two of 1 us.
    for (from, to, latency) in [(0, b, 1e-6), (b, d, 1e-5), (d, 0, 1e-6)] {
        c.links.push(Link {
            from,
            to,
            latency,
            rate: 1e9,
            tag: Tag::Assumed,
        });
    }
    assert_eq!(c.all_reduce(&[0], 3e6), Some(0.0));
    // Four rounds of a third each, on the slow link.
    let ring = c.all_reduce(&[0, b, d], 3e6).unwrap();
    assert!((ring - 4.0 * (1e-5 + 1e-3)).abs() < 1e-9, "{ring}");
    assert_eq!(c.all_reduce(&[0, d, b], 3e6), None);
}

#[test]
fn residency_sets_waves() {
    let fit = Fit {
        tile: 1e15,
        cores: 1.0,
        store: 64e3,
        resident: 1024.0,
        held: 128.0,
        mma: 1e-9,
        spill: 1e-6,
        ..fit()
    };
    let chip = fit.pool("a resident pool");
    let tiled = |group_bytes, held| Load {
        threads: 1024,
        group: 128,
        trips: 100.0,
        trip: Ops {
            tile_flops: 500.0,
            held,
            ..Ops::default()
        },
        group_bytes,
        ..Load::default()
    };
    let wave = 100.0 * 500.0 * 1e-9;
    let start = 3e-6 + 1024.0 * 1e-10;
    let seconds = |bytes, held| chip.estimate(&tiled(bytes, held)).seconds - start;
    // Eight groups fit the store at once, then four.
    assert!((seconds(8e3, 0.0) - wave).abs() < 1e-12);
    assert!((seconds(16e3, 0.0) - 2.0 * wave).abs() < 1e-12);
    // Registers share the store, and past the cap they spill.
    assert!((seconds(8e3, 64.0) - 8.0 * wave).abs() < 1e-12);
    let spilled = 8.0 * 100.0 * (200.0 - 56e3 / 512.0) * 1e-6;
    assert!((seconds(8e3, 200.0) - 8.0 * wave - spilled).abs() < 1e-9);
}
