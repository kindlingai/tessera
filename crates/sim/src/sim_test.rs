use crate::{Chip, Const, Limit, Load, Ops, Pool, Tag, gb10, known, m3_pro, strix_halo};

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

fn pool() -> Pool {
    let c = |value| Const::measured(value, "a test value");
    Pool {
        name: "a test pool".into(),
        floor: c(3e-6),
        spawn: c(1e-10),
        step: c(20e-9),
        curve: c(60e-9),
        rate: c(400e9),
        wait: c(300e-9),
        near: c(40e-9),
        cache: c(8e6),
        cached: c(250e9),
        read: c(125e9),
        scan: c(100e9),
        write: c(110e9),
    }
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
}
