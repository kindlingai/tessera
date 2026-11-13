use crate::probe::{Kind, Probe, Reading, fit, links, read, suite};
use crate::{Grid, headroom, judge, path};
use gpukit::testing::backend_ready;
use lower::{Device, rules};
use sim::Chip;

/// A crate's rules register once this one links it.
#[test]
fn linked_rules_register() {
    assert!(rules().iter().any(|r| r.name == "lanes"));
}

#[test]
fn grid_reaches_rules() {
    if !backend_ready() {
        return eprintln!("no GPU here; skipped");
    }
    let mut device = Device::guess();
    let timed = Grid::tiny().run(&mut device).unwrap();
    assert!(device.samples() > 0 && !device.name.is_empty());
    for rule in rules() {
        let ran = timed.iter().any(|(step, _)| step.rule == rule.name);
        assert!(ran, "the grid never ran {}", rule.name);
    }
    // Everything is known now, so a second pass times nothing.
    assert!(Grid::tiny().run(&mut device).unwrap().is_empty());
    let file = std::env::temp_dir().join("tessera-calibrate-test.json");
    device.save(&file).unwrap();
    assert!(Device::load(&file).unwrap() == device, "a round trip");
    std::fs::remove_file(&file).unwrap();
    assert!(path(&device.name).ends_with(format!("{}.json", device.name.replace(' ', "-"))));
}

/// A made-up chip's reading for one probe.
fn fake(probe: Probe) -> Reading {
    let (threads, trips) = (probe.threads as f64, probe.trips as f64);
    let floor = 2e-6;
    let seconds = match probe.kind {
        // 10 ns a trip, or 1e12 trips a second.
        Kind::Spin => floor + (trips * 10e-9).max(threads * trips / 1e12),
        Kind::Curve => floor + trips * 30e-9,
        // 300 ns a wait, or 100 GB/s.
        Kind::Read { depth } => (trips * 300e-9).max(4.0 * threads * trips * depth as f64 / 100e9),
        Kind::Scan => (trips * 50e-9).max(4.0 * threads * trips / 80e9),
        Kind::Write => 4.0 * threads * trips / 90e9,
        Kind::Chase => trips * 400e-9,
        // 1e13 tile flops a second.
        Kind::Tile => threads / 32.0 * trips * 65_536.0 / 1e13,
    };
    Reading { probe, seconds }
}

#[test]
fn probes_fit_pools() {
    let readings: Vec<Reading> = suite().into_iter().map(fake).collect();
    let fit = fit(&readings);
    let near = |got: f64, want: f64| (got / want - 1.0).abs() < 0.02;
    assert!(near(fit.floor, 2e-6) && near(fit.step, 10e-9));
    assert!(near(fit.curve, 30e-9) && near(fit.rate, 1e12));
    assert!(near(fit.wait, 300e-9) && near(fit.near, 40e-9));
    assert!(near(fit.read, 100e9) && near(fit.scan, 80e9));
    assert!(near(fit.write, 90e9) && near(fit.tile, 1e13), "{fit:?}");
    assert_eq!(fit.pool("a made-up chip").constants().len(), 13);
}

#[test]
fn judge_skips_unmodelled_kernels() {
    if !backend_ready() {
        return eprintln!("no GPU here; skipped");
    }
    let mut device = Device::guess();
    let timed = Grid::tiny().run(&mut device).unwrap();
    let verdicts = judge(&sim::m3_pro(), &timed);
    let names: Vec<&str> = verdicts.iter().map(|v| v.kernel).collect();
    assert!(names.contains(&"lanes") && names.contains(&"map"));
    // Attention states its load later, so judging skips it.
    assert!(!names.contains(&"attend"));
    for v in verdicts {
        let [low, median, high] = v.ratio;
        assert!(0.0 < low && low <= median && median <= high, "{v:?}");
    }
    // A kernel has a bound when it states a load.
    for h in headroom(&sim::m3_pro(), &timed) {
        assert_eq!(h.bound.is_some(), names.contains(&h.kernel), "{h:?}");
    }
}

/// A float probe, an index probe and the links run.
#[test]
fn probes_and_links_run() {
    if !backend_ready() {
        return eprintln!("no GPU here; skipped");
    }
    let probe = |kind, threads, trips| Probe {
        kind,
        threads,
        trips,
    };
    let readings = read(&[probe(Kind::Spin, 256, 16), probe(Kind::Chase, 1, 4)]).unwrap();
    assert!(readings.iter().all(|r| r.seconds > 0.0));
    let cluster = links().unwrap();
    assert_eq!(cluster.links.len(), 2);
    assert!(cluster.links.iter().all(|l| l.rate > 0.0));
}
