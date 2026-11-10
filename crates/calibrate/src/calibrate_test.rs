use crate::probe::{Kind, Probe, Reading, fit, suite};
use crate::{Grid, judge, path};
use lower::{Device, rules};
use ops::testing::gpu_available;
use sim::Chip;

/// A crate's rules register once this one links it.
#[test]
fn linked_rules_register() {
    assert!(rules().iter().any(|r| r.name == "lanes"));
}

#[test]
fn grid_reaches_rules() {
    if !gpu_available() {
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
    };
    Reading { probe, seconds }
}

#[test]
fn probes_fit_pools() {
    let readings: Vec<Reading> = suite().into_iter().map(fake).collect();
    let pool = fit("a made-up chip".into(), &readings);
    let near = |got: f64, want: f64| (got / want - 1.0).abs() < 0.02;
    assert!(near(pool.floor.value, 2e-6) && near(pool.step.value, 10e-9));
    assert!(near(pool.curve.value, 30e-9) && near(pool.rate.value, 1e12));
    assert!(near(pool.wait.value, 300e-9) && near(pool.near.value, 40e-9));
    assert!(near(pool.read.value, 100e9) && near(pool.scan.value, 80e9));
    assert!(near(pool.write.value, 90e9), "{:?}", pool.write);
    assert_eq!(pool.constants().len(), 12);
}

#[test]
fn judge_skips_unmodelled_kernels() {
    if !ops::testing::gpu_available() {
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
}
