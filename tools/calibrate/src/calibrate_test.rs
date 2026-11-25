use crate::bench::{Mark, slower, text};
use crate::probe::{Kind, Probe, Reading, fit, links, read, suite};
use crate::{Grid, headroom, judge};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use std::collections::{BTreeMap, BTreeSet};
use tessera_engine::trellis;
use tessera_graph::DimName::Rows;
use tessera_graph::{Composite, NodeId, Op, verify};
use tessera_lower::form::BF16_FORM;
use tessera_lower::{
    Backend, Cluster, Device, Pick, Pins, Settings, Source, Zone, ZoneKind, cut_region, lower,
    rules,
};
use tessera_sim::Chip;
use tessera_testkit::graphs::picked;
use tessera_testkit::scratch;

/// This build's GPU when ready, else the host.
fn settings() -> Settings {
    tessera_run::testing::runs().pop().unwrap()
}

/// A crate's rules register once this one links it.
#[test]
fn linked_rules_register() {
    assert!(rules().iter().any(|r| r.name == "lanes"));
}

#[test]
fn grid_reaches_rules() {
    let settings = settings();
    let mut device = Device::guess(&settings);
    let timed = Grid::tiny().run(&mut device, &settings).unwrap();
    assert!(device.samples() > 0 && !device.name.is_empty());
    // CPU and compiled device rules need matching hardware.
    let device_rules = [
        "trellis_mma",
        "fast_gemv",
        "fast_mm",
        "fast_stacked",
        "fast_mla_attend",
        "picked_trellis",
    ];
    // The grid's trellis experts reach picked trellis on CUDA.
    let cuda = device.platform.backend == "cuda";
    let device_only = |name| device_rules.contains(&name) && !(cuda && name == "picked_trellis");
    let skipped = |r: &&&tessera_lower::Rule| r.name.starts_with("cpu_") || device_only(r.name);
    for rule in rules().iter().filter(|r| !skipped(r)) {
        let ran = timed.iter().any(|(step, _)| step.rule == rule.name);
        assert!(ran, "the grid never ran {}", rule.name);
    }
    // Everything is known now, so a second pass times nothing.
    assert!(Grid::tiny().run(&mut device, &settings).unwrap().is_empty());
    // A later run on this platform reads what this stored.
    let costs = Settings {
        costs: Some(scratch("tessera-calibrate-costs")),
        ..settings
    };
    let file = Device::path(&costs, &device.platform).unwrap();
    device.save(&file).unwrap();
    let stored = Device::stored(device.platform.clone(), &costs).unwrap();
    assert!(stored == device, "a round trip");
    std::fs::remove_file(&file).unwrap();
}

/// A guessed CUDA device plans trellis experts like the engine.
#[test]
fn guess_picks_trellis() {
    let settings = settings();
    if settings.gpu() != Some(Backend::Cuda) {
        return;
    }
    let pins = Pins::default().pin("set", &trellis::TRELLIS_K3_PMCGX);
    let device = Device::guess(&settings).pinned(pins);
    assert!(
        device.platform.bf16_tile().is_some(),
        "{:?}",
        device.platform
    );
    let g = picked(&crate::grid::bounded(&[Rows]), [4, 128, 128]);
    let plan = lower(&g, 0, &[(Rows, 1)], &device).unwrap();
    assert!(plan.steps.iter().any(|s| s.rule == "picked_trellis"));
}

/// On a CPU zone, packed and plain matmuls are timed.
#[test]
fn cpu_matmuls_compete() {
    let cpu = Settings {
        backend: Pick::Cpu,
        ..tessera_run::testing::host()
    };
    let grid = Grid {
        rows: vec![1, 8],
        widths: vec![32],
        spans: vec![[4, 4]],
        ..Grid::tiny()
    };
    let mut host = Device::guess(&cpu);
    let timed = grid.run(&mut host, &cpu).unwrap();
    for rule in ["cpu_dense", "cpu_tiled"] {
        assert!(timed.iter().any(|(s, _)| s.rule == rule), "{rule} untimed");
    }
    // A GPU's host zone reads the host's samples.
    let mut cluster = Cluster::one("gpu");
    let kind = ZoneKind::Cpu;
    let parent = None;
    cluster.add(Zone {
        name: "host".into(),
        kind,
        parent,
    });
    let gpu = Device::guess(&tessera_run::testing::host())
        .zoned(cluster)
        .hosted(&host);
    let (step, _) = &timed[0];
    let (k, len) = (&*step.kernel, step.write.len);
    assert_eq!(gpu.seconds_in(1, k, len, false), host.seconds(k, len));
}

/// A saved bench reads back, and names what got slower.
#[test]
fn bench_baselines() {
    let mark = |name: &str, seconds| Mark {
        name: name.into(),
        seconds,
        flops: 1.0,
    };
    let base = text(&[mark("a", 1.0), mark("b", 1.0)]);
    let now = [mark("a", 1.5), mark("b", 1.1)];
    let got = slower(&now, &base, 0.2);
    assert_eq!(got.len(), 1);
    assert!(got[0].starts_with("a: "), "{got:?}");
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
        Kind::Tile => threads / 32.0 * trips * 16_384.0 / 1e13,
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
    assert_eq!(fit.pool("a made-up chip").constants().len(), 21);
}

#[test]
fn judge_skips_unmodelled_kernels() {
    let settings = settings();
    let mut device = Device::guess(&settings);
    let timed = Grid::tiny().run(&mut device, &settings).unwrap();
    let verdicts = judge(&tessera_sim::m3_pro(), &timed, &settings);
    let names: Vec<&str> = verdicts.iter().map(|v| v.kernel).collect();
    assert!(names.contains(&"lanes") && names.contains(&"map"));
    // Attention states its load later, so judging skips it.
    assert!(!names.contains(&"attend"));
    for v in verdicts {
        let [low, median, high] = v.ratio;
        assert!(0.0 < low && low <= median && median <= high, "{v:?}");
    }
    // A kernel has a bound when it states a load.
    for h in headroom(&tessera_sim::m3_pro(), &timed) {
        assert_eq!(h.bound.is_some(), names.contains(&h.kernel), "{h:?}");
    }
}

/// A float probe, an index probe and the links run.
#[test]
fn probes_and_links_run() {
    let settings = settings();
    let probe = |kind, threads, trips| Probe {
        kind,
        threads,
        trips,
    };
    let probes = [probe(Kind::Spin, 256, 16), probe(Kind::Chase, 1, 4)];
    let readings = read(&probes, &settings).unwrap();
    assert!(readings.iter().all(|r| r.seconds > 0.0));
    let cluster = links(&settings).unwrap();
    assert_eq!(cluster.links.len(), 2);
    assert!(cluster.links.iter().all(|l| l.rate > 0.0));
}

/// Validates WGSL with naga.
fn validated(text: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(text).map_err(|e| e.emit_to_string(text))?;
    let (flags, caps) = (ValidationFlags::all(), Capabilities::all());
    let mut validator = Validator::new(flags, caps);
    validator.validate(&module).map_err(|e| format!("{e:?}"))?;
    Ok(())
}

/// Grid kernels print as WGSL naga accepts, or refuse.
#[test]
fn grid_prints_wgsl() {
    let settings = tessera_run::testing::host();
    let mut plans = Vec::new();
    for pins in [Pins::default(), Pins::all(BF16_FORM)] {
        let guess = || Device::guess(&settings).pinned(pins.clone());
        plans.extend(Grid::tiny().plans(&guess).unwrap());
    }
    let mut seen = BTreeSet::new();
    let mut refused: BTreeMap<String, usize> = BTreeMap::new();
    for plan in plans {
        for step in &plan.steps {
            let Source::Made(body) = step.kernel.source() else {
                continue;
            };
            if !seen.insert(body.entry.clone()) {
                continue;
            }
            match body.wgsl() {
                Ok(module) => {
                    let checked = validated(&module.text);
                    checked.unwrap_or_else(|e| panic!("{}:\n{e}\n{}", step.rule, module.text));
                }
                Err(why) => *refused.entry(format!("{}: {why}", step.rule)).or_default() += 1,
            }
        }
    }
    println!("{} kernels, refused: {refused:#?}", seen.len());
    assert!(seen.len() > refused.values().sum::<usize>());
}

/// A layer cut from a model keeps its weights.
#[test]
fn one_layer_plans() {
    let s = &crate::plans::subjects()[0];
    let layer = s.graph.regions.iter().find(|r| r.kind == Composite::Layer);
    let layer = layer.unwrap();
    let g = cut_region(&s.graph, layer);
    verify(&g).unwrap();
    let weight = |i: &&NodeId| matches!(s.graph.node(**i).op, Op::Weight(_));
    assert_eq!(g.weights.len(), layer.inputs.iter().filter(weight).count());
    assert!(g.regions.iter().any(|r| r.kind == Composite::Attention));
    let device = Device::guess(&settings()).pinned(s.pins.clone());
    lower(&g, 0, &crate::plans::sizes(8, 8), &device).unwrap();
}
