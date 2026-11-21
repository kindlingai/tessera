//! The tiled matmul against the CPU interpreter, on the GPU.

// The other matmul rules come from here.
use tessera_kernels as _;

use tessera_codegen::trellis;
use tessera_graph::DimName::Rows;
use tessera_interp::{Tensor, run};

mod common;
use common::{linear, noise, terms};
use tessera_gpukit::platform::{Asked, Platform};
use tessera_graph::F16;
use tessera_linear::{PARTS, SCHEDULES, Schedule};
use tessera_lower::form::{BF16_FORM, LOADS, named};
use tessera_lower::{Device, Dialect, Pins, Settings, Source, Weight, lower, select};
use tessera_run::open;
use tessera_run::testing::{emulations, host, runs};
use tessera_testkit::{Noise, cosine_f32, max_abs_diff};

/// A device on `platform` that can only choose schedule `name`.
fn only(name: &str, settings: &Settings, platform: &Platform) -> Device {
    let others = SCHEDULES.iter().chain(&PARTS).filter(|s| s.name != name);
    let bar = |d: Device, s: &Schedule| LOADS.iter().fold(d, |d, f| d.without(f.suffixed(s.name)));
    let device = Device::guess(settings).on(platform.clone());
    others.fold(device.favoring("tiled"), bar)
}

/// The platform a runner for `settings` reports.
fn platform(settings: &Settings) -> Platform {
    let plan = lower(&linear(4, 4), 0, &[(Rows, 1)], &Device::guess(settings)).unwrap();
    open(plan, settings).unwrap().platform()
}

/// Calibrate reads the ways to time each schedule.
#[test]
fn ways_list_schedules() {
    let device = Device::guess(&host());
    let picked = select(&linear(72, 50), 0, &[(Rows, 37)], &device).unwrap();
    assert_eq!(picked.ways["tiled"].len(), SCHEDULES.len());
}

#[test]
fn rule_registers() {
    let g = linear(72, 50);
    let device = Device::guess(&host()).favoring("tiled");
    let plan = lower(&g, 0, &[(Rows, 37)], &device).unwrap();
    let Source::Made(body) = plan.steps[0].kernel.source() else {
        panic!("tiled is generated");
    };
    for dialect in [Dialect::Metal, Dialect::Cuda] {
        let text = body.text().render(dialect);
        assert!(text.contains("TILE_MMA(acc_t[i][j], ta[i], tb[j])"));
    }
}

#[test]
fn tiles_match_cpu() {
    let mut draws = Noise(29);
    // Sizes off the block, on it, beyond it, over bands.
    let sizes = [
        (1, 3, 4),
        (37, 50, 72),
        (64, 64, 64),
        (600, 40, 512),
        (256, 512, 1024),
    ];
    let runs = emulations();
    let platforms: Vec<Platform> = runs.iter().map(platform).collect();
    for (m, n, k) in sizes {
        let g = linear(k, n);
        let x = noise(vec![m as usize, k as usize], &mut draws);
        for form in common::forms(k as usize) {
            // The reference sees the weights in the kernel's stored form.
            let w = noise(vec![n as usize, k as usize], &mut draws);
            let rounded = form.roundtrip(w.floats().unwrap());
            let weights = [Tensor::f32(w.shape.clone(), rounded)];
            let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
            // bf16 weights also multiply from parts.
            let parts = PARTS.iter().filter(|_| form.name() == BF16_FORM.name());
            let cases = || runs.iter().zip(&platforms).cycle();
            let listed = SCHEDULES.iter().zip(cases()).chain(parts.zip(cases()));
            for (schedule, (settings, platform)) in listed {
                let is_parts = PARTS.contains(schedule);
                if is_parts && platform.bf16_tile().is_none() {
                    continue;
                }
                // The host skips the largest size and the timing.
                let host = settings.gpu().is_none();
                if host && (m, n, k) == sizes[sizes.len() - 1] {
                    continue;
                }
                let device = only(schedule.name, settings, platform).pinned(Pins::all(form));
                let plan = lower(&g, 0, &[(Rows, m)], &device).unwrap();
                let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
                let name = form.suffixed(schedule.name);
                assert_eq!(names, [name]);
                let terms = terms(&plan);
                let mut runner = open(plan, settings).unwrap();
                runner.load(0, weights[0].view()).unwrap();
                let got = runner.run(&[x.view()]).unwrap().remove(0);
                let off = max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
                let (xs, ws) = (x.floats().unwrap(), weights[0].floats().unwrap());
                let bound = terms.matmul(&runner.precision(), xs, ws, k as usize);
                let label = format!("{m}x{n}x{k} {name} on {}", runner.device());
                assert!(off <= bound, "{label}: off by {off:e}, allowed {bound:e}");
                // A bound past the outputs themselves says nothing.
                let most = want[0]
                    .floats()
                    .unwrap()
                    .iter()
                    .fold(0.0f32, |m, v| m.max(v.abs()));
                assert!(
                    bound < most / 10.0,
                    "{label}: {bound:e} against outputs of {most:e}"
                );
                if host {
                    eprintln!("{label}: off by {off:.2e} of {bound:.2e}");
                    continue;
                }
                let seconds = runner.run_timed(&[x.view()]).unwrap()[0];
                let rate = 2.0 * (m * n * k) as f64 / seconds / 1e12;
                eprintln!("{label}: {rate:.2} TFLOP/s, off by {off:.2e} of {bound:.2e}");
            }
        }
    }
}

#[test]
fn codes_match_decode() {
    let g = linear(512, 256);
    let mut draws = Noise(3);
    let (inp, out) = (512, 256);
    // Rotated files add f16 `suh` and `svh` after the codes.
    let forms = [
        (2, false, ""),
        (3, false, ""),
        (4, false, ""),
        (2, true, ""),
        (4, true, ""),
        (3, false, "_mul1"),
        (3, true, "_mul1"),
        (2, false, "_pmcgx"),
        (3, true, "_pmcgx"),
        (4, true, "_pmcgx"),
    ];
    for (k, turned, book) in forms {
        let codes = named(&format!("trellis_k{k}{book}")).unwrap();
        let entries: Vec<u8> = (0..out * inp * k as usize / 8)
            .map(|_| (draws.draw() * 256.0) as u8)
            .collect();
        let mut bytes = trellis::packed(inp, out, 1, &entries);
        let mut form = codes;
        if turned {
            let scale =
                (0..inp + out).flat_map(|i| F16::from(1.0 + (i % 7) as f32 / 8.0).0.to_le_bytes());
            bytes.extend(scale);
            form = named(&format!("{}_rot", codes.name())).unwrap();
        }
        let weights = [Tensor::f32(vec![out, inp], form.unpack_all(&bytes))];
        // Few rows take lane or fused kernels, many the tiled.
        let mut cases = Vec::new();
        for settings in runs() {
            for rows in [3usize, 40] {
                cases.push((settings.clone(), rows, false));
                if turned && rows == 3 {
                    cases.push((settings.clone(), rows, true));
                }
            }
        }
        for (settings, rows, fused) in cases {
            let x = noise(vec![rows, inp], &mut draws);
            let want = run(&g, &weights, 0, std::slice::from_ref(&x)).unwrap();
            let mut device = Device::guess(&settings).pinned(Pins::all(form));
            if fused {
                device = device.favoring("turned");
            }
            let plan = lower(&g, 0, &[(Rows, rows as u64)], &device).unwrap();
            let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
            let reads = |n: &&str| n.contains(codes.name());
            assert!(names.iter().any(reads), "{names:?} read the codes");
            if fused {
                assert_eq!(names, [codes.suffixed("turned")]);
            }
            let mut runner = open(plan, &settings).unwrap();
            let (compiled, made) = runner.compiled();
            assert_eq!(compiled, made, "{}: every tree compiles", form.name());
            runner
                .load_weight(0, &Weight::Packed(form, bytes.clone()))
                .unwrap();
            let got = runner.run(&[x.view()]).unwrap().remove(0);
            let cos = cosine_f32(got.floats().unwrap(), want[0].floats().unwrap());
            assert!(cos > 0.99999, "{}, {rows} rows: cos {cos}", form.name());
        }
    }
}

/// Unsampled, a known chip's table picks its fastest schedule.
#[test]
fn tables_pick_schedules() {
    let chips = [
        ("cuda", "sm_121", "NVIDIA GB10"),
        ("metal", "apple9", "Apple M3 Pro"),
    ];
    for (backend, arch, name) in chips {
        let platform = Platform::new(Asked {
            backend,
            arch: arch.into(),
            name: name.into(),
            driver: String::new(),
            limits: tessera_gpukit::Limits::assumed(),
            units: 0,
            bandwidth: 0.0,
            integrated: true,
            memory: 0,
        });
        let device = Device::guess(&host())
            .on(platform)
            .pinned(Pins::all(BF16_FORM));
        let plan = lower(&linear(3072, 3072), 0, &[(Rows, 256)], &device).unwrap();
        let names: Vec<&str> = plan.steps.iter().map(|s| s.kernel.name()).collect();
        assert_eq!(names, ["tiled_64x64_w32x16_bf16"], "{name}");
    }
}

/// Neighbours gathered while staged, against the interpreter.
#[test]
fn near_matches_cpu() {
    let g =
        tessera_testkit::graphs::neighbours(&[tessera_trace::Dim::bounded(Rows, 64)], [12, 9, 32]);
    let mut draws = Noise(31);
    // Ids past the twelve rows read zeros.
    let ids = (0..12 * 9).map(|i| ((i / 9 + i % 9) % 13) as f32).collect();
    let weights = [
        Tensor::f32(vec![12, 9], ids),
        noise(vec![32, 288], &mut draws),
    ];
    let x = [0, 1].map(|_| noise(vec![2, 12, 32], &mut draws));
    let want = run(&g, &weights, 0, &x).unwrap().remove(1);
    for settings in runs() {
        let plain = |d: Device, s: &Schedule| d.without(s.name);
        let device = SCHEDULES
            .iter()
            .fold(Device::guess(&settings).favoring("tiled"), plain);
        let plan = lower(&g, 0, &[(Rows, 2)], &device).unwrap();
        assert!(
            plan.steps
                .iter()
                .any(|s| s.kernel.name().starts_with("near_"))
        );
        let mut runner = open(plan, &settings).unwrap();
        for (at, w) in weights.iter().enumerate() {
            runner.load(at as u32, w.view()).unwrap();
        }
        let got = runner
            .run(&x.each_ref().map(Tensor::view))
            .unwrap()
            .remove(1);
        let (got, want) = (got.floats().unwrap(), want.floats().unwrap());
        let most = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let off = max_abs_diff(got, want);
        assert!(
            off < most * 1e-4,
            "{}: off by {off:e} of {most:e}",
            runner.device()
        );
    }
}
