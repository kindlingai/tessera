//! Each GPU toolkit compiles every kernel small models plan.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use tessera_gpukit::backend::Backend;
use tessera_gpukit::platform::{Asked, Platform, backend_of, rows};
use tessera_lower::form::BF16_FORM;
use tessera_lower::{Device, Pins, lower};
use tessera_models::decoder::{Config, Moe, Swiglu, stepper};
use tessera_models::mla;
use tessera_rules as _;
use tessera_run::kernel_text;
use tessera_run::testing::host;
use tessera_testkit::{run, scratch, toolkit};
use tessera_trace::DimName::{Context, Rows, Seqs};
use tessera_trace::Norm;

/// Each table row's compiler targets on `backend`, as platforms.
fn platforms(backend: Backend) -> Vec<Platform> {
    let name = match backend {
        Backend::Cuda => "cuda",
        Backend::Rocm => "rocm",
        Backend::Metal => "metal",
    };
    let mut out = Vec::new();
    for row in rows().iter().filter(|r| backend_of(r) == name) {
        for target in &row.check {
            out.push(Platform::new(Asked {
                backend: name,
                arch: target.clone(),
                name: target.clone(),
                driver: String::new(),
                limits: tessera_gpukit::Limits::assumed(),
                units: 0,
                bandwidth: 0.0,
                integrated: true,
                memory: 0,
            }));
        }
    }
    out
}

/// Each kernel text small models plan, and its targets.
fn planned(backend: Backend) -> BTreeMap<String, BTreeSet<String>> {
    let graphs = [
        stepper::<Norm, Swiglu>(&Config::qwen3_small()),
        stepper::<Norm, Moe>(&Config::qwen3_moe_small()),
        mla::stepper::<mla::RotatedMoe>(&mla::Config::glm53_small()),
    ];
    let pins = [Pins::default(), Pins::all(BF16_FORM).activations(BF16_FORM)];
    let mut texts: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for platform in platforms(backend) {
        for (g, pins) in graphs.iter().flat_map(|g| pins.iter().map(move |p| (g, p))) {
            let device = Device::guess(&host())
                .on(platform.clone())
                .pinned(pins.clone());
            for rows in [1, 64] {
                let sizes = [(Seqs, 4), (Rows, rows), (Context, 64)];
                // Some tables cannot cover every form. Nothing to compile.
                let Ok(plan) = lower(g, 0, &sizes, &device) else {
                    println!("{} rows={rows}: does not lower, skipped", platform.arch);
                    continue;
                };
                for step in &plan.steps {
                    let text = kernel_text(step, backend);
                    texts.entry(text).or_default().insert(platform.arch.clone());
                }
            }
        }
    }
    texts
}

/// Compiles planned texts in parallel, per target when `each`.
fn compile_all(
    backend: Backend,
    (ext, each): (&str, bool),
    command: impl Fn(&str, &str) -> Vec<String> + Sync,
) {
    let dir = scratch(&format!("compiles-{ext}"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("hiprtc.h"), include_str!("hiprtc.h")).unwrap();
    let mut jobs = Vec::new();
    for (at, (text, targets)) in planned(backend).into_iter().enumerate() {
        let file = format!("k{at}.{ext}");
        std::fs::write(dir.join(&file), text).unwrap();
        for target in targets.into_iter().take(if each { usize::MAX } else { 1 }) {
            jobs.push((file.clone(), target));
        }
    }
    let width = std::thread::available_parallelism().map_or(4, |n| n.get());
    let failed = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for chunk in jobs.chunks(jobs.len().div_ceil(width).max(1)) {
            let (dir, failed, command) = (&dir, &failed, &command);
            s.spawn(move || {
                for (file, target) in chunk {
                    let args = command(file, target);
                    let rest: Vec<&str> = args[1..].iter().map(String::as_str).collect();
                    let (ok, log) = run(&args[0], &rest, dir);
                    if !ok {
                        failed
                            .lock()
                            .unwrap()
                            .push(format!("{file} {target}:\n{log}"));
                    }
                }
            });
        }
    });
    let failed = failed.into_inner().unwrap();
    println!("{ext}: {} compiles, {} failed", jobs.len(), failed.len());
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// Words as owned arguments.
fn words(text: &str) -> Vec<String> {
    text.split_whitespace().map(String::from).collect()
}

/// nvcc compiles each planned kernel as NVRTC would.
#[test]
fn cuda_kernels_compile() {
    if !toolkit("cuda") {
        return;
    }
    let (_, listed) = run("nvcc", &["--list-gpu-arch"], Path::new("."));
    let known = |version: &str| {
        listed
            .lines()
            .any(|l| l.trim() == format!("compute_{version}"))
    };
    for platform in platforms(Backend::Cuda) {
        if !known(platform.arch.trim_start_matches("sm_")) {
            println!("cuda {}: nvcc does not list it, skipped", platform.arch);
        }
    }
    compile_all(Backend::Cuda, ("cu", true), |file, target| {
        let version = target.trim_start_matches("sm_");
        if !known(version) {
            return words("true");
        }
        // The runner's spelling: compute_XY, with `a` from Blackwell on.
        let major: u32 = version[..version.len() - 1].parse().unwrap();
        let suffix = if major >= 12 { "a" } else { "" };
        let mut args = words("nvcc --std=c++17 --diag-suppress=177 -ptx -o /dev/null");
        args.extend([
            format!("--gpu-architecture=compute_{version}{suffix}"),
            file.into(),
        ]);
        args
    });
}

/// clang compiles each planned kernel as hiprtc would.
#[test]
fn rocm_kernels_compile() {
    if !toolkit("rocm") {
        return;
    }
    compile_all(Backend::Rocm, ("hip", true), |file, target| {
        let mut args = words(
            "clang++ -x hip -std=c++17 --cuda-device-only -nogpulib -nogpuinc -D__HIPCC_RTC__ \
             -include __clang_hip_runtime_wrapper.h -include hiprtc.h -O3 -S -o /dev/null",
        );
        args.extend([format!("--offload-arch={target}"), file.into()]);
        args
    });
}

/// The Metal compiler builds each planned kernel.
#[test]
fn metal_kernels_compile() {
    if !toolkit("metal") {
        return;
    }
    compile_all(Backend::Metal, ("metal", false), |file, _| {
        let mut args = words("xcrun -sdk macosx metal -c -o /dev/null");
        args.push(file.into());
        args
    });
}
