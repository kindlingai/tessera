use super::{
    ACTIVE, CheckConfig, Collected, check_file, check_source, compile_groups, sm_target, start,
    target_platform, verify_collected,
};
use crate::Run;
use crate::overrides::Overrides;
use crate::run::Outcome;
use tessera_engine_comm::{Error as CommError, Transport};
use tessera_engine_settings::{Backend, Pick, Settings};
use tessera_gpukit::backend::Backend as GpuBackend;
use tessera_lower::Plan;

struct DummyRank;

impl Transport for DummyRank {
    fn rank(&self) -> u32 {
        0
    }

    fn ranks(&self) -> u32 {
        2
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, CommError> {
        Ok(vec![mine.to_vec(); 2])
    }
}

#[test]
fn checks_platform_and_config() {
    let p = target_platform("sm_121").unwrap();
    assert_eq!((p.backend.as_str(), p.arch.as_str()), ("cuda", "sm_121"));
    assert_eq!(p.name, "NVIDIA GB10");
    assert!(p.tabled() && p.integrated && p.limits.compiler && p.limits.pdl);
    let dev = tessera_lower::Device::stored(p, &Settings::default()).unwrap();
    assert_eq!(dev.name, "cuda NVIDIA GB10");
    let p120 = target_platform("sm_120").unwrap();
    assert_eq!((p120.arch.as_str(), p120.integrated), ("sm_120", false));
    assert!(p120.tabled() && p120.limits.pdl);
    assert_eq!(sm_target("sm_121").unwrap(), "sm_121a");
    assert_eq!(sm_target("sm_89").unwrap(), "sm_89");
    assert!(sm_target("sm_12é").is_err());

    let bad = CheckConfig {
        run_ptxas: false,
        max_spills: Some(0),
        ..CheckConfig::default()
    };
    assert!(check_source("k.cu", None, "", &[], &bad).is_err());
}

#[test]
fn isolates_sessions() {
    let s1 = start("sm_121").unwrap();
    assert!(start("sm_121").is_err());
    assert!(super::active());

    let other = std::thread::spawn(|| {
        assert!(!super::active());
        let plan = Plan::default();
        assert!(super::open(&plan).is_none());
        let runner = crate::open(plan, &Settings::default()).unwrap();
        assert!(!runner.device().contains("offline check"));
    });
    other.join().unwrap();

    let ranked_plan = Plan {
        ranks: 2,
        rank: 0,
        ..Plan::default()
    };
    let gpu_settings = Settings {
        backend: Pick::Gpu(Backend::Cuda),
        ..Settings::default()
    };
    let ranked = crate::Rank::open(ranked_plan, &gpu_settings, DummyRank).unwrap();
    assert!(ranked.device().contains("offline check"));

    let cfg = CheckConfig {
        run_ptxas: false,
        ..CheckConfig::default()
    };
    let empty_plan = s1.finish(&cfg).unwrap();
    assert!(empty_plan.kernels.is_empty() && !empty_plan.ok());
    assert!(!super::active());

    let lost = start("sm_121").unwrap();
    ACTIVE.with(|cell| {
        *cell.borrow_mut() = None;
    });
    assert!(lost.finish(&cfg).is_err());
}

#[test]
fn checks_sources_offline() {
    if !cfg!(feature = "cuda") {
        return;
    }
    let cfg = CheckConfig {
        run_ptxas: false,
        ..CheckConfig::default()
    };
    let good = "extern \"C\" __global__ void good_k(float* y) { y[0] = 2.0f; }\n";
    let report = check_source("good_k.cu", Some("good_k"), good, &[], &cfg).unwrap();
    if report.nvrtc.is_none() {
        return;
    }
    assert!(report.ok() && report.kernels[0].ptx_bytes > 0);

    let no_entry = "__device__ int helper(int x) { return x + 1; }\n";
    let empty_rep = check_source("helper.cu", None, no_entry, &[], &cfg).unwrap();
    assert!(!empty_rep.ok() && empty_rep.failed() == 1);

    let bad = "extern \"C\" __global__ void bad_k(float* y) {\n    y[0] = no_such_var;\n}\n";
    let err = check_source("bad_k.cu", Some("bad_k"), bad, &[], &cfg)
        .unwrap()
        .kernels[0]
        .error
        .clone()
        .unwrap_or_default();
    assert!(err.contains("bad_k.cu") && err.contains("(2)"), "{err}");

    let miss = check_source("good_k.cu", Some("wrong_name"), good, &[], &cfg).unwrap();
    assert!(!miss.ok());
}

fn collected(entry: &str, source: &str) -> Collected {
    Collected {
        entry: entry.into(),
        rule: "test".into(),
        kind: "fixed",
        source: source.into(),
        flags: Vec::new(),
        prep_error: None,
        resident: false,
        dynamic_shared: 0,
        threads: 32,
    }
}

#[test]
fn distinguishes_unit_names() {
    let shared = "\
static_assert(__FILE__[0] == 'a', \"unit name mismatch\");
extern \"C\" __global__ void alpha(float* y) { y[0] = 1.0f; }
extern \"C\" __global__ void beta(float* y) { y[0] = 2.0f; }
";
    let items = vec![
        collected("alpha", shared),
        collected("alpha.cu", shared),
        collected("beta", shared),
    ];
    let groups = compile_groups(&items);
    assert_eq!(groups.len(), 2);
    assert_eq!(
        groups[&("alpha.cu".into(), shared, &[][..], None)],
        vec![0, 1]
    );
    assert_eq!(groups[&("beta.cu".into(), shared, &[][..], None)], vec![2]);

    if !cfg!(feature = "cuda") {
        return;
    }
    let cfg = CheckConfig {
        run_ptxas: false,
        ..CheckConfig::default()
    };
    let report = verify_collected(
        vec![collected("alpha", shared), collected("beta", shared)],
        &cfg,
    )
    .unwrap();
    if report.nvrtc.is_none() {
        return;
    }
    assert_eq!(report.kernels.len(), 2);
    assert!(report.kernels[0].error.is_none(), "{:?}", report.kernels[0]);
    let err = report.kernels[1].error.as_deref().unwrap_or("");
    assert!(
        err.contains("beta.cu") && err.contains("unit name mismatch"),
        "{err}"
    );
    assert!(!report.ok());
}

#[test]
fn checks_override_headers_flags() {
    let (mut a, mut b) = (collected("flag_k", "src"), collected("flag_k", "src"));
    a.flags = vec!["-DMY_SCALE=4".into()];
    b.flags = vec!["-DMY_SCALE=8".into()];
    assert_eq!(compile_groups(&[a, b]).len(), 2);

    let dir = tessera_testkit::scratch(&format!("check-ov-flags-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let cu = dir.join("flag_k.cu");
    let cfg_path = dir.join("flag_k.launch.toml");
    std::fs::write(
        dir.join("scale.cuh"),
        "#pragma once\n__device__ int get_scale() { return MY_SCALE; }\n",
    )
    .unwrap();
    std::fs::write(
        &cu,
        "#include \"scale.cuh\"\nextern \"C\" __global__ void flag_k(int* y) { y[0] = get_scale(); }\n",
    )
    .unwrap();
    std::fs::write(
        &cfg_path,
        "threads = 128\ndynamic_shared = 2048\nnvrtc_flags = [\"-DMY_SCALE=4\"]\n",
    )
    .unwrap();

    // Runtime overrides path prepares the exact same source and flags.
    let mut ov = Overrides::new(dir.clone(), GpuBackend::Cuda);
    let (mut seen_src, mut seen_flags) = (String::new(), Vec::new());
    let res = ov
        .check(
            "flag_k",
            false,
            || "planned".into(),
            |src, flags| {
                seen_src = src.to_string();
                seen_flags = flags.iter().map(|s| (*s).to_string()).collect();
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(res.outcome, Outcome::Loaded);
    assert!(seen_src.contains("get_scale") && seen_flags == vec!["-DMY_SCALE=4"]);

    // Resident override with launch.toml fails offline check explicitly.
    let mut res_item = collected("flag_k", "planned");
    res_item.resident = true;
    let cfg = CheckConfig {
        run_ptxas: false,
        overrides: Some(dir.clone()),
        ..CheckConfig::default()
    };
    let res_rep = verify_collected(vec![res_item], &cfg).unwrap();
    assert!(
        !res_rep.ok()
            && res_rep.kernels[0]
                .error
                .as_deref()
                .unwrap_or("")
                .contains("resident")
    );

    if cfg!(feature = "cuda") {
        let rep = verify_collected(vec![collected("flag_k", "planned")], &cfg).unwrap();
        if rep.nvrtc.is_some() {
            let k = &rep.kernels[0];
            assert!(
                rep.ok()
                    && k.kind == "override"
                    && k.threads == 128
                    && k.dynamic_shared == 2048
                    && k.ptx_bytes > 0
            );
            assert!(check_file(&cu, Some("flag_k"), &cfg).unwrap().ok());
            std::fs::remove_file(&cfg_path).unwrap();
            let miss = verify_collected(vec![collected("flag_k", "planned")], &cfg).unwrap();
            assert!(
                !miss.ok()
                    && miss.kernels[0]
                        .error
                        .as_deref()
                        .unwrap_or("")
                        .contains("MY_SCALE")
            );
        }
    }
    // Single-read I/O errors preserve exact KernelFile error.
    let missing = dir.join("absent.cu");
    assert!(matches!(
        check_file(&missing, None, &cfg),
        Err(crate::Error::KernelFile(p, io)) if p == missing && io.kind() == std::io::ErrorKind::NotFound
    ));
    assert!(
        matches!(check_file(&dir, None, &cfg), Err(crate::Error::KernelFile(p, _)) if p == dir)
    );
    let _ = std::fs::remove_dir_all(&dir);
}
