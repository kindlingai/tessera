use super::{
    ACTIVE, CheckConfig, Collected, PtxStats, TempDir, check_file, check_source, compile_groups,
    eval_entry, find_ptxas, parse_ptxas, ptx_entries, sm_target, start, target_platform,
    verify_collected,
};
use crate::Run;
use crate::overrides::Overrides;
use crate::run::Outcome;
use std::collections::HashMap;
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
fn parses_ptxas_states() {
    let log = "\
ptxas info    : Compiling entry function 'k_header_only' for 'sm_121a'
ptxas info    : Compiling entry function 'k_partial' for 'sm_121a'
ptxas info    : Used 12 registers, used 0 barriers
ptxas info    : Compiling entry function 'k_spill' for 'sm_121a'
ptxas info    : Function properties for helper_dev
    64 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Function properties for k_spill
    256 bytes stack frame, 16 bytes spill stores, 24 bytes spill loads
ptxas info    : Used 64 registers, used 1 barriers, 4096 bytes smem, 256 bytes cumulative stack size
ptxas info    : Compiling entry function 'k_clean' for 'sm_121a'
ptxas info    : Function properties for k_clean
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers
";
    let parsed = parse_ptxas(log);
    assert!(!parsed.contains_key("k_header_only"));
    assert_eq!(parsed["k_partial"].regs, Some(12));
    assert_eq!(parsed["k_partial"].spill_bytes(), None);
    assert_eq!(
        parsed.get("k_spill"),
        Some(&PtxStats {
            regs: Some(64),
            smem: 4096,
            stack_bytes: Some(256),
            spill_stores: Some(16),
            spill_loads: Some(24),
            barriers: 1,
        })
    );
    assert_eq!(parsed["k_spill"].spill_bytes(), Some(256));
    assert_eq!(parsed["k_clean"].spill_bytes(), Some(0));

    let found = vec![
        "k_header_only".into(),
        "k_partial".into(),
        "k_spill".into(),
        "k_clean".into(),
    ];
    let ok = Ok(parsed);
    let (_, absent_err) = eval_entry("k_header_only", false, &found, Some(&ok), Some(0));
    assert!(absent_err.is_some());
    let (_, partial_err) = eval_entry("k_partial", false, &found, Some(&ok), Some(0));
    assert!(partial_err.is_some());
    let (_, clean_err) = eval_entry("k_clean", false, &found, Some(&ok), Some(0));
    assert!(clean_err.is_none());
    let (_, no_ptxas_err) = eval_entry("k_clean", false, &found, None, Some(0));
    assert!(no_ptxas_err.is_some());
    let empty: Result<HashMap<String, PtxStats>, String> = Ok(HashMap::new());
    let (_, empty_err) = eval_entry("k_clean", false, &found, Some(&empty), Some(0));
    assert!(empty_err.is_some());
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
    let ptx = ".visible .entry alpha(\n)\n.entry beta (\n)\n";
    assert_eq!(ptx_entries(ptx), vec!["alpha", "beta"]);
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

    let root = tessera_testkit::scratch(&format!("check-tmp-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&root);
    let keep = root.join("keep.txt");
    std::fs::write(&keep, b"ok").unwrap();
    let td = TempDir::create(&root).unwrap();
    let created = td.0.clone();
    assert!(created.is_dir());
    drop(td);
    assert!(!created.exists() && keep.is_file());
    let _ = std::fs::remove_dir_all(&root);
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

    if find_ptxas().is_some() {
        let strict = CheckConfig {
            max_spills: Some(0),
            run_ptxas: true,
            ..CheckConfig::default()
        };
        let clean = check_source("good_k.cu", Some("good_k"), good, &[], &strict).unwrap();
        assert!(clean.ok(), "{:?}", clean.kernels);
        assert_eq!(clean.kernels[0].stats.unwrap().spill_bytes(), Some(0));

        let spill = "extern \"C\" __global__ void spill_k(float* y, int idx) {\n\
            volatile float buf[64];\n\
            for (int i = 0; i < 64; i++) buf[i] = (float)(i + idx);\n\
            y[0] = buf[idx & 63];\n\
        }\n";
        let rep = check_source("spill_k.cu", Some("spill_k"), spill, &[], &strict).unwrap();
        assert!(!rep.ok() && rep.spilling() == 1);
    }
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
