use super::{
    ACTIVE, CheckConfig, PtxStats, TempDir, check_source, eval_entry, find_ptxas, parse_ptxas,
    ptx_entries, sm_target, start, target_platform,
};
use crate::Run;
use std::collections::HashMap;
use tessera_engine_comm::{Error as CommError, Transport};
use tessera_engine_settings::{Backend, Pick, Settings};
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
    assert!(p.limits.compiler && p.limits.pdl);
    assert_eq!(sm_target("sm_121").unwrap(), "sm_121a");
    assert_eq!(sm_target("sm_89").unwrap(), "sm_89");
    assert!(sm_target("sm_12é").is_err());

    let bad = CheckConfig {
        run_ptxas: false,
        max_spills: Some(0),
        ..CheckConfig::default()
    };
    assert!(check_source("k.cu", None, "", &bad).is_err());
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
    assert!(s1.finish(&cfg).unwrap().kernels.is_empty());
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
    let report = check_source("good_k.cu", Some("good_k"), good, &cfg).unwrap();
    if report.nvrtc.is_none() {
        return;
    }
    assert!(report.ok() && report.kernels[0].ptx_bytes > 0);

    let no_entry = "__device__ int helper(int x) { return x + 1; }\n";
    let empty_rep = check_source("helper.cu", None, no_entry, &cfg).unwrap();
    assert!(!empty_rep.ok() && empty_rep.failed() == 1);

    let bad = "extern \"C\" __global__ void bad_k(float* y) {\n    y[0] = no_such_var;\n}\n";
    let err = check_source("bad_k.cu", Some("bad_k"), bad, &cfg)
        .unwrap()
        .kernels[0]
        .error
        .clone()
        .unwrap_or_default();
    assert!(err.contains("bad_k.cu") && err.contains("(2)"), "{err}");

    let miss = check_source("good_k.cu", Some("wrong_name"), good, &cfg).unwrap();
    assert!(!miss.ok());

    if find_ptxas().is_some() {
        let strict = CheckConfig {
            max_spills: Some(0),
            run_ptxas: true,
            ..CheckConfig::default()
        };
        let clean = check_source("good_k.cu", Some("good_k"), good, &strict).unwrap();
        assert!(clean.ok(), "{:?}", clean.kernels);
        assert_eq!(clean.kernels[0].stats.unwrap().spill_bytes(), Some(0));

        let spill = "extern \"C\" __global__ void spill_k(float* y, int idx) {\n\
            volatile float buf[64];\n\
            for (int i = 0; i < 64; i++) buf[i] = (float)(i + idx);\n\
            y[0] = buf[idx & 63];\n\
        }\n";
        let rep = check_source("spill_k.cu", Some("spill_k"), spill, &strict).unwrap();
        assert!(!rep.ok() && rep.spilling() == 1);
    }
}
