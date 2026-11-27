use super::{PtxStats, TempDir, find_ptxas, parse_ptxas, ptx_entries};
use crate::check::{CheckConfig, check_source, eval_entry};
use std::collections::HashMap;

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
    let fails = |name: &str, asm| eval_entry(name, false, &found, asm, Some(0)).1.is_some();
    assert!(fails("k_header_only", Some(&ok)) && fails("k_partial", Some(&ok)));
    assert!(!fails("k_clean", Some(&ok)) && fails("k_clean", None));
    let empty: Result<HashMap<String, PtxStats>, String> = Ok(HashMap::new());
    assert!(fails("k_clean", Some(&empty)));
}

#[test]
fn makes_private_dirs() {
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
fn checks_spills_with_ptxas() {
    if !cfg!(feature = "cuda") || find_ptxas().is_none() {
        return;
    }
    let strict = CheckConfig {
        max_spills: Some(0),
        ..CheckConfig::default()
    };
    let good = "extern \"C\" __global__ void good_k(float* y) { y[0] = 2.0f; }\n";
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

#[test]
fn lists_entries() {
    let ptx = ".visible .entry alpha(\n)\n.entry beta (\n)\n";
    assert_eq!(ptx_entries(ptx), vec!["alpha", "beta"]);
}
