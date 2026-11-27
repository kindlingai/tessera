use super::{
    CheckConfig, PtxStats, check_source, find_ptxas, parse_ptxas, ptx_entries, sm_target,
    target_platform,
};

#[test]
fn parses_ptxas_output() {
    let log = "\
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'k_spill' for 'sm_121a'
ptxas info    : Function properties for helper_dev
    64 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Function properties for k_spill
    256 bytes stack frame, 16 bytes spill stores, 24 bytes spill loads
ptxas info    : Used 64 registers, used 1 barriers, 4096 bytes smem, 256 bytes cumulative stack size, 384 bytes cmem[0]
ptxas info    : Compiling entry function 'k_clean' for 'sm_121a'
ptxas info    : Function properties for k_clean
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers
";
    let parsed = parse_ptxas(log);
    assert_eq!(
        parsed.get("k_spill"),
        Some(&PtxStats {
            regs: 64,
            smem: 4096,
            stack_bytes: 256,
            spill_stores: 16,
            spill_loads: 24,
            barriers: 1,
        })
    );
    assert_eq!(parsed["k_spill"].spill_bytes(), 256);
    assert_eq!(
        parsed.get("k_clean"),
        Some(&PtxStats {
            regs: 12,
            smem: 0,
            stack_bytes: 0,
            spill_stores: 0,
            spill_loads: 0,
            barriers: 0,
        })
    );
    assert_eq!(parsed["k_clean"].spill_bytes(), 0);
}

#[test]
fn builds_offline_platform() {
    let p = target_platform("sm_121").unwrap();
    assert_eq!(p.backend, "cuda");
    assert_eq!(p.arch, "sm_121");
    assert!(p.limits.compiler && p.limits.pdl);
    assert_eq!(p.mxfp8_tile(), Some([16, 8, 32]));
    assert_eq!(sm_target("sm_121").unwrap(), "sm_121a");
    assert_eq!(sm_target("sm_89").unwrap(), "sm_89");
}

#[test]
fn extracts_ptx_entries() {
    let ptx = ".visible .entry alpha(\n.param .u64 x\n)\n.entry beta (\n)\n";
    assert_eq!(ptx_entries(ptx), vec!["alpha", "beta"]);
}

#[test]
fn checks_sources_offline() {
    if !cfg!(feature = "cuda") {
        return;
    }
    let cfg = CheckConfig {
        arch: "sm_121".into(),
        max_spills: None,
        ptxas: None,
        run_ptxas: false,
        overrides: None,
        temp: None,
    };
    let good = "extern \"C\" __global__ void good_k(float* y) { y[0] = 2.0f; }\n";
    let report = check_source("good_k.cu", Some("good_k"), good, &cfg).unwrap();
    if report.nvrtc.is_none() {
        return;
    }
    assert!(report.ok(), "{:?}", report.kernels);
    assert_eq!(report.kernels.len(), 1);
    assert!(report.kernels[0].ptx_bytes > 0);

    let bad = "extern \"C\" __global__ void bad_k(float* y) {\n    y[0] = no_such_var;\n}\n";
    let bad_rep = check_source("bad_k.cu", Some("bad_k"), bad, &cfg).unwrap();
    assert!(!bad_rep.ok());
    let err = bad_rep.kernels[0].error.as_deref().unwrap_or("");
    assert!(err.contains("bad_k.cu") && err.contains("(2)"), "{err}");

    let missing = check_source("good_k.cu", Some("wrong_name"), good, &cfg).unwrap();
    assert!(!missing.ok());
    let err = missing.kernels[0].error.as_deref().unwrap_or("");
    assert!(
        err.contains("wrong_name") && err.contains("good_k"),
        "{err}"
    );

    if find_ptxas().is_some() {
        let spill_src = "extern \"C\" __global__ void spill_k(float* y, int idx) {\n\
            float buf[256];\n\
            for (int i = 0; i < 256; i++) buf[i] = y[i];\n\
            buf[idx & 255] = y[256];\n\
            float s = 0.0f;\n\
            for (int i = 0; i < 16; i++) s += buf[(idx + i * 13) & 255];\n\
            y[0] = s;\n\
        }\n";
        let strict = CheckConfig {
            arch: "sm_121".into(),
            max_spills: Some(0),
            ptxas: None,
            run_ptxas: true,
            overrides: None,
            temp: None,
        };
        let spill_rep = check_source("spill_k.cu", Some("spill_k"), spill_src, &strict).unwrap();
        assert!(!spill_rep.ok(), "{:?}", spill_rep.kernels);
        assert_eq!(spill_rep.spilling(), 1, "{:?}", spill_rep.kernels);
        let stats = spill_rep.kernels[0].stats.unwrap();
        assert!(stats.spill_bytes() > 0, "{stats:?}");
    }
}
