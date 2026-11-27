//! CLI regression tests for `check-kernels`.

use crate::check::{finish_outcome, report_json, run};
use tessera_engine::check::{CheckReport, KernelResult};

fn sample_kernel(error: Option<&str>) -> KernelResult {
    KernelResult {
        entry: "demo_k".into(),
        rule: "file".into(),
        kind: "file",
        dynamic_shared: 0,
        threads: 32,
        ptx_bytes: 64,
        stats: None,
        error: error.map(String::from),
    }
}

fn empty_report() -> CheckReport {
    CheckReport {
        arch: "sm_121a".into(),
        nvrtc: Some("13.1".into()),
        ptxas: None,
        kernels: Vec::new(),
    }
}

fn run_args(args: &[&str]) -> Result<(), crate::source::Stop> {
    let owned: Vec<String> = args.iter().map(|s| (*s).into()).collect();
    run(None, &owned)
}

fn assert_file_checks() {
    if !cfg!(feature = "cuda") {
        return;
    }
    let dir = tessera_testkit::scratch(&format!("cli-check-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let good = dir.join("good.cu");
    let bad = dir.join("bad.cu");
    let empty_tu = dir.join("empty.cu");
    std::fs::write(
        &good,
        "extern \"C\" __global__ void ok_k(float* y) { y[0] = 1.0f; }\n",
    )
    .unwrap();
    std::fs::write(
        &bad,
        "extern \"C\" __global__ void bad_k(float* y) { y[0] = nope; }\n",
    )
    .unwrap();
    std::fs::write(
        &empty_tu,
        "__device__ int helper(int x) { return x + 1; }\n",
    )
    .unwrap();
    let g = good.to_string_lossy().into_owned();
    let b = bad.to_string_lossy().into_owned();
    let e = empty_tu.to_string_lossy().into_owned();
    let ok = run_args(&["--file", &g, "--entry", "ok_k", "--no-ptxas", "--json"]);
    if ok.is_ok() {
        assert!(run_args(&["--file", &b, "--no-ptxas"]).is_err());
        assert!(run_args(&["--file", &e, "--no-ptxas"]).is_err());
        assert!(run_args(&["--file", &g, "--entry", "missing", "--no-ptxas"]).is_err());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn agrees_on_outcomes() {
    let empty = empty_report();
    let json = report_json(&empty);
    assert!(!empty.ok());
    assert_eq!(json["ok"], false);
    assert_eq!(json["total"], 0);
    assert!(finish_outcome(&empty).is_err());

    let pass = CheckReport {
        kernels: vec![sample_kernel(None)],
        ..empty_report()
    };
    assert!(pass.ok() && report_json(&pass)["ok"] == true);
    assert!(finish_outcome(&pass).is_ok());

    let fail = CheckReport {
        kernels: vec![sample_kernel(Some("compile failed"))],
        ..empty_report()
    };
    assert!(!fail.ok() && report_json(&fail)["ok"] == false);
    assert!(finish_outcome(&fail).is_err());
}

#[test]
fn rejects_bad_inputs() {
    let missing = tessera_testkit::scratch("cli-check-no-such.cu");
    let path = missing.to_string_lossy().into_owned();
    assert!(run_args(&["--file", &path, "--no-ptxas"]).is_err());
    assert!(run_args(&["--file", &path, "--no-ptxas", "--max-spills", "0"]).is_err());
    assert!(run_args(&["--file", &path, "--arch", "sm_12é", "--no-ptxas"]).is_err());
    assert_file_checks();
}
