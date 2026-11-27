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

fn assert_fresh_cli() {
    let dir = tessera_testkit::scratch("cli-fresh-flat");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("helper.cuh"),
        "#pragma once\n#include <tessera/1/core>\nstatic __device__ tessera::u32 bump(tessera::u32 x) { return x + 1u; }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("trans_k.cu"),
        "#include \"helper.cuh\"\nextern \"C\" __global__ void trans_k(tessera::u32* y) { y[0] = bump(y[0]); }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("direct_k.cu"),
        "#include <tessera/1/core>\nextern \"C\" __global__ void direct_k(tessera::u32* y) { y[0] += 1u; }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("bad_imp.cu"),
        "#include <tessera/1/no_such_hdr>\nextern \"C\" __global__ void bad_imp(int* y) { y[0] = 0; }\n",
    )
    .unwrap();
    let exe = std::env::current_exe().expect("test exe");
    let out = std::process::Command::new(exe)
        .args(["--exact", "check_test::fresh_cli_child", "--ignored"])
        .output()
        .expect("spawn child");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        out.status.success(),
        "fresh CLI child failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

fn run_fresh_case(cu: &std::path::Path, entry: &str, expect_missing: bool) {
    let args: Vec<String> = [
        "check-kernels",
        "--file",
        cu.to_str().unwrap(),
        "--entry",
        entry,
        "--no-ptxas",
        "--json",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let res = crate::run(&args);
    let settings = tessera_frontend_settings::read();
    let cfg = tessera_engine::check::CheckConfig {
        run_ptxas: false,
        temp: settings.temp,
        ..tessera_engine::check::CheckConfig::default()
    };
    let rep = tessera_engine::check::check_file(cu, Some(entry), &cfg).unwrap();
    let err = rep.kernels[0].error.as_deref().unwrap_or("");
    assert!(!err.contains("no flattener is registered"), "{err}");
    if expect_missing {
        assert!(res.is_err() && err.contains("no_such_hdr"), "{err}");
    } else if rep.nvrtc.is_some() {
        assert!(res.is_ok() && rep.ok(), "{err}");
    }
}

#[test]
#[ignore = "spawned by rejects_bad_inputs for a fresh process"]
fn fresh_cli_child() {
    let dir = tessera_testkit::scratch("cli-fresh-flat");
    run_fresh_case(&dir.join("trans_k.cu"), "trans_k", false);
    run_fresh_case(&dir.join("direct_k.cu"), "direct_k", false);
    run_fresh_case(&dir.join("bad_imp.cu"), "bad_imp", true);
}

#[test]
fn rejects_bad_inputs() {
    let missing = tessera_testkit::scratch("cli-check-no-such.cu");
    let path = missing.to_string_lossy().into_owned();
    assert!(run_args(&["--file", &path, "--no-ptxas"]).is_err());
    assert!(run_args(&["--file", &path, "--no-ptxas", "--max-spills", "0"]).is_err());
    assert!(run_args(&["--file", &path, "--arch", "sm_12é", "--no-ptxas"]).is_err());
    assert_file_checks();
    assert_fresh_cli();
}
