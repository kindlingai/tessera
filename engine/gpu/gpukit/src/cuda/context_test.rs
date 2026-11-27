use super::context::compute_arch;
use super::{compile_ptx, nvrtc_version};

#[test]
fn normalizes_cuda_targets() {
    assert_eq!(compute_arch("sm_80").unwrap(), "compute_80");
    assert_eq!(compute_arch("sm_89").unwrap(), "compute_89");
    assert_eq!(compute_arch("sm_90a").unwrap(), "compute_90a");
    assert_eq!(compute_arch("sm_120").unwrap(), "compute_120a");
    assert_eq!(compute_arch("sm_121").unwrap(), "compute_121a");
    assert_eq!(compute_arch("compute_121a").unwrap(), "compute_121a");
    assert_eq!(compute_arch("12.1").unwrap(), "compute_121a");
    assert!(compute_arch("").is_err());
    assert!(compute_arch("sm_bad").is_err());
    assert!(compute_arch("0.1").is_err());
}

#[test]
fn compiles_ptx_offline() {
    if nvrtc_version().is_none() {
        return;
    }
    let good = "extern \"C\" __global__ void ok_kernel(float* y) { y[0] = 1.0f; }\n";
    let ptx = compile_ptx("sm_121", "ok_kernel", good).unwrap();
    let text = String::from_utf8_lossy(&ptx);
    assert!(text.contains(".entry ok_kernel"), "{text}");

    let bad = "extern \"C\" __global__ void bad_kernel(float* y) {\n    y[0] = missing_sym;\n}\n";
    let err = compile_ptx("sm_121", "bad_kernel", bad)
        .unwrap_err()
        .to_string();
    assert!(err.contains("bad_kernel.cu"), "{err}");
    assert!(err.contains("(2)"), "{err}");
}
