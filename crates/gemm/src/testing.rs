//! Test support: backend availability and the f32 oracle comparison.

/// Whether GPU tests should run here.
pub fn gpu_available() -> bool {
    if cfg!(target_os = "macos") {
        // Hosted macOS CI runners have no usable Metal device.
        return std::env::var_os("CI").is_none();
    }
    gpukit::testing::cuda_ready()
}

pub use gpukit::testing::{assert_oracle, cosine_f32, max_abs_diff};

/// Tests one shape: a finite oracle, and GPU matching it.
#[macro_export]
macro_rules! gemm_case {
    (
        mod $mod_name:ident,
        fixture = $fixture_fn:path,
        max_tol = $max_tol:expr,
        min_cos = $min_cos:expr $(,)?
    ) => {
        mod $mod_name {
            #[test]
            fn cpu_is_finite() {
                let call = $fixture_fn();
                let out = $crate::cpu(&call);
                assert_eq!(out.len(), $crate::out_len(&call));
                assert!(
                    out.iter().all(|v| v.is_finite()),
                    "cpu oracle produced non-finite values"
                );
            }

            #[test]
            fn gpu_matches_cpu() {
                if !$crate::testing::gpu_available() {
                    return;
                }
                let call = $fixture_fn();
                let cpu = $crate::cpu(&call);
                let gpu = $crate::gpu(&call).expect("gpu dispatch");
                $crate::testing::assert_oracle(&gpu, &cpu, $max_tol, $min_cos);
            }
        }
    };
}
