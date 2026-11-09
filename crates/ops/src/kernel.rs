//! `op_kernel!`: `gpukit::kernel!` bound to this crate's error and helpers.

#[macro_export]
macro_rules! op_kernel {
    ($($rest:tt)*) => {
        gpukit::kernel! {
            op,
            error = $crate::Error,
            assert_oracle = $crate::testing::assert_oracle,
            gpu_available = $crate::testing::gpu_available,
            $($rest)*
        }
    };
}
