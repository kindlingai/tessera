//! Runs kernel bodies on the host, each group in lockstep.

pub mod attend;
pub mod exec;
mod intrinsic {
    include!("../../codegen/src/intrinsic.rs");
}
pub mod lockstep;
#[cfg(target_arch = "aarch64")]
mod neon;
mod simd;

#[cfg(test)]
mod exec_test;
#[cfg(test)]
mod lockstep_test;
#[cfg(test)]
mod simd_test;
