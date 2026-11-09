//! Elementwise maths, softmax and RMS norm.

// Fixtures are never empty, so `is_empty` would mean nothing.
#![allow(clippy::len_without_is_empty)]

pub mod fill_zero;
pub mod gelu;
pub mod rms_norm;
pub mod rms_norm_rows;
pub mod softmax;
pub mod swiglu_gelu;
pub mod vec_add;
pub mod vec_scale;
