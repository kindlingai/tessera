//! Training ops: backward passes and AdamW.

// Fixtures are never empty, so `is_empty` would mean nothing.
#![allow(clippy::len_without_is_empty)]

pub mod adamw;
pub mod gelu_backward;
pub mod rms_norm_backward;
pub mod scatter_add_rows;
pub mod softmax_backward;
