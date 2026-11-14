//! Helpers that tests in every crate share.

mod compare;
mod fill;
mod noise;
mod scratch;

pub use compare::{argmax, assert_oracle, cosine_f32, max_abs_diff};
pub use fill::{fill, weights, weights_with};
pub use noise::Noise;
pub use scratch::{scratch, stage};
