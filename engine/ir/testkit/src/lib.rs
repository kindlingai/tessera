//! Helpers that tests in every crate share.

mod compare;
mod fill;
mod noise;
mod reference;
mod scratch;

pub use compare::{argmax, cosine_f32, max_abs_diff};
pub use fill::{fill, weights, weights_with};
pub mod graphs;
pub use noise::Noise;
pub use reference::{reference, references};
pub use scratch::{scratch, stage};
