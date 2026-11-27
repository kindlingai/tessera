//! Helpers that tests in every crate share.

mod compare;
mod fill;
mod net;
mod noise;
mod reference;
mod scratch;
mod tools;

pub use compare::{argmax, cosine_f32, max_abs_diff};
pub use fill::{fill, weights, weights_with};
pub mod graphs;
pub use net::{TCP_TIMEOUT, tcp_bind, tcp_connect, tcp_pair, tcp_timeouts};
pub use noise::Noise;
pub use reference::{reference, references};
pub use scratch::{scratch, stage};
pub use tools::{run, toolkit};
