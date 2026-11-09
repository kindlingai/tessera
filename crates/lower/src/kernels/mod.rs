//! One file per kernel, with the rule proposing it.

mod append;
mod argmax;
mod attend;
mod dense;
mod gather;
mod join;
mod map;
mod matmul;
mod rms_norm;
mod rope;
mod softmax;
mod thin;

pub use append::Append;
pub use argmax::{Argmax, Peaks};
pub use attend::Attend;
pub use dense::{Causal, DENSE_SCORES, Scores, Values};
pub use gather::Gather;
pub use join::Join;
pub use map::{Map, MapStage};
pub use matmul::Matmul;
pub use rms_norm::RmsNorm;
pub use rope::{Angles, Rope};
pub use softmax::Softmax;
pub use thin::{THIN_ROWS, Thin};
