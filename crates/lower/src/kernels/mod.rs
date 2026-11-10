//! The kernels lowering itself launches: maps, copies and appends.

mod append;
mod map;

pub use append::Append;
pub use map::{Map, MapStage};
