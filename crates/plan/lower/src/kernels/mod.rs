//! The kernels lowering itself launches: maps, copies and appends.

mod append;
mod map;
mod moves;

pub use append::Append;
pub use map::{Map, MapStage, operands, stage_lines};
pub use moves::Move;
