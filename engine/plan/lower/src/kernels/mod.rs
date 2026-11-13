//! The kernels lowering itself launches: maps, copies and appends.

mod append;
mod map;
mod moves;
mod scatter;
mod widen;

pub use append::Append;
pub use map::{Map, MapStage, operands, stage_stmts};
pub use moves::Move;
pub use scatter::Scatter;
pub use widen::Widen;
