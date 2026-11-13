//! The kernels lowering itself launches: maps, copies and appends.

mod append;
mod map;
mod moves;
mod widen;

pub use append::Append;
pub use map::{Map, MapStage, operands, stage_lines, stage_stmts};
pub use moves::Move;
pub use widen::Widen;
