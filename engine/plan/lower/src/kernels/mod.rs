//! The kernels lowering itself launches: maps, copies and appends.

mod map;
mod moves;
mod scatter;
mod widen;

pub use map::{Map, MapStage, add_operands, operands, stage_stmts};
pub use moves::Move;
pub use scatter::Scatter;
pub use widen::Widen;
