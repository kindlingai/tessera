//! The kernels lowering itself launches: maps, copies, sums and appends.

use crate::Kernel;

mod collective;
mod join;
mod map;
mod scatter;
mod stage;
mod widen;

pub use collective::{AllGather, AllReduce, GatherRows, ReduceScatter, crosses, moved};
pub use join::Join;
pub use map::{Map, MapStage, add_operand_forms, add_operands, finish, hash_stages, operands};
pub(crate) use scatter::Scatter;
pub use stage::{Batches, Ring, Stage, place};
pub(crate) use widen::Narrow;
pub use widen::Widen;

/// Copies a value into the step's zone.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = crate,
    name = "move",
    sizes = [],
    // The runner copies, so nothing compiles.
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
)]
pub(crate) struct Move;
