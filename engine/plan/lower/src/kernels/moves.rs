//! A value crossing from one zone to another.

use crate::Kernel;
use crate::kernels::Map;

/// Copies `len` elements into the step's zone.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = crate,
    name = "move",
    sizes = [],
    // The runner copies, so nothing compiles.
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
)]
pub struct Move {
    pub len: usize,
}
