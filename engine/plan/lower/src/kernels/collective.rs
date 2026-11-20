//! A sum every rank takes part in.

use crate::Kernel;
use crate::kernels::Map;

/// Sums `len` partials across ranks. The runner's transport does it.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = crate,
    name = "all_reduce",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
)]
pub struct AllReduce {
    pub len: usize,
}

impl AllReduce {
    /// Its kernel name. Runners find sums by it.
    pub const NAME: &'static str = "all_reduce";
}
