//! A sum or a gather every rank takes part in.

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

/// Joins each rank's part of `rows` rows into `len` values.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = crate,
    name = "all_gather",
    sizes = [self.rows],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
)]
pub struct AllGather {
    pub len: usize,
    pub rows: usize,
}

impl AllGather {
    /// Its kernel name. Runners find gathers by it.
    pub const NAME: &'static str = "all_gather";
}

/// Whether a step of kernel `name` crosses ranks.
pub fn crosses(name: &str) -> bool {
    name == AllReduce::NAME || name == AllGather::NAME
}
