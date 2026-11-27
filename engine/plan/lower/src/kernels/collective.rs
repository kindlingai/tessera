//! A sum or a gather every rank takes part in.

use crate::Kernel;
use crate::kernels::Map;
use tessera_graph::DType;

/// Sums `len` partials across ranks. The runner's transport does it.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = crate,
    name = "all_reduce",
    sizes = [],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
    wire = self.wire,
)]
pub struct AllReduce {
    pub len: usize,
    /// Each rank's part rounds to this before the add.
    pub wire: DType,
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

/// Sums `total` partials across ranks, keeping `len` at `start`.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = crate,
    name = "reduce_scatter",
    sizes = [self.start, self.total],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.total,
    wire = self.wire,
)]
pub struct ReduceScatter {
    pub start: usize,
    pub total: usize,
    /// Each rank's part rounds to this before the add.
    pub wire: DType,
}

impl ReduceScatter {
    /// Its kernel name. Runners find reduce-scatters by it.
    pub const NAME: &'static str = "reduce_scatter";
}

/// Joins each rank's contiguous rows into `len` values.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = crate,
    name = "gather_rows",
    sizes = [self.start],
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    work = self.len,
)]
pub struct GatherRows {
    pub len: usize,
    pub start: usize,
}

impl GatherRows {
    /// Its kernel name. Runners find row gathers by it.
    pub const NAME: &'static str = "gather_rows";
}

/// Bytes a crossing step moves, at any wire dtype.
pub fn moved(kernel: &dyn Kernel, len: usize, stored: usize) -> usize {
    kernel.wire().map_or(stored, |w| w.bytes() * len)
}

/// Whether a step of kernel `name` crosses ranks.
pub fn crosses(name: &str) -> bool {
    matches!(
        name,
        AllReduce::NAME | AllGather::NAME | ReduceScatter::NAME | GatherRows::NAME
    )
}
