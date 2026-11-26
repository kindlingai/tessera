//! How reducing ops round and order their sums.

use crate::node::{Graph, Op};
use crate::stages::Stage;
use crate::types::DType;
use serde::{Deserialize, Serialize};

/// Tracing sets the strongest values. Passes can lower, never raise.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Numerics {
    /// The accumulator rounds to this after each add.
    pub accumulate: DType,
    pub order: Order,
    pub math: Math,
    /// Each rank's part rounds to this before a cross-rank add.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wire: Option<DType>,
}

impl Default for Numerics {
    fn default() -> Self {
        Self {
            accumulate: DType::F32,
            order: Order::Fixed,
            math: Math::Exact,
            wire: None,
        }
    }
}

impl Numerics {
    /// Exact sums whose parts cross ranks as `wire`.
    pub fn pinned(wire: DType) -> Self {
        Self {
            wire: Some(wire),
            ..Self::default()
        }
    }
}

/// Whether `op` sums parts across ranks.
pub fn sums_ranks(op: &Op) -> bool {
    let sums = |s: &Stage| matches!(s, Stage::AllReduce | Stage::ReduceScatter);
    matches!(op, Op::Row(r) if r.0.iter().any(sums))
}

impl Graph {
    /// Gives every cross-rank sum the model left unpinned `wire`.
    pub fn wired(mut self, wire: DType) -> Self {
        for n in &mut self.nodes {
            if let (true, Some(num)) = (sums_ranks(&n.op), &mut n.numerics) {
                num.wire.get_or_insert(wire);
            }
        }
        self
    }
}

/// Which reduction orders a kernel may use, strongest first.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Order {
    /// Same bits for a row in any batch size.
    Fixed,
    /// Deterministic for one row bucket and configuration.
    PerConfig,
    /// Any order, including atomics.
    Any,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Math {
    Exact,
    /// Approximate transcendentals and fused multiply-adds allowed.
    Approx,
}
