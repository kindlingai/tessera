//! How reducing ops round and order their sums.

use crate::types::DType;
use serde::{Deserialize, Serialize};

/// Tracing sets the strongest values. Passes may only weaken them.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Numerics {
    /// The accumulator rounds to this after each add.
    pub accumulate: DType,
    pub order: Order,
    pub math: Math,
}

impl Default for Numerics {
    fn default() -> Self {
        Self {
            accumulate: DType::F32,
            order: Order::Fixed,
            math: Math::Exact,
        }
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
