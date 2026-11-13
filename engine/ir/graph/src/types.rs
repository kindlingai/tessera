//! Element types, placements and tensor types.

use crate::dims::Shape;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    Bf16,
    F16,
    I32,
    U32,
    Bool,
}

impl DType {
    /// Bytes one element takes.
    pub fn bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::Bf16 | DType::F16 => 2,
            DType::Bool => 1,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::Bf16 | DType::F16)
    }

    pub fn is_int(self) -> bool {
        matches!(self, DType::I32 | DType::U32)
    }
}

/// How a tensor lies across ranks after sharding.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    Replicated,
    /// Each rank holds a slice of this axis.
    Split {
        axis: u32,
    },
    /// Per-rank partial sums. An all-reduce makes it whole.
    Partial,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TensorType {
    pub shape: Shape,
    pub dtype: DType,
    pub placement: Placement,
}

impl TensorType {
    /// A replicated tensor.
    pub fn new(shape: impl Into<Shape>, dtype: DType) -> Self {
        Self {
            shape: shape.into(),
            dtype,
            placement: Placement::Replicated,
        }
    }

    pub fn placed(mut self, placement: Placement) -> Self {
        self.placement = placement;
        self
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }
}
