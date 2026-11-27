//! The model graph: typed nodes, symbolic shapes, a JSON form.

mod attend;
mod chain;
mod dims;
mod infer;
mod json;
mod node;
mod numerics;
mod ops;
mod own;
mod parts;
mod regions;
mod scalar;
mod select;
mod shard;
mod split;
mod stages;
mod trim;
mod types;
mod verify;

pub use chain::layers;
pub use dims::{Dim, DimExpr, DimName, Shape, Size};
pub use infer::{ErrorKind, TypeError, infer};
pub use json::{FormatError, VERSION};
pub use node::{
    Act, Attend, CalibTap, Composite, Conv, Delta, Entry, EntryKind, Graph, Mask, Node, NodeId, Op,
    Output, Pairs, PathSeg, Port, Region, Role, Row, Select, View, Weight,
};
pub use numerics::{Math, Numerics, Order, sums_ranks};
pub use own::{own_rows, own_rows_attn};
pub use parts::Parts;
pub use scalar::{Bf16, E2M1, E4M3, E8M0, F16};
pub use shard::{Sharded, Shares, Whole, shard, shard_aligned, shard_mode};
pub use split::{HALF_ROWS, split_batch};
pub use stages::{Lower, Pin, STAGES, Stage, Takes};
pub use trim::trim;
pub use types::{DType, Placement, TensorType};
pub use verify::{VerifyError, verify};

#[cfg(test)]
mod dims_test;
#[cfg(test)]
mod json_test;
#[cfg(test)]
mod regions_test;
#[cfg(test)]
mod rules_test;
#[cfg(test)]
mod scalar_test;
#[cfg(test)]
mod shard_test;
#[cfg(test)]
mod verify_test;
