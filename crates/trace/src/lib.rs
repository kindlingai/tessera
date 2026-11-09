//! Builds graphs by running model descriptions.

// The derive names `::trace`, also inside this crate.
extern crate self as trace;

mod model;
mod tensor;
mod weights;

pub use graph::{
    Act, Attend, Composite, DType, Dim, DimExpr, DimName, EntryKind, Graph, Mask, PathSeg, Port,
    Role, Shape, Size, TensorType,
};
pub use model::Model;
pub use tensor::Tensor;
pub use trace_derive::Weights;
pub use weights::{Embed, Linear, Norm, Weights, declare};

#[cfg(test)]
mod trace_test;
