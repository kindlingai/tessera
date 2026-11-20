//! Builds graphs by running model descriptions.

// The derive names `::trace`, also inside this crate.
extern crate self as sensai_trace;

mod model;
mod tensor;
mod weights;

pub use model::{Kept, Model};
pub use sensai_graph::{
    Act, Composite, DType, Dim, DimExpr, DimName, EntryKind, Graph, Mask, Pairs, PathSeg, Port,
    Role, Shape, Size, TensorType,
};
pub use sensai_trace_derive::Weights;
pub use tensor::Tensor;
pub use weights::{Embed, Linear, Linears, Norm, Scalar, Vector, Weights, declare, each};

#[cfg(test)]
mod trace_test;
