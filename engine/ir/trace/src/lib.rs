//! Builds graphs by running model descriptions.

// The derive names `::trace`, also inside this crate.
extern crate self as tessera_trace;

mod model;
mod tensor;
mod weights;

pub use model::{Kept, Model};
pub use tensor::Tensor;
pub use tessera_graph::{
    Act, Composite, DType, Dim, DimExpr, DimName, EntryKind, Graph, Mask, Pairs, PathSeg, Port,
    Role, Shape, Size, TensorType,
};
pub use tessera_trace_derive::Weights;
pub use weights::{Embed, Linear, Linears, Norm, Scalar, Taps, Vector, Weights, declare, each};

#[cfg(test)]
mod trace_test;
