//! The model graph: typed nodes, symbolic shapes, a JSON form.

mod dims;
mod infer;
mod json;
mod node;
mod types;
mod verify;

pub use dims::{Dim, DimExpr, DimName, Poly, Shape, Size};
pub use infer::{ErrorKind, TypeError, infer};
pub use json::{FormatError, VERSION};
pub use node::{
    Act, Attend, Entry, EntryKind, Graph, Mask, Node, NodeId, Op, Output, PathSeg, Port, Role, Row,
    Select, Stage, View, Weight,
};
pub use types::{DType, Placement, TensorType};
pub use verify::{VerifyError, verify};

#[cfg(test)]
mod dims_test;
#[cfg(test)]
mod json_test;
#[cfg(test)]
mod rules_test;
#[cfg(test)]
mod verify_test;
