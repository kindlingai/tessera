//! The model graph: typed nodes, symbolic shapes, a JSON form.

mod dims;
mod infer;
mod json;
mod node;
mod numerics;
mod regions;
mod types;
mod verify;

pub use dims::{Dim, DimExpr, DimName, Poly, Shape, Size};
pub use infer::{ErrorKind, TypeError, infer};
pub use json::{FormatError, VERSION};
pub use node::{
    Act, Attend, Composite, Entry, EntryKind, Graph, Mask, Node, NodeId, Op, Output, PathSeg, Port,
    Region, Role, Row, Select, Stage, View, Weight,
};
pub use numerics::{Math, Numerics, Order};
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
mod verify_test;
