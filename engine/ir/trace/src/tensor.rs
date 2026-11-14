//! Traced tensors: each method appends one node.

use crate::model::{Kept, Shared, push};
use crate::weights::{Embed, Linear, Norm, Scalar, Vector};
use graph::{
    Act, DType, DimExpr, Mask, NodeId, Op, Pairs, Row, Select, Shape, Stage, TensorType, View,
};

#[derive(Clone)]
pub struct Tensor {
    pub(crate) id: NodeId,
    pub(crate) ty: TensorType,
    pub(crate) g: Shared,
}

impl Tensor {
    pub fn ty(&self) -> &TensorType {
        &self.ty
    }

    fn op(&self, op: Op, rest: &[&Tensor]) -> Tensor {
        let mut args = vec![self];
        args.extend_from_slice(rest);
        push(&self.g, op, &args)
    }

    fn stage(&self, stage: Stage, rest: &[&Tensor]) -> Tensor {
        self.op(Op::Row(Row(vec![stage])), rest)
    }

    pub fn matmul(&self, w: &Linear) -> Tensor {
        self.op(Op::Matmul, &[&w.0])
    }

    /// Logits over a table's rows, for tied embeddings.
    pub fn unembed(&self, table: &Embed) -> Tensor {
        self.op(Op::Matmul, &[&table.0])
    }

    /// Each row times `w[.., N, K]`, whose batch axes match.
    pub fn matmul_rows(&self, w: &Tensor) -> Tensor {
        self.op(Op::Matmul, &[w])
    }

    pub fn add_vector(&self, v: &Vector) -> Tensor {
        self.stage(Stage::Add, &[&v.0])
    }

    pub fn mul_vector(&self, v: &Vector) -> Tensor {
        self.stage(Stage::Mul, &[&v.0])
    }

    pub fn mul_scalar(&self, s: &Scalar) -> Tensor {
        self.stage(Stage::Mul, &[&s.0])
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        self.stage(Stage::Add, &[other])
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.stage(Stage::Mul, &[other])
    }

    pub fn scale(&self, by: f32) -> Tensor {
        self.stage(Stage::Scale(by), &[])
    }

    pub fn rms_norm(&self, w: &Norm, eps: f32) -> Tensor {
        self.stage(Stage::RmsNorm { eps }, &[&w.0])
    }

    /// The row over its root mean square, with no weight.
    pub fn rms(&self, eps: f32) -> Tensor {
        self.stage(Stage::Rms { eps }, &[])
    }

    /// The scaled Walsh-Hadamard transform of each `block` of the row.
    pub fn hadamard(&self, block: u32) -> Tensor {
        self.stage(Stage::Hadamard { block }, &[])
    }

    pub fn center(&self) -> Tensor {
        self.stage(Stage::Center, &[])
    }

    pub fn unit_sum(&self) -> Tensor {
        self.stage(Stage::UnitSum, &[])
    }

    pub fn rope(&self, positions: &Tensor, theta: f32, pairs: Pairs) -> Tensor {
        self.stage(Stage::Rope { theta, pairs }, &[positions])
    }

    pub fn act(&self, act: Act) -> Tensor {
        self.stage(Stage::Act(act), &[])
    }

    pub fn softmax(&self) -> Tensor {
        self.stage(Stage::Softmax, &[])
    }

    pub fn reshape(&self, shape: impl Into<Shape>) -> Tensor {
        self.op(Op::View(View::Reshape(shape.into())), &[])
    }

    pub fn slice(&self, axis: u32, start: impl Into<DimExpr>, len: impl Into<DimExpr>) -> Tensor {
        let (start, len) = (start.into(), len.into());
        self.op(Op::View(View::Slice { axis, start, len }), &[])
    }

    pub fn permute(&self, order: &[u32]) -> Tensor {
        self.op(Op::View(View::Permute(order.to_vec())), &[])
    }

    pub fn expand(&self, axis: u32, to: impl Into<DimExpr>) -> Tensor {
        let to = to.into();
        self.op(Op::View(View::Expand { axis, to }), &[])
    }

    pub fn concat(&self, other: &Tensor) -> Tensor {
        self.op(Op::View(View::Concat), &[other])
    }

    pub fn mask(&self, mask: Mask) -> Tensor {
        self.op(Op::Mask(mask), &[])
    }

    pub fn top_k(&self, k: impl Into<DimExpr>) -> Tensor {
        self.op(Op::Select(Select::TopK { k: k.into() }), &[])
    }

    pub fn take(&self, ids: &Tensor) -> Tensor {
        self.op(Op::Select(Select::Take), &[ids])
    }

    /// Writes rows by slot and position. Returns the kept rows.
    pub fn append(&self, kept: &Kept, [slots, positions]: [&Tensor; 2]) -> Tensor {
        self.op(Op::Append(kept.0), &[slots, positions])
    }

    /// The mean over `axis`. The axis goes away.
    pub fn mean(&self, axis: u32) -> Tensor {
        self.op(Op::Mean(axis), &[])
    }

    pub fn cast(&self, dtype: DType) -> Tensor {
        self.op(Op::Cast(dtype), &[])
    }
}
