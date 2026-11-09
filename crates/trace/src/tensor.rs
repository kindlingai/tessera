//! Traced tensors: each method appends one node.

use crate::model::{Shared, push};
use crate::weights::{Linear, Norm};
use graph::{Act, DType, DimExpr, NodeId, Op, Row, Shape, Stage, TensorType, View};

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

    pub fn cast(&self, dtype: DType) -> Tensor {
        self.op(Op::Cast(dtype), &[])
    }
}
