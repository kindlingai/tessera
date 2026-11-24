//! Traced tensors: each method appends one node.

use crate::model::{Kept, Shared, push};
use crate::weights::{Embed, Linear, Norm, Scalar, Taps, Vector};
use tessera_graph::{
    Act, Conv, DType, DimExpr, Mask, NodeId, Op, Pairs, Row, Select, Shape, Stage, TensorType, View,
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

    /// Each channel over its last rows, through `w`'s taps.
    pub fn conv(&self, w: &Taps) -> Tensor {
        self.conv_dilated(w, 1)
    }

    /// Dilated causal filter over whole sequences.
    pub fn conv_dilated(&self, w: &Taps, dilation: u32) -> Tensor {
        let op = Op::Conv(Conv {
            dilation,
            state: None,
        });
        self.op(op, &[&w.0])
    }

    /// Ragged dilated causal filter updating kept state per slot.
    pub fn conv_rows(
        &self,
        w: &Taps,
        dilation: u32,
        state: &Kept,
        [slots, positions]: [&Tensor; 2],
    ) -> Tensor {
        let op = Op::Conv(Conv {
            dilation,
            state: Some(state.0),
        });
        self.op(op, &[&w.0, slots, positions])
    }

    /// Logits over a table's rows, for tied embeddings.
    pub fn unembed(&self, table: &Embed) -> Tensor {
        self.op(Op::Matmul, &[&table.0])
    }

    /// Each row times `w[.., N, K]`. Batch axes match.
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

    pub fn rope(&self, positions: &Tensor, theta: f32, pairs: Pairs) -> Tensor {
        let turned = None;
        self.stage(
            Stage::Rope {
                theta,
                pairs,
                turned,
            },
            &[positions],
        )
    }

    /// The same, with only the first `turned` pairs rotating.
    pub fn rope_part(&self, positions: &Tensor, theta: f32, pairs: Pairs, turned: u32) -> Tensor {
        let turned = Some(turned);
        self.stage(
            Stage::Rope {
                theta,
                pairs,
                turned,
            },
            &[positions],
        )
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

    /// Rows of this tensor at `ids`, like an embedding lookup.
    pub fn gather(&self, ids: &Tensor) -> Tensor {
        self.op(Op::Select(Select::Gather), &[ids])
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

    /// The sum over `axis`, each slice scaled by `weights[.., axis]`.
    pub fn mix(&self, axis: u32, weights: &Tensor) -> Tensor {
        self.op(Op::Mix(axis), &[weights])
    }

    pub fn cast(&self, dtype: DType) -> Tensor {
        self.op(Op::Cast(dtype), &[])
    }
}

/// Builders for rows with `as`: operand first, then parameters.
macro_rules! builders {
    ($(
        $(#[$m:meta])*
        $name:ident
        $(($tb:ident: $tty:ty = $ts:expr))?
        $({$($(#[$fm:meta])* $f:ident: $fty:ty = $fs:expr),* $(,)?})?
        $(as $method:ident)?
        => $lower:tt {$($col:tt)*}
    )*) => {
        impl Tensor {$(
            builder! {
                [$($method)?] [$(#[$m])*] $name
                [$($tb: $tty)? $($($f: $fty),*)?]
                [$(($tb))? $({$($f),*})?]
                [$($col)*]
            }
        )*}
    };
}

/// Finds the row's operand column, then writes the builder.
macro_rules! builder {
    ([] $($skip:tt)*) => {};
    ($head:tt $docs:tt $name:ident $params:tt $ctor:tt [operand: $kind:ident $(, $($rest:tt)*)?]) => {
        builder!(@emit $head $docs $name $params $ctor [$kind]);
    };
    ($head:tt $docs:tt $name:ident $params:tt $ctor:tt [$col:ident: $val:expr $(, $($rest:tt)*)?]) => {
        builder!($head $docs $name $params $ctor [$($($rest)*)?]);
    };
    ($head:tt $docs:tt $name:ident $params:tt $ctor:tt []) => {
        builder!(@emit $head $docs $name $params $ctor []);
    };
    (@emit [$method:ident] [$(#[$m:meta])*] $name:ident [$($p:ident: $pt:ty),*] [$($ctor:tt)*] [$($kind:ident)?]) => {
        $(#[$m])*
        pub fn $method(&self, $(operand: &builder!(@type $kind),)? $($p: $pt),*) -> Tensor {
            self.stage(Stage::$name $($ctor)*, &[$(builder!(@arg $kind operand))?])
        }
    };
    (@type RowWeight) => { Norm };
    (@type $kind:ident) => { Tensor };
    (@arg RowWeight $o:ident) => { &$o.0 };
    (@arg $kind:ident $o:ident) => { $o };
}

tessera_graph::stages!(builders);
