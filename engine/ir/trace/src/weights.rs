//! Weight handles and the trait `#[derive(Weights)]` implements.

use crate::model::Model;
use crate::tensor::Tensor;
use sensai_graph::{DType, DimExpr, DimName, PathSeg, Shape, TensorType};

/// Declares weights under `path`, shaped by the field's `#[shape]`.
pub trait Weights: Sized {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, shape: &[DimExpr]) -> Self;
}

/// Declares a whole weight tree from the root.
pub fn declare<W: Weights>(m: &Model) -> W {
    W::declare(m, &mut Vec::new(), &[])
}

/// `count` items under `name`, each shaped by `shape(index)`.
pub fn each<T: Weights>(
    m: &Model,
    name: &str,
    count: DimName,
    shape: impl Fn(u32) -> Vec<DimExpr>,
) -> Vec<T> {
    let mut path = vec![PathSeg::Field(name.into())];
    (0..m.size(count) as u32)
        .map(|i| {
            path.push(PathSeg::Index(i));
            let item = T::declare(m, &mut path, &shape(i));
            path.pop();
            item
        })
        .collect()
}

/// `#[shape(In, Out)]`, stored `[Out, In]` as checkpoints do.
pub struct Linear(pub(crate) Tensor);

/// `#[shape(Width)]`.
pub struct Norm(pub(crate) Tensor);

/// `#[shape(Width)]`: added to rows.
pub struct Vector(pub(crate) Tensor);

/// `#[shape()]`: one value, for every element.
pub struct Scalar(pub(crate) Tensor);

/// `#[shape(Rows, Width)]`: a table read by index.
pub struct Embed(pub(crate) Tensor);

/// `#[shape(Count, In, Out)]`, stored `[Count, Out, In]`.
pub struct Linears(pub(crate) Tensor);

impl Linears {
    /// Output rows `start..start + len` of every matrix.
    pub fn rows(&self, start: impl Into<DimExpr>, len: impl Into<DimExpr>) -> Linears {
        Linears(self.0.slice(1, start, len))
    }
}

/// A field a model leaves out. It declares nothing.
impl Weights for () {
    fn declare(_: &Model, _: &mut Vec<PathSeg>, _: &[DimExpr]) -> Self {}
}

fn leaf(m: &Model, path: &[PathSeg], shape: Vec<DimExpr>, want: usize) -> Tensor {
    assert_eq!(
        shape.len(),
        want,
        "{path:?} needs {want} dimensions in #[shape]"
    );
    m.weight(path, TensorType::new(Shape(shape), DType::F32))
}

impl Linear {
    /// A matrix computed from weights, stored `[out, in]`.
    pub fn of(t: Tensor) -> Self {
        Linear(t)
    }
}

impl Norm {
    /// Norm gains a model computes from its weights.
    pub fn of(t: Tensor) -> Self {
        Norm(t)
    }
}

impl Vector {
    /// A row a model computes from its weights.
    pub fn of(t: Tensor) -> Self {
        Vector(t)
    }
}

impl Weights for Linear {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, shape: &[DimExpr]) -> Self {
        let swapped = shape.iter().rev().cloned().collect();
        Linear(leaf(m, path, swapped, 2))
    }
}

impl Weights for Norm {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, shape: &[DimExpr]) -> Self {
        Norm(leaf(m, path, shape.to_vec(), 1))
    }
}

impl Weights for Vector {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, shape: &[DimExpr]) -> Self {
        Vector(leaf(m, path, shape.to_vec(), 1))
    }
}

impl Weights for Scalar {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, shape: &[DimExpr]) -> Self {
        Scalar(leaf(m, path, shape.to_vec(), 0))
    }
}

impl Weights for Linears {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, shape: &[DimExpr]) -> Self {
        let mut stored = shape.to_vec();
        stored.reverse();
        stored.rotate_right(1);
        Linears(leaf(m, path, stored, 3))
    }
}

impl Weights for Embed {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, shape: &[DimExpr]) -> Self {
        Embed(leaf(m, path, shape.to_vec(), 2))
    }
}

/// The first `#[shape]` entry is a fixed count, like `Layers`.
impl<T: Weights> Weights for Vec<T> {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, shape: &[DimExpr]) -> Self {
        let Some((DimExpr::Dim(count), rest)) = shape.split_first() else {
            panic!("{path:?}: a Vec field needs #[shape(Count, ..)]");
        };
        (0..m.size(*count) as u32)
            .map(|i| {
                path.push(PathSeg::Index(i));
                let item = T::declare(m, path, rest);
                path.pop();
                item
            })
            .collect()
    }
}
