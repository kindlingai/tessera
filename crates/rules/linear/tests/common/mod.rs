//! Shared graphs and inputs for the matmul tests.

use gpukit::testing::Noise;
use graph::DimName::{Ffn, Hidden, Rows};
use interp::Tensor;
use trace::{DType, Dim, EntryKind, Graph, Linear, Model, Port, Role, Size, TensorType, Weights};

#[derive(Weights)]
struct Layer {
    #[shape(Hidden, Ffn)]
    w: Linear,
}

/// A graph's work on its input and the product.
pub type Finish = fn(&trace::Tensor, trace::Tensor) -> trace::Tensor;

/// `k` inputs to `n` outputs, over any number of rows.
pub fn linear(k: u64, n: u64) -> Graph {
    finished(k, n, |_, product| product)
}

/// The same, with stages after the matmul.
pub fn finished(k: u64, n: u64, finish: Finish) -> Graph {
    let fixed = |name, n| Dim {
        name,
        size: Size::Fixed(n),
    };
    let size = Size::Bounded(256);
    let rows = Dim { name: Rows, size };
    let m = Model::new(vec![rows, fixed(Hidden, k), fixed(Ffn, n)]);
    let w: Layer = trace::declare(&m);
    let ty = TensorType::new([Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let [x] = m.entry(EntryKind::Forward, [Port { role, ty }]);
    m.finish([(Role::Logits, &finish(&x, x.matmul(&w.w)))])
}

pub fn noise(shape: Vec<usize>, noise: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
    Tensor::f32(shape, data)
}
