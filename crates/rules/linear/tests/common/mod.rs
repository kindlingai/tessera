//! Shared graphs and inputs for the matmul tests.

use codegen::ir::Param;
use engine::settings;
use gpukit::backend::Backend;
use gpukit::testing::Noise;
use graph::DimName::{Ffn, Hidden, Rows};
use interp::Tensor;
use lower::{Arg, Plan, Settings, Source, Terms};
use std::collections::HashMap;
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

/// A GPU run, or each emulation on the host.
pub fn runs() -> Vec<Settings> {
    let s = settings();
    if s.gpu().is_some() {
        return vec![s];
    }
    let emulations = [
        None,
        Some(Backend::Metal),
        Some(Backend::Cuda),
        Some(Backend::Rocm),
    ];
    emulations
        .map(|emulate| Settings {
            emulate,
            ..s.clone()
        })
        .to_vec()
}

/// The sums one output of `plan`'s only step gathers.
pub fn terms(plan: &Plan) -> Terms {
    let step = &plan.steps[0];
    let Source::Made(body) = step.kernel.source() else {
        panic!("a generated kernel");
    };
    let launch = step.kernel.launches(step).remove(0);
    let mut params = HashMap::new();
    for (param, arg) in body.params.iter().zip(&launch.args) {
        let Param::Scalar { name, .. } = param else {
            continue;
        };
        let value = match arg {
            Arg::U32(v) => *v as f64,
            Arg::F32(v) => *v as f64,
            other => panic!("{other:?} is not a scalar"),
        };
        params.insert(name.clone(), value);
    }
    lower::terms(&body, &params).expect("every loop bound is a parameter")
}
