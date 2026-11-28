//! Shared graphs and inputs for the matmul tests.

use std::collections::HashMap;
use tessera_codegen::ir::Param;
use tessera_graph::DimName::Rows;
use tessera_interp::Tensor;
use tessera_lower::{Plan, Source, Terms};
use tessera_testkit::Noise;
pub use tessera_testkit::graphs::Finish;
use tessera_trace::{Dim, Graph};

/// `k` inputs to `n` outputs, over any number of rows.
pub fn linear(k: u64, n: u64) -> Graph {
    finished(k, n, |_, product| product)
}

/// The same, with stages after the matmul.
pub fn finished(k: u64, n: u64, finish: Finish) -> Graph {
    tessera_testkit::graphs::linear_then(&[Dim::bounded(Rows, 640)], [k, n], finish)
}

pub fn noise(shape: Vec<usize>, noise: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
    Tensor::f32(shape, data)
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
        let Param::Scalar { ty, .. } = param else {
            unreachable!()
        };
        let value = match ty {
            tessera_codegen::ir::Ty::F32 => arg.f32().map(f64::from),
            _ => arg.u32().map(f64::from),
        };
        let value = value.unwrap_or_else(|| panic!("{arg:?} is not a scalar"));
        params.insert(name.clone(), value);
    }
    tessera_lower::terms(&body, &params).expect("every loop bound is a parameter")
}

/// Every weight form whose blocks fit rows of `k`.
pub fn forms(k: usize) -> Vec<&'static dyn tessera_lower::form::Load> {
    let fits = |f: &&&dyn tessera_lower::form::Load| k.is_multiple_of(f.align());
    tessera_lower::form::LOADS
        .iter()
        .filter(fits)
        .copied()
        .filter(|f| f.dtype().is_float() && f.packs())
        .collect()
}
