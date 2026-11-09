//! Thin wrappers over the GPU ops and their CPU oracles.

use gemm::{Call, Problem};
use ops::Error;

/// The GPU kernels, or their CPU oracles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Gpu,
    Cpu,
}

impl Target {
    fn run<F>(
        self,
        fix: &F,
        gpu: fn(&F) -> Result<Vec<f32>, Error>,
        cpu: fn(&F) -> Vec<f32>,
    ) -> Result<Vec<f32>, Error> {
        match self {
            Target::Gpu => gpu(fix),
            Target::Cpu => Ok(cpu(fix)),
        }
    }
}

/// C = alpha * op(A) @ op(B) + beta * C, with explicit leading dimensions.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    t: Target,
    a: &[f32],
    b: &[f32],
    c: Vec<f32>,
    (m, n, k): (usize, usize, usize),
    (lda, ldb, ldc): (usize, usize, usize),
    (trans_a, trans_b): (bool, bool),
    beta: f32,
) -> Result<Vec<f32>, Error> {
    let problem = Problem {
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        alpha: 1.0,
        beta,
        trans_a,
        trans_b,
    };
    let call = Call {
        a: a.to_vec(),
        b: b.to_vec(),
        c,
        problem,
    };
    t.run(&call, gemm::gpu, gemm::cpu)
}

/// Y = X @ W^T, with W in PyTorch linear layout (n,k).
pub fn linear(
    t: Target,
    x: &[f32],
    w: &[f32],
    (m, n, k): (usize, usize, usize),
) -> Result<Vec<f32>, Error> {
    gemm(
        t,
        x,
        w,
        vec![0.0; m * n],
        (m, n, k),
        (k, k, n),
        (false, true),
        0.0,
    )
}

pub fn rms_norm(
    t: Target,
    x: &[f32],
    weight: &[f32],
    rows: usize,
    eps: f32,
) -> Result<Vec<f32>, Error> {
    use ops_math::rms_norm::{Fixture, cpu, gpu};
    let fix = Fixture {
        x: x.to_vec(),
        weight: weight.to_vec(),
        rows,
        hidden: weight.len(),
        eps,
    };
    t.run(&fix, gpu, cpu)
}

pub fn gelu(t: Target, x: &[f32]) -> Result<Vec<f32>, Error> {
    use ops_math::gelu::{Fixture, cpu, gpu};
    t.run(&Fixture { x: x.to_vec() }, gpu, cpu)
}

pub fn vec_add(t: Target, x: &[f32], addend: &[f32]) -> Result<Vec<f32>, Error> {
    use ops_math::vec_add::{Fixture, cpu, gpu};
    let fix = Fixture {
        x: x.to_vec(),
        addend: addend.to_vec(),
    };
    t.run(&fix, gpu, cpu)
}

pub fn softmax(t: Target, x: &[f32], cols: usize) -> Result<Vec<f32>, Error> {
    use ops_math::softmax::{Fixture, cpu, gpu};
    let fix = Fixture {
        logits: x.to_vec(),
        rows: x.len() / cols,
        cols,
    };
    t.run(&fix, gpu, cpu)
}

pub fn gather_rows(
    t: Target,
    src: &[f32],
    indices: &[u32],
    hidden: usize,
) -> Result<Vec<f32>, Error> {
    use ops_model::gather_rows::{Fixture, cpu, gpu};
    let fix = Fixture {
        src: src.to_vec(),
        indices: indices.to_vec(),
        hidden,
    };
    t.run(&fix, gpu, cpu)
}
