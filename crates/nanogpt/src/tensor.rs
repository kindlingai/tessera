//! Thin wrappers over the GPU ops and their CPU oracles.

use gemm::{Call, Problem, gpu as gemm_gpu};
use ops::Error;

/// C = alpha * op(A) @ op(B) + beta * C, with explicit leading dimensions.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    a: &[f32],
    b: &[f32],
    c: Vec<f32>,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    trans_a: bool,
    trans_b: bool,
    beta: f32,
) -> Result<Vec<f32>, Error> {
    let call = Call {
        a: a.to_vec(),
        b: b.to_vec(),
        c,
        problem: Problem {
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
        },
    };
    gemm_gpu(&call)
}

/// Y = X @ W^T, with W in PyTorch linear layout (n,k).
pub fn linear(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>, Error> {
    gemm(x, w, vec![0.0; m * n], m, n, k, k, k, n, false, true, 0.0)
}

pub fn rms_norm(
    x: &[f32],
    weight: &[f32],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<Vec<f32>, Error> {
    ops_math::rms_norm::gpu(&ops_math::rms_norm::Fixture {
        x: x.to_vec(),
        weight: weight.to_vec(),
        rows,
        hidden,
        eps,
    })
}

pub fn gelu(x: &[f32]) -> Result<Vec<f32>, Error> {
    ops_math::gelu::gpu(&ops_math::gelu::Fixture { x: x.to_vec() })
}

pub fn vec_add(x: &[f32], addend: &[f32]) -> Result<Vec<f32>, Error> {
    ops_math::vec_add::gpu(&ops_math::vec_add::Fixture {
        x: x.to_vec(),
        addend: addend.to_vec(),
    })
}

pub fn softmax(x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, Error> {
    ops_math::softmax::gpu(&ops_math::softmax::Fixture {
        logits: x.to_vec(),
        rows,
        cols,
    })
}

pub fn gather_rows(src: &[f32], indices: &[u32], hidden: usize) -> Result<Vec<f32>, Error> {
    ops_model::gather_rows::gpu(&ops_model::gather_rows::Fixture {
        src: src.to_vec(),
        indices: indices.to_vec(),
        hidden,
    })
}

// CPU oracle versions, for the reference forward pass.

/// C = alpha * op(A) @ op(B) + beta * C on the CPU oracle.
#[allow(clippy::too_many_arguments)]
pub fn gemm_cpu(
    a: &[f32],
    b: &[f32],
    c: Vec<f32>,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    trans_a: bool,
    trans_b: bool,
    beta: f32,
) -> Vec<f32> {
    let call = Call {
        a: a.to_vec(),
        b: b.to_vec(),
        c,
        problem: Problem {
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
        },
    };
    gemm::cpu(&call)
}

/// Y = X @ W^T on the CPU oracle.
pub fn linear_cpu(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    gemm_cpu(x, w, vec![0.0; m * n], m, n, k, k, k, n, false, true, 0.0)
}

pub fn rms_norm_cpu(x: &[f32], weight: &[f32], rows: usize, hidden: usize, eps: f32) -> Vec<f32> {
    ops_math::rms_norm::cpu(&ops_math::rms_norm::Fixture {
        x: x.to_vec(),
        weight: weight.to_vec(),
        rows,
        hidden,
        eps,
    })
}

pub fn gelu_cpu(x: &[f32]) -> Vec<f32> {
    ops_math::gelu::cpu(&ops_math::gelu::Fixture { x: x.to_vec() })
}

pub fn vec_add_cpu(x: &[f32], addend: &[f32]) -> Vec<f32> {
    ops_math::vec_add::cpu(&ops_math::vec_add::Fixture {
        x: x.to_vec(),
        addend: addend.to_vec(),
    })
}

pub fn softmax_cpu(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    ops_math::softmax::cpu(&ops_math::softmax::Fixture {
        logits: x.to_vec(),
        rows,
        cols,
    })
}

pub fn gather_rows_cpu(src: &[f32], indices: &[u32], hidden: usize) -> Vec<f32> {
    ops_model::gather_rows::cpu(&ops_model::gather_rows::Fixture {
        src: src.to_vec(),
        indices: indices.to_vec(),
        hidden,
    })
}
