//! The GEMM problem and the buffers it runs on.

/// A row-major GEMM shape with explicit leading dimensions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Problem {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// Each leading dimension is at least its stored row width.
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
    pub alpha: f32,
    pub beta: f32,
    /// A is (k,m) when set, else (m,k).
    pub trans_a: bool,
    /// B is (n,k) when set, else (k,n).
    pub trans_b: bool,
}

impl Problem {
    /// C = A @ B^T, with B in PyTorch linear layout (n,k).
    pub fn linear(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            lda: k,
            ldb: k,
            ldc: n,
            alpha: 1.0,
            beta: 0.0,
            trans_a: false,
            trans_b: true,
        }
    }

    /// C = A @ B with A (m,k) and B (k,n).
    pub fn matmul(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            lda: k,
            ldb: n,
            ldc: n,
            alpha: 1.0,
            beta: 0.0,
            trans_a: false,
            trans_b: false,
        }
    }

    /// Elements A occupies under this layout.
    pub fn a_len(&self) -> usize {
        if self.trans_a {
            self.k * self.lda
        } else {
            self.m * self.lda
        }
    }

    /// Elements B occupies under this layout.
    pub fn b_len(&self) -> usize {
        if self.trans_b {
            self.n * self.ldb
        } else {
            self.k * self.ldb
        }
    }

    /// Elements C occupies under this layout.
    pub fn c_len(&self) -> usize {
        self.m * self.ldc
    }

    /// Elements the result occupies (m * n, ignoring the leading dimension).
    pub fn out_len(&self) -> usize {
        self.m * self.n
    }
}

/// Kernel arguments, laid out as in gemm.metal and gemm.cu.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct AbiParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lda: u32,
    pub ldb: u32,
    pub ldc: u32,
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: u32,
    pub trans_b: u32,
}

impl From<Problem> for AbiParams {
    fn from(p: Problem) -> Self {
        Self {
            m: p.m as u32,
            n: p.n as u32,
            k: p.k as u32,
            lda: p.lda as u32,
            ldb: p.ldb as u32,
            ldc: p.ldc as u32,
            alpha: p.alpha,
            beta: p.beta,
            trans_a: u32::from(p.trans_a),
            trans_b: u32::from(p.trans_b),
        }
    }
}

/// One GEMM call: operand buffers and their problem.
#[derive(Debug, Clone)]
pub struct Call {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub problem: Problem,
}

impl Call {
    /// Buffers must already match `Problem::a_len`, `b_len` and `c_len`.
    pub fn new(a: Vec<f32>, b: Vec<f32>, c: Vec<f32>, problem: Problem) -> Self {
        Self { a, b, c, problem }
    }

    pub fn out_len(&self) -> usize {
        self.problem.out_len()
    }
}

pub fn out_len(call: &Call) -> usize {
    call.out_len()
}
