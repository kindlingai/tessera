//! Test shapes, each sized exactly to what the kernel reads.

use crate::problem::{Call, Problem};

fn build(m: usize, n: usize, k: usize, trans_a: bool, trans_b: bool, beta: f32) -> Call {
    let lda = if trans_a { m } else { k };
    let ldb = if trans_b { k } else { n };
    let ldc = n;
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
    let a: Vec<f32> = (0..problem.a_len())
        .map(|i| ((i as f32) * 0.013).sin() * 0.25)
        .collect();
    let b: Vec<f32> = (0..problem.b_len())
        .map(|i| ((i as f32) * 0.007).cos() * 0.02)
        .collect();
    let c: Vec<f32> = (0..problem.c_len())
        .map(|i| ((i as f32) * 0.021).sin() * 0.1)
        .collect();
    Call { a, b, c, problem }
}

/// Small square case, no transposes, beta = 0.
pub fn tiny() -> Call {
    build(3, 4, 5, false, false, 0.0)
}

/// Linear layer: W is (n,k), so trans_b.
pub fn linear() -> Call {
    build(8, 16, 32, false, true, 0.0)
}

/// A is (k,m), B is (k,n).
pub fn trans_a() -> Call {
    build(6, 5, 7, true, false, 0.0)
}

/// Both transposes, with a nonzero C accumulated at beta.
pub fn both_trans() -> Call {
    build(9, 7, 11, true, true, 0.5)
}

/// m, n and k each end in a partial tile.
pub fn tile() -> Call {
    build(
        crate::BM + 13,
        crate::BN + 7,
        3 * crate::BK + 5,
        false,
        true,
        0.0,
    )
}
