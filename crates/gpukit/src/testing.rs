//! Oracle comparison helpers shared by every crate's tests.

/// Largest absolute element-wise difference. Panics if the lengths differ.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Cosine similarity, or 1.0 when both vectors are zero.
pub fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    // Rounding can push an exact match just past 1.0.
    (dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0)
}

/// Assert `a` is finite and matches `b` within both tolerances.
pub fn assert_oracle(a: &[f32], b: &[f32], max_tol: f32, min_cos: f32) {
    let max = max_abs_diff(a, b);
    // Identical output must pass `min_cos = 1.0` despite cosine rounding.
    let cos = if max == 0.0 { 1.0 } else { cosine_f32(a, b) };
    assert!(
        max <= max_tol && cos >= min_cos,
        "oracle mismatch max_abs={max:.6} cos={cos:.6} (tol={max_tol}, min_cos={min_cos})"
    );
    assert!(
        a.iter().all(|v| v.is_finite()),
        "output has non-finite values"
    );
}

/// Whether a CUDA driver loads on this host.
pub fn cuda_ready() -> bool {
    #[cfg(feature = "cuda")]
    return crate::cuda::driver::driver().is_ok();
    #[cfg(not(feature = "cuda"))]
    false
}
