/// The invariant that must hold regardless of backend: rows sum to 1.
#[test]
fn rows_sum_to_one() {
    if !ops::testing::gpu_available() {
        return;
    }
    let fix = crate::softmax::wide_fixture();
    let out = crate::softmax::gpu(&fix).expect("gpu dispatch");
    for r in 0..fix.rows {
        let sum: f32 = out[r * fix.cols..(r + 1) * fix.cols].iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "row {r} sums to {sum}");
    }
}
