/// The defining invariant: every row's ds sums to zero.
#[test]
fn rows_sum_to_zero() {
    if !ops::testing::gpu_available() {
        return;
    }
    let fix = crate::softmax_backward::wide_fixture();
    let out = crate::softmax_backward::gpu(&fix).expect("gpu dispatch");
    for r in 0..fix.rows {
        let sum: f32 = out[r * fix.cols..(r + 1) * fix.cols].iter().sum();
        assert!(sum.abs() < 1e-5, "row {r} sums to {sum}");
    }
}
