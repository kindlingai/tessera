use crate::round::Acc;
use graph::DType;

/// An accumulator's sum of 2048 and three ones.
const SUMS: [(DType, f32); 3] = [
    (DType::F32, 2051.0),
    (DType::F16, 2048.0),
    (DType::Bf16, 2048.0),
];

fn sums(&(dtype, want): &(DType, f32)) {
    let xs = [2048.0, 1.0, 1.0, 1.0];
    assert_eq!(
        Acc::new(dtype).unwrap().sum(xs.into_iter()),
        want,
        "{dtype:?}"
    );
}

#[test]
fn sums_round() {
    SUMS.iter().for_each(sums);
    assert!(Acc::new(DType::I32).is_err());
}
