macro_rules! case {
    ($name:ident, $fixture:path) => {
        crate::gemm_case! {
            mod $name,
            fixture = $fixture,
            max_tol = 1e-4,
            min_cos = 0.9999,
        }
    };
}

case!(tiny, crate::fixtures::tiny);
case!(linear, crate::fixtures::linear);
case!(trans_a, crate::fixtures::trans_a);
case!(both_trans, crate::fixtures::both_trans);
case!(tile, crate::fixtures::tile);
