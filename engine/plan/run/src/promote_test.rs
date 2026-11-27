use crate::promote::{Decayed, Paged, Promote, Swap};

#[test]
fn decayed_swaps_expert() {
    let mut policy = Decayed::default();
    let slots = [Some(0), None, None];
    let paged = |ready| {
        [Paged {
            slots: &slots,
            stride: 8,
            ready,
        }]
    };
    policy.observe(0, &[1, 0, 9]);
    let swap = Swap {
        weight: 0,
        cold: 2,
        hot: 0,
    };
    assert_eq!(policy.swaps(&paged(false)), []);
    assert_eq!(policy.swaps(&paged(true)), [swap]);
    assert_eq!(policy.swaps(&paged(true)), [], "odd steps wait");
    policy.bytes = 4;
    assert_eq!(policy.swaps(&paged(true)), []);
}
