use crate::paging::{Decayed, Dense, Forecast, Moves, Moving, Paged, Promote, Swap};

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

#[test]
fn dense_stages_reads() {
    let mut forecast = Dense::default();
    assert!(!forecast.stages(7));
    // A prefill plan reads every cold expert.
    forecast.observe(7, 1.0);
    assert!(forecast.stages(7));
    // A decode plan reads few, and stays unstaged.
    forecast.observe(8, 0.2);
    assert!(!forecast.stages(8));
    // One light launch keeps a staged plan staged.
    forecast.observe(7, 0.3);
    assert!(forecast.stages(7));
    forecast.observe(7, 0.0);
    assert!(!forecast.stages(7));
}

#[test]
fn moves_hold_slots() {
    let mut moves = Moves::default();
    let mut slots = [Some(4), None, None];
    let swap = Swap {
        weight: 1,
        cold: 2,
        hot: 0,
    };
    assert_eq!(moves.begin(&mut slots, &swap), Some(4));
    // Neither expert owns the slot while bytes copy.
    assert_eq!(slots, [None, None, None]);
    assert_eq!(moves.begin(&mut slots, &swap), None);
    let moving = Moving {
        weight: 1,
        cold: 2,
        slot: 4,
    };
    assert_eq!(moves.land(), [moving]);
    assert!(moves.land().is_empty());
}
