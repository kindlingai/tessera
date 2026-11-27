use crate::promote::{Decayed, Dense, Forecast, Moves, Moving, Paged, Promote, Swap};
use tessera_lower::kernels::Map;
use tessera_lower::{KernelRef, Loaded, Plan, Slot, Step};

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

/// One step picks `picks` rows of four experts.
fn picking(picks: usize) -> Plan {
    let weight = Slot {
        dims: 3,
        shape: [4, 8, 16, 0],
        ..Slot::floats(0, 4 * 8 * 16)
    };
    let step = Step {
        kernel: KernelRef::new(Map::plain(Vec::new())),
        rule: "test",
        site: None,
        reads: vec![weight],
        write: Slot::floats(1, picks * 8),
        also: Vec::new(),
        covers: Vec::new(),
        zone: 0,
        lane: 0,
        table: None,
        index_table: None,
    };
    let loaded = Loaded {
        weight: 7,
        slot: weight,
        share: None,
        part: None,
    };
    Plan {
        steps: vec![step],
        weights: vec![loaded],
        ..Plan::default()
    }
}

#[test]
fn dense_stages_reads() {
    let mut forecast = Dense::default();
    let slots = [Some(0), None, Some(1), None];
    assert_eq!(forecast.cold(&picking(1000), 7, &slots), [1, 3]);
    assert!(forecast.cold(&picking(2), 7, &slots).is_empty());
    assert!(forecast.cold(&picking(1000), 8, &slots).is_empty());
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
