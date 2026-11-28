use crate::form::{F32_FORM, named};
use crate::paging::{Decayed, Dense, Forecast, Lent, Moves, Moving, Paged, Promote, Swap};
use crate::paging::{Layout, desc_bytes, is_pageable};
use crate::plan::{Loaded, Plan, Slot};

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

fn slot(buffer: u32, offset: usize, dims: u8, shape: [u32; 4], form: &str) -> Slot {
    let form = named(form).unwrap_or(F32_FORM);
    let len = shape[..dims as usize].iter().map(|&d| d as usize).product();
    Slot {
        buffer,
        offset,
        len,
        form,
        dims,
        shape,
    }
}

#[test]
fn pageable_filter_and_rewrite() {
    let dense = Loaded {
        weight: 0,
        slot: slot(0, 0, 2, [256, 512, 0, 0], "trellis_k4"),
        share: None,
        part: None,
    };
    let rotated = Loaded {
        weight: 1,
        slot: slot(0, 1024, 3, [4, 256, 512, 0], "trellis_k3_rot"),
        share: None,
        part: Some(0),
    };
    let experts = Loaded {
        weight: 2,
        slot: slot(0, 4096, 3, [8, 256, 512, 0], "trellis_k3_mul1"),
        share: None,
        part: None,
    };
    assert!(!is_pageable(&dense));
    assert!(!is_pageable(&rotated));
    assert!(is_pageable(&experts));
    let plan = Plan {
        buffers: vec![4096 + experts.slot.bytes(), 2048, 64, 128],
        zones: vec![0, 0, 0, 0],
        weights: vec![dense, rotated, experts],
        kept: vec![(0, slot(1, 0, 1, [512, 0, 0, 0], "f32"))],
        inputs: vec![slot(2, 0, 1, [16, 0, 0, 0], "f32")],
        outputs: vec![crate::plan::Output {
            slot: slot(3, 0, 1, [32, 0, 0, 0], "f32"),
            shape: vec![32],
        }],
        ..Plan::default()
    };
    let layout = Layout::build(&plan).expect("finds expert weight");
    assert_eq!(layout.specs.len(), 1);
    assert_eq!(layout.by_weight.get(&2), Some(&0));
    assert_eq!(layout.specs[0].experts, 8);
    assert_eq!(layout.desc_total, desc_bytes(8));
    let rewritten = layout.rewrite(&plan);
    let old_total: usize = plan.buffers.iter().sum();
    let new_total: usize = rewritten.buffers.iter().sum();
    assert!(new_total < old_total);
    // Non-weight buffer indices stay fixed for caller slots.
    assert_eq!(rewritten.kept[0].1.buffer, 1);
    assert_eq!(rewritten.inputs[0].buffer, 2);
    assert_eq!(rewritten.outputs[0].slot.buffer, 3);
    assert_eq!(rewritten.buffers[1], 2048);
    // Paged weight moved into the descriptor buffer.
    let moved = |l: &&Loaded| l.weight == 2;
    let exp_slot = rewritten.weights.iter().find(moved).unwrap().slot;
    assert_eq!(rewritten.buffers[exp_slot.buffer as usize], desc_bytes(8));
    assert_eq!(exp_slot.offset, 0);
}

#[test]
fn loans_leave_lowest() {
    let mut lent = Lent::default();
    lent.rebase(100);
    // Slots fill top-down, above the plan's 10 bytes.
    assert_eq!(lent.take(0, 1, 30, 10), Some(70));
    assert_eq!(lent.take(0, 2, 30, 10), Some(40));
    assert_eq!(lent.take(0, 3, 30, 10), Some(10));
    assert_eq!(lent.take(0, 4, 30, 10), None);
    assert_eq!(lent.land().len(), 3);
    // An ended loan's slot goes to the next expert.
    lent.forget(|loan| loan.expert == 2);
    assert_eq!(lent.take(1, 9, 30, 10), Some(40));
    // A plan writing 45 bytes pushes out the lowest two.
    let out: Vec<_> = lent.evict(45).iter().map(|loan| loan.expert).collect();
    assert_eq!(out, [3, 9]);
    let left: Vec<_> = lent.loans().iter().map(|loan| loan.expert).collect();
    assert_eq!(left, [1]);
    assert_eq!(lent.take(0, 5, 30, 45), None);
}
