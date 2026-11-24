use crate::paged::{Layout, desc_bytes, is_pageable};
use tessera_lower::form::{F32_FORM, named};
use tessera_lower::{Loaded, Plan, Slot};

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
        outputs: vec![tessera_lower::Output {
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
    let exp_slot = rewritten
        .weights
        .iter()
        .find(|l| l.weight == 2)
        .unwrap()
        .slot;
    assert_eq!(rewritten.buffers[exp_slot.buffer as usize], desc_bytes(8));
    assert_eq!(exp_slot.offset, 0);
}
