use crate::lanes::LaneSync;
use tessera_lower::kernels::{Batches, Map, Ring, place};
use tessera_lower::{KernelRef, Plan, Slot, Step};

fn reader(reads: Vec<Slot>, write: Slot) -> Step {
    Step {
        kernel: KernelRef::new(Map::plain(Vec::new())),
        rule: "test",
        site: None,
        reads,
        write,
        also: Vec::new(),
        covers: Vec::new(),
        zone: 0,
        lane: 0,
        table: None,
        index_table: None,
    }
}

/// Three weights read in turn. Their slots overlap.
fn chain() -> (Plan, Vec<Option<(Slot, usize)>>) {
    let weight = |i: usize| Slot {
        offset: i * 256,
        ..Slot::floats(9, 1000)
    };
    let temp = |i: u32| Slot::floats(i, 4);
    let steps = vec![
        reader(vec![weight(0)], temp(0)),
        reader(vec![weight(1), temp(0)], temp(1)),
        reader(vec![weight(2), temp(1)], temp(2)),
    ];
    let plan = Plan {
        steps,
        ..Plan::default()
    };
    (plan, (0..3).map(|i| Some((weight(i), 64))).collect())
}

#[test]
fn copies_reuse_ring() {
    let (mut plan, weights) = chain();
    let ring = Ring { parts: 2, slots: 1 };
    let mut batches = Batches::default();
    place(&mut plan, &weights, ring, &mut batches, false, |_| {
        vec![3, 5]
    });
    let order = plan.sequence(0..3);
    let kinds: Vec<_> = order.iter().map(|(_, s)| s.kernel.staged()).collect();
    // Weight 2 reuses weight 0's part once its reader ran.
    let expect = [
        Some(0),
        Some(1),
        None,
        Some(2),
        None,
        None,
        Some(3),
        Some(4),
    ];
    assert_eq!(kinds, expect);
    assert_eq!(plan.steps.len(), 3);
    assert_eq!(batches.list()[2].experts, vec![3]);
    assert_eq!(batches.list()[2].evicts, Some(0));
    assert_eq!(order[3].1.also, vec![weights[0].unwrap().0]);
    let sync = LaneSync::plan(order.iter().map(|(_, s)| *s), false);
    assert_eq!(sync.waits[3], vec![sync.records[2].unwrap()]);
    assert_eq!(sync.waits[5], vec![sync.records[3].unwrap()]);
    // The same plan asks for the same batches again.
    let (mut again, _) = chain();
    place(&mut again, &weights, ring, &mut batches, false, |_| {
        vec![3, 5]
    });
    assert_eq!(batches.list().len(), 5);
}

#[test]
fn empty_forecast_adds_nothing() {
    let (mut plan, weights) = chain();
    let ring = Ring { parts: 2, slots: 1 };
    place(
        &mut plan,
        &weights,
        ring,
        &mut Batches::default(),
        false,
        |_| Vec::new(),
    );
    assert!(plan.staged.is_empty());
}
