use crate::marks::{ROOM, read, slots};
use crate::profile::Marks;
use tessera_lower::kernels::{Map, Stage};
use tessera_lower::{Plan, Slot};

#[test]
fn copies_follow_span() {
    let at = Slot::floats(0, 4);
    let copy = |batch| Stage { bytes: 8, batch }.step(at, Vec::new());
    let mut step = copy(0);
    step.kernel = tessera_lower::KernelRef::new(Map::plain(Vec::new()));
    let plan = Plan {
        steps: vec![step; 3],
        staged: vec![(1, copy(0)), (3, copy(1))],
        ..Plan::default()
    };
    let found = |range| -> Vec<_> {
        let all = slots(&plan, range).into_iter();
        all.map(|(slot, i, step)| (slot, i, step.kernel.staged()))
            .collect()
    };
    assert_eq!(
        found(0..3),
        [
            (0, Some(0), None),
            (4, None, Some(0)),
            (1, Some(1), None),
            (2, Some(2), None),
            (5, None, Some(1)),
        ]
    );
    assert_eq!(found(1..2), [(2, None, Some(0)), (0, Some(1), None)]);
}

#[test]
fn gaps_between_marks() {
    let mut records = vec![0u64; 2 * ROOM];
    records[..6].copy_from_slice(&[10, 100, 20, 400, 10, 1000]);
    let mut marks = Marks::default();
    read(&[ROOM as u32 + 2], &records, 1e9, &mut marks);
    let (first, second) = (&marks.regions[0], &marks.regions[1]);
    assert_eq!(
        (first.from, first.to, first.seconds.as_slice()),
        (10, 20, &[300e-9][..])
    );
    assert_eq!((second.from, second.to), (20, 10));
    assert_eq!(marks.lost, 2);
}

#[test]
fn unified_budget_full() {
    use tessera_engine_settings::Memory::{Bytes, Device, Off};
    let mem = 16u64 << 30;
    let mut p = tessera_gpukit::Platform {
        memory: mem,
        ..Default::default()
    };
    let b = |m, p: &_| crate::Base::new(m, p).budget;
    assert_eq!(b(Device, &p), Some(mem - mem / 10));
    p.integrated = true;
    assert_eq!(
        (b(Device, &p), b(Off, &p), b(Bytes(1), &p)),
        (Some(mem), None, Some(1))
    );
}
