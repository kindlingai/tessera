use crate::lanes::LaneSync;
use tessera_graph::DType;
use tessera_lower::kernels::{AllReduce, Map, Stage};
use tessera_lower::{KernelRef, Slot, Step};

fn dummy(lane: u8, reads: Vec<Slot>, write: Slot) -> Step {
    Step {
        kernel: KernelRef::new(Map::plain(Vec::new())),
        rule: "test",
        site: None,
        reads,
        write,
        also: Vec::new(),
        covers: Vec::new(),
        zone: 0,
        lane,
        table: None,
        index_table: None,
    }
}

#[test]
fn ping_pong_events() {
    let a_in = Slot::floats(0, 4);
    let a_mid = Slot::floats(1, 4);
    let a_out = Slot::floats(2, 4);
    let b_in = Slot::floats(3, 4);
    let b_mid = Slot::floats(4, 4);
    let b_out = Slot::floats(5, 4);
    let steps = vec![
        dummy(0, vec![a_in], a_mid),
        dummy(1, vec![a_mid], a_out),
        dummy(0, vec![b_in], b_mid),
        dummy(1, vec![b_mid], b_out),
        dummy(0, vec![a_out], a_in),
    ];
    let sync = LaneSync::plan(&steps, false);
    assert!(sync.active);
    assert!(sync.forks.is_empty());
    assert!(sync.waits[0].is_empty());
    assert_eq!(sync.waits[1], vec![sync.records[0].unwrap()]);
    assert!(sync.waits[2].is_empty());
    assert_eq!(sync.waits[3], vec![sync.records[2].unwrap()]);
    assert_eq!(sync.waits[4], vec![sync.records[1].unwrap()]);
    assert_eq!(sync.joins, vec![sync.records[3].unwrap()]);
}

#[test]
fn leading_comm_forks() {
    let steps = vec![dummy(1, vec![Slot::floats(0, 4)], Slot::floats(1, 4))];
    let sync = LaneSync::plan(&steps, false);
    assert!(sync.active);
    assert_eq!(sync.forks.len(), 1);
    assert_eq!(sync.forks[0].0, 1);
    assert_eq!(sync.joins, vec![sync.records[0].unwrap()]);
}

#[test]
fn collectives_serialize_across_lanes() {
    let ar = |lane, r, w| Step {
        kernel: KernelRef::new(AllReduce {
            len: 4,
            wire: DType::F32,
        }),
        ..dummy(lane, vec![r], w)
    };
    let steps = vec![
        ar(1, Slot::floats(0, 4), Slot::floats(1, 4)),
        ar(0, Slot::floats(2, 4), Slot::floats(3, 4)),
    ];
    let sync = LaneSync::plan(&steps, false);
    assert_eq!(sync.waits[1], vec![sync.records[0].unwrap()]);
    assert!(sync.joins.is_empty());
    // A copy waits for a sum only with turns.
    let copy = Stage { bytes: 8, batch: 0 }.step(Slot::floats(9, 4), Vec::new());
    let steps = vec![ar(1, Slot::floats(0, 4), Slot::floats(1, 4)), copy];
    let waits = |turns| LaneSync::plan(&steps, turns).waits[1].len();
    assert_eq!((waits(false), waits(true)), (0, 1));
}
