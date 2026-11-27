use crate::tiering::EVICT_HOT;
use crate::{Aged, EVICT_COLD, EVICT_PINNED, Room, Tiering};

#[test]
fn aged_demotes_cold() {
    let mut policy = Aged::default();
    for page in 0..4 {
        policy.placed(page);
    }
    let level = |p: u32| match p {
        0 => Some(EVICT_HOT),
        3 => None,
        _ => Some(EVICT_COLD),
    };
    assert_eq!(policy.victims(1, 2, &level), [1, 2]);
    assert_eq!(policy.victims(1, 1, &level), [0], "hot goes last");
    let room = Room {
        need: 0,
        vram_free: 1,
        host_free: 0,
    };
    let host = [(1, EVICT_COLD), (2, EVICT_HOT), (0, EVICT_PINNED)];
    assert_eq!(policy.promotes(room, &level, &host), [2]);
}
