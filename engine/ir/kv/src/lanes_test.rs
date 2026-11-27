use crate::tiering::EVICT_HOT;
use crate::{Aged, EVICT_COLD, Error, Lanes};

const HOST: u32 = 1 << 30;

#[test]
fn tracks_dirty_ranges() {
    let mut leader = Lanes::new(16, 16, 2, 12);
    let mut follower = Lanes::new(16, 16, 2, 12);
    let init = leader.take_dirty();
    assert_eq!(init.len(), 2);
    assert!(leader.pending().is_empty());
    let _ = follower.take_dirty();

    leader.set(1, &[0, 0, 7]);
    let patches = leader.pending();
    assert_eq!(patches, vec![(10, &[7][..])]);
    let owned: Vec<(u32, Vec<u32>)> = patches.iter().map(|&(o, v)| (o, v.to_vec())).collect();
    follower.apply(&owned);
    assert_eq!(leader.take_dirty(), vec![(40, 7u32.to_le_bytes().to_vec())]);
    assert_eq!(
        follower.take_dirty(),
        vec![(40, 7u32.to_le_bytes().to_vec())]
    );

    leader.set(1, &[0, 0, 7]);
    assert!(leader.pending().is_empty());
    assert!(leader.take_dirty().is_empty());

    leader.reset();
    assert_eq!(leader.pending(), vec![(10, &[0][..])]);
}

/// A split table with 4 VRAM and 16 host pages.
fn tiered() -> Lanes {
    let mut lanes = Lanes::new_split(16, 16, 2, 4, 16);
    lanes.enable_tier(16, Box::new(Aged::default()));
    lanes
}

/// Places `alloc`, maps `lane` to slot 0, returns its entries.
fn place(lanes: &mut Lanes, alloc: &[u32], priority: &[u8], lane: &[u32]) -> Vec<u32> {
    let mut moves = |_, _: &[(u32, u32)]| Ok::<bool, Error>(true);
    lanes.ensure_vram(alloc, &[], priority, &mut moves).unwrap();
    lanes.set(0, lane);
    let (offset, values) = lanes.pending()[0];
    assert_eq!(offset, 0);
    values[..lane.len()].to_vec()
}

#[test]
fn tier_demotes_cold_first() {
    let mut lanes = tiered();
    let mut priority = vec![EVICT_COLD; lanes.pages()];
    priority[0] = EVICT_HOT;
    place(&mut lanes, &[0, 1, 2, 3], &priority, &[0, 1, 2, 3]);
    let lane = place(&mut lanes, &[4], &priority, &[0, 1, 2, 3, 4]);
    assert_eq!(lane[0] & HOST, 0, "hot page stays in VRAM");
    assert!(lane[1..4].iter().all(|&e| e & HOST != 0));
    assert_eq!(lane[4] & HOST, 0, "new page lands in VRAM");

    priority[1] = EVICT_HOT;
    let lane = place(&mut lanes, &[], &priority, &[0, 1, 2, 3, 4]);
    assert_eq!(lane[1] & HOST, 0, "hot host page promotes");
    assert!(lane[2..4].iter().all(|&e| e & HOST != 0));
}
