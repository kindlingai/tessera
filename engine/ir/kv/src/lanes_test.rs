use crate::tiering::EVICT_HOT;
use crate::{Aged, EVICT_COLD, EVICT_PINNED, Error, Lanes, Move};

const HOST: u32 = 1 << 30;
const PAGES: usize = 16;

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

/// A split table of `vram` pages, holding `ahead` steps free.
fn tiered(vram: usize, ahead: usize) -> Lanes {
    let mut lanes = Lanes::new_split(16, 2 * PAGES, 2, vram, PAGES);
    let mut policy = Box::new(Aged::default());
    policy.ahead = ahead;
    lanes.enable_tier(PAGES, policy);
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
    let mut lanes = tiered(4, 0);
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

/// Each tier's page contents. Started copies run at the wait.
#[derive(Default)]
struct Memory {
    vram: Vec<u32>,
    host: Vec<u32>,
    flying: Vec<(u32, u32)>,
    departed: usize,
    /// Host pages a promote read in this step.
    promoted: Vec<u32>,
}

impl Memory {
    fn act(&mut self, kind: Move, pairs: &[(u32, u32)]) -> Result<bool, Error> {
        match kind {
            Move::Depart => self.flying.extend_from_slice(pairs),
            Move::Settle => self.departed += self.flying.len(),
            Move::Demote => assert!(self.flying.is_empty()),
            Move::Promote => self.promoted.extend(pairs.iter().map(|p| p.1)),
        }
        let reused = |p: &(u32, u32)| kind == Move::Depart && self.promoted.contains(&p.1);
        assert!(!pairs.iter().any(reused), "a promoted host page waits");
        let copies = match kind {
            Move::Settle => std::mem::take(&mut self.flying),
            Move::Depart => Vec::new(),
            Move::Demote | Move::Promote => pairs.to_vec(),
        };
        for (v, h) in copies {
            match kind {
                Move::Promote => self.vram[v as usize] = self.host[h as usize],
                _ => self.host[h as usize] = self.vram[v as usize],
            }
        }
        Ok(true)
    }

    fn read(&self, entry: u32) -> u32 {
        match entry & HOST {
            0 => self.vram[entry as usize],
            _ => self.host[(entry & !HOST) as usize],
        }
    }
}

/// Writes one page per step, then reads all.
fn evicting(ahead: usize) -> (Vec<u32>, usize) {
    let mut lanes = tiered(4, ahead);
    let mut memory = Memory {
        vram: vec![0; 4],
        host: vec![0; PAGES],
        ..Memory::default()
    };
    let steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 6, 10, 11];
    let mut lane: Vec<u32> = Vec::new();
    for (step, &page) in steps.iter().enumerate() {
        if step == 8 {
            // Step 7 started the copies of both pages.
            assert_eq!(memory.flying.len(), 2 * ahead);
            lanes.release_freed(&[5, 6]);
            lane.retain(|p| ![5, 6].contains(p));
        }
        memory.promoted.clear();
        let mut priority = vec![EVICT_COLD; lanes.pages()];
        // Page 0 turns hot on the host, so it promotes.
        priority[0] = if step < 5 { EVICT_COLD } else { EVICT_HOT };
        priority[page as usize] = EVICT_PINNED;
        let mut moves = |kind, pairs: &[(u32, u32)]| memory.act(kind, pairs);
        lanes
            .ensure_vram(&[page], &[page], &priority, &mut moves)
            .unwrap();
        lane.push(page);
        lanes.set(0, &lane);
        let entries = lanes.pending()[0].1[..lane.len()].to_vec();
        let at = *entries.last().unwrap();
        assert_eq!(at & HOST, 0, "a step writes VRAM");
        let copying = memory.flying.iter().any(|&(v, _)| v == at);
        assert!(!copying, "a copying page is not reused");
        memory.vram[at as usize] = 100 + step as u32;
        let read: Vec<u32> = entries.iter().map(|&e| memory.read(e)).collect();
        let wrote = |p: &u32| 100 + steps[..=step].iter().rposition(|s| s == p).unwrap() as u32;
        let want: Vec<u32> = lane.iter().map(wrote).collect();
        assert_eq!(read[..lane.len() - 1], want[..lane.len() - 1]);
    }
    let entries = lanes.pending()[0].1[..lane.len()].to_vec();
    let read = entries.iter().map(|&e| memory.read(e)).collect();
    (read, memory.departed)
}

#[test]
fn started_demotes_keep_rows() {
    let (blocking, none) = evicting(0);
    let (started, departed) = evicting(1);
    assert_eq!(started, blocking);
    assert_eq!(none, 0);
    assert!(departed > 0, "copies ran beside steps");
}
