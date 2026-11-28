use tessera_kv::{Oldest, Pool, Prefixes, Tag};

#[test]
fn longest_newest_wins() {
    let (mut pool, mut index) = (
        Pool::new(8, 2, Box::new(Oldest::default())),
        Prefixes::new(2),
    );
    let mut runs = Vec::new();
    for history in [vec![1, 2, 3, 4], vec![1, 2, 3, 4, 5], vec![1, 2, 3, 4, 5]] {
        let run = pool.reserve(Tag(0), 4, 4).unwrap();
        pool.retain(run).unwrap();
        index.keep(run, history, 4);
        runs.push(run);
    }
    let prompt = [1, 2, 3, 4, 5, 6];
    assert_eq!(index.find(&pool, Tag(0), &prompt, 5), Some((runs[2], 4)));
    assert_eq!(index.find(&pool, Tag(1), &prompt, 5), None);
    pool.release(runs[2]).unwrap();
    assert_eq!(index.find(&pool, Tag(0), &prompt, 3), Some((runs[1], 2)));
}

#[test]
fn trie_matches_naive() {
    use tessera_kv::begin;

    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || -> u32 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as u32
    };
    for grid in [1usize, 4, 16] {
        let mut pool = Pool::new(256, 4, Box::new(Oldest::default()));
        let mut index = Prefixes::new(grid);
        let mut naive: Vec<(tessera_kv::RunId, Vec<u32>, usize)> = Vec::new();
        let base: Vec<u32> = (0..64).map(|i| i % 5).collect();
        for step in 0..40 {
            let mut h = base[..(next() as usize % 48)].to_vec();
            for _ in 0..(next() as usize % 24) {
                h.push(next() % 4);
            }
            let valid = (next() as usize % (h.len() + 1)).max(1);
            let tag = Tag((next() % 3).into());
            if let Ok(run) = pool.reserve(tag, 4, 4) {
                pool.retain(run).unwrap();
                index.keep(run, h.clone(), valid);
                naive.push((run, h, valid));
            }
            if step % 5 == 4 && !naive.is_empty() {
                let drop_at = next() as usize % naive.len();
                let _ = pool.release(naive[drop_at].0);
            }
            for _ in 0..8 {
                let mut q = base[..(next() as usize % 48)].to_vec();
                for _ in 0..(next() as usize % 24) {
                    q.push(next() % 4);
                }
                let limit = next() as usize % 70;
                let tag = Tag((next() % 3).into());
                naive.retain(|(r, _, _)| pool.rows(*r).is_ok());
                let mut want = None;
                for (r, h, v) in &naive {
                    if pool.tag(*r) != Ok(tag) {
                        continue;
                    }
                    let skip = begin(h, *v, &q, limit, grid).skip;
                    if skip > 0 && want.is_none_or(|(_, most)| skip >= most) {
                        want = Some((*r, skip));
                    }
                }
                assert_eq!(index.find(&pool, tag, &q, limit), want);
            }
        }
        // High-fanout prefix exercises Small->Many edge promotion and lazy pruning.
        let mut fan = Vec::new();
        for b in 0u32..16 {
            let mut h = vec![99u32; grid];
            h.extend(std::iter::repeat_n(1000 + b, grid * 2));
            if let Ok(run) = pool.reserve(Tag(u64::from(b % 2)), 4, 4) {
                pool.retain(run).unwrap();
                index.keep(run, h.clone(), h.len());
                fan.push((run, h, b));
            }
        }
        for &(run, _, b) in &fan {
            if b % 3 == 0 {
                let _ = pool.release(run);
            }
        }
        for &(run, ref h, b) in &fan {
            let got = index.find(&pool, Tag(u64::from(b % 2)), h, h.len());
            if b % 3 != 0 {
                assert_eq!(got, Some((run, h.len())));
            } else {
                assert!(got.is_none_or(|(r, skip)| r != run && skip == grid));
            }
        }
    }
}
