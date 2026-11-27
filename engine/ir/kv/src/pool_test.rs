use crate::{Error, Evict, Oldest, PageCopy, Pool, Tag};

const T: Tag = Tag(0);

#[test]
fn admission_counts_worst() {
    let mut pool = Pool::new(4, 2, Box::new(Oldest::default()));
    let a = pool.reserve(T, 1, 6).unwrap();
    assert_eq!(pool.reserve(T, 1, 4), Err(Error::NoRoom(2)));
    let stats = pool.stats();
    assert_eq!((stats.size, stats.active, stats.worst), (8, 2, 6));
    pool.retain(a).unwrap();
    let stats = pool.stats();
    assert_eq!((stats.live_runs, stats.kept_runs, stats.active), (0, 1, 0));
    pool.release(a).unwrap();
    pool.reserve(T, 1, 8).unwrap();
}

#[test]
fn shared_writes_copy() {
    let mut pool = Pool::new(8, 2, Box::new(Oldest::default()));
    let a = pool.reserve(T, 3, 6).unwrap();
    let b = pool.fork(a, 3, 6).unwrap();
    let [first, last] = pool.pages(a).unwrap()[..] else {
        panic!("two pages")
    };
    let copies = pool.grow(b, 4).unwrap();
    let to = copies[0].to;
    assert_eq!(
        copies,
        [PageCopy {
            from: last,
            to,
            rows: 1
        }]
    );
    assert_eq!(pool.pages(b).unwrap(), [first, to]);
    assert_eq!(pool.grow(a, 4).unwrap(), []);
}

#[test]
fn retained_rows_fork() {
    let mut pool = Pool::new(4, 2, Box::new(Oldest::default()));
    let a = pool.reserve(T, 3, 8).unwrap();
    pool.grow(a, 8).unwrap();
    pool.truncate(a, 3).unwrap();
    pool.retain(a).unwrap();
    let b = pool.fork(a, 3, 4).unwrap();
    assert_eq!(pool.pages(b), pool.pages(a));
    pool.release(b).unwrap();
    pool.reserve(T, 8, 8).unwrap();
    assert_eq!(pool.pages(a), Err(Error::Unknown(a)));
}

#[test]
fn full_pool_adopts() {
    let mut pool = Pool::new(2, 4, Box::new(Oldest::default()));
    let a = pool.reserve(T, 3, 4).unwrap();
    pool.reserve(T, 3, 4).unwrap();
    pool.retain(a).unwrap();
    assert_eq!(pool.fork(a, 2, 4), Err(Error::NoRoom(1)));
    let page = pool.pages(a).unwrap().to_vec();
    let b = pool.adopt(a, 2, 4).unwrap();
    assert_eq!(pool.adopt(b, 2, 4), Err(Error::Live(b)));
    assert_eq!((pool.pages(b).unwrap(), pool.rows(b)), (&page[..], Ok(2)));
    assert_eq!(pool.rows(a), Err(Error::Unknown(a)));
    pool.grow(b, 4).unwrap();
}

/// Random calls on small pools. The pool checks itself.
#[test]
fn random_calls() {
    for (pages, mut seed) in [4, 6, 8, 12]
        .into_iter()
        .flat_map(|n| (1..=4u64).map(move |s| (n, s)))
    {
        let mut pool = Pool::new(pages, 4, Box::new(Oldest::default()));
        let mut runs = Vec::new();
        for _ in 0..1000 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let pick = (seed >> 33) as usize;
            let run = runs.get(pick / 8 % runs.len().max(1)).copied();
            let before = pool.digest();
            let done = match (pick % 8, run) {
                (7, Some(r)) => pool.widen(r, 8 + pick % 17),
                (1, Some(r)) => pool.fork(r, pick % 9, 16).map(|r| runs.push(r)),
                (2, Some(r)) => pool.grow(r, pick % 17).map(drop),
                (3, Some(r)) => pool.truncate(r, pick % 9),
                (4, Some(r)) => pool.retain(r),
                (5, Some(r)) => pool.release(r).map(|()| runs.retain(|o| *o != r)),
                (6, Some(r)) => pool.adopt(r, pick % 9, 16).map(|n| {
                    runs.retain(|o| *o != r);
                    runs.push(n);
                }),
                _ => pool
                    .reserve(T, pick % 9, 8 + pick % 9)
                    .map(|r| runs.push(r)),
            };
            if done.is_err() {
                assert_eq!(pool.digest(), before);
            }
        }
    }
}

#[test]
fn widen_shares_pool() {
    let mut pool = Pool::new(3, 4, Box::new(Oldest::default()));
    let a = pool.reserve(T, 4, 4).unwrap();
    let b = pool.reserve(T, 4, 4).unwrap();
    pool.widen(a, 8).unwrap();
    pool.grow(a, 8).unwrap();
    assert_eq!(pool.widen(b, 8), Err(Error::NoRoom(1)));
    pool.release(a).unwrap();
    pool.widen(b, 12).unwrap();
    pool.grow(b, 12).unwrap();
}

#[test]
fn oldest_ends_first() {
    let mut policy = Oldest::default();
    for at in [2, 0, 1] {
        policy.retained(at);
    }
    policy.used(2);
    assert_eq!(policy.victim(), Some(2));
    policy.gone(2);
    assert_eq!(policy.victim(), Some(0));
}
