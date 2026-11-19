use sensai_kv::{Pool, Prefixes, Tag};

#[test]
fn longest_newest_wins() {
    let (mut pool, mut index) = (Pool::new(8, 2), Prefixes::new(2));
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
