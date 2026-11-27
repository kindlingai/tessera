use crate::ranks::zigzag_order;

#[test]
fn zigzag_balances_chunks() {
    for (filled, chunks, ranks, page) in [(4096, 2, 4, 64), (2048, 1, 4, 64), (16, 2, 2, 8)] {
        let (perm, inv) = zigzag_order(filled, chunks, ranks, page).expect("order");
        assert_eq!(perm.len(), filled);
        assert_eq!(inv.len(), filled);
        for i in 0..filled {
            assert_eq!(perm[inv[i] as usize], i);
            assert_eq!(inv[perm[i]] as usize, i);
        }
        let len = filled / chunks;
        let per = len / ranks;
        for c in 0..chunks {
            let base = c * len;
            let mut sums = Vec::with_capacity(ranks);
            for r in 0..ranks {
                let slice = &perm[base + r * per..base + (r + 1) * per];
                assert!(slice.iter().all(|&i| i >= base && i < base + len));
                assert!(slice.windows(2).all(|w| w[0] < w[1]));
                sums.push(slice.iter().sum::<usize>());
            }
            assert!(sums.iter().all(|&s| s == sums[0]), "imbalanced: {sums:?}");
        }
    }
    assert!(zigzag_order(4, 2, 2, 8).is_none());
    assert!(zigzag_order(16, 1, 1, 8).is_none());
}
