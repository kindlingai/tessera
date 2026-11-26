//! How a split axis divides among ranks, evenly or not.

/// Each rank's weight. Equal weights give equal parts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parts(pub Vec<u32>);

impl Parts {
    pub fn even(ranks: usize) -> Self {
        Self(vec![1; ranks])
    }

    pub fn ranks(&self) -> usize {
        self.0.len()
    }

    /// Each rank's elements of a `whole` axis the sum divides.
    pub fn counts(&self, whole: usize) -> Option<Vec<usize>> {
        let sum: u32 = self.0.iter().sum();
        if sum == 0 || !whole.is_multiple_of(sum as usize) {
            return None;
        }
        let unit = whole / sum as usize;
        Some(self.0.iter().map(|w| *w as usize * unit).collect())
    }

    /// Rank `rank`'s start and length of `whole` rows, any count.
    pub fn rows(&self, whole: usize, rank: usize) -> (usize, usize) {
        let sum: u64 = self.0.iter().map(|w| u64::from(*w)).sum();
        let before = |r: usize| -> usize {
            let held: u64 = self.0[..r].iter().map(|w| u64::from(*w)).sum();
            (whole as u64 * held / sum.max(1)) as usize
        };
        (before(rank), before(rank + 1) - before(rank))
    }

    /// Rank `rank`'s start and length along a `whole` axis.
    pub fn part(&self, whole: usize, rank: usize) -> Option<(usize, usize)> {
        let counts = self.counts(whole)?;
        let first = counts[..rank].iter().sum();
        Some((first, counts[rank]))
    }
}
