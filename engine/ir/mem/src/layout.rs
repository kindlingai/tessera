//! Where each element of a view lies, in elements.

/// A shape with a stride per axis. Strides may skip.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Layout {
    /// Row-major, with no gaps.
    pub fn contiguous(shape: Vec<usize>) -> Self {
        let mut strides = vec![1; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }
        Self { shape, strides }
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_contiguous(&self) -> bool {
        *self == Self::contiguous(self.shape.clone())
    }

    /// One past the furthest element, from the first.
    pub fn span(&self) -> usize {
        if self.is_empty() {
            return 0;
        }
        let last: usize = self
            .shape
            .iter()
            .zip(&self.strides)
            .map(|(n, s)| (n - 1) * s)
            .sum();
        last + 1
    }

    /// Each element's offset, in row-major order.
    pub fn offsets(&self) -> impl Iterator<Item = usize> + '_ {
        let rank = self.shape.len();
        (0..self.len()).map(move |mut flat| {
            let mut at = 0;
            for d in (0..rank).rev() {
                at += (flat % self.shape[d]) * self.strides[d];
                flat /= self.shape[d];
            }
            at
        })
    }

    /// The axes in `order`. No element moves.
    pub fn permute(&self, order: &[usize]) -> Self {
        Self {
            shape: order.iter().map(|d| self.shape[*d]).collect(),
            strides: order.iter().map(|d| self.strides[*d]).collect(),
        }
    }

    /// `len` entries along `axis` from `start`, and where they begin.
    pub fn slice(&self, axis: usize, start: usize, len: usize) -> (Self, usize) {
        let mut shape = self.shape.clone();
        shape[axis] = len;
        let layout = Self {
            shape,
            strides: self.strides.clone(),
        };
        (layout, start * self.strides[axis])
    }
}
