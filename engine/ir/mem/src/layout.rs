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
        self.shape.contains(&0)
    }

    pub fn is_contiguous(&self) -> bool {
        if self.strides.len() != self.shape.len() {
            return false;
        }
        let mut want = 1usize;
        for d in (0..self.shape.len()).rev() {
            if self.strides[d] != want {
                return false;
            }
            if d > 0 {
                want *= self.shape[d];
            }
        }
        true
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
        Offsets::new(self)
    }

    /// Trailing contiguous run length and outer axes for chunked copies.
    pub(crate) fn chunk_layout(&self) -> (usize, usize) {
        let mut chunk = 1usize;
        let mut outer = self.shape.len();
        while outer > 0 && self.strides[outer - 1] == chunk {
            outer -= 1;
            chunk *= self.shape[outer];
        }
        (chunk, outer)
    }

    /// Row-major starting offsets for the first `outer` axes.
    pub(crate) fn prefix_offsets(&self, outer: usize) -> Offsets<'_> {
        Offsets::for_prefix(self, outer)
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

enum Coords {
    Small([usize; 4]),
    Large(Vec<usize>),
}

impl Coords {
    fn as_mut_slice(&mut self) -> &mut [usize] {
        match self {
            Coords::Small(a) => a,
            Coords::Large(v) => v,
        }
    }
}

pub(crate) struct Offsets<'a> {
    shape: &'a [usize],
    strides: &'a [usize],
    outer_left: usize,
    at: usize,
    inner: usize,
    inner_len: usize,
    inner_stride: usize,
    inner_back: usize,
    coords: Coords,
}

impl<'a> Offsets<'a> {
    fn new(layout: &'a Layout) -> Self {
        Self::for_prefix(layout, layout.shape.len())
    }

    fn for_prefix(layout: &'a Layout, rank: usize) -> Self {
        let shape = &layout.shape[..rank];
        let strides = &layout.strides[..rank];
        let total: usize = shape.iter().product();
        if total == 0 {
            return Self {
                shape,
                strides,
                outer_left: 0,
                at: 0,
                inner: 0,
                inner_len: 0,
                inner_stride: 0,
                inner_back: 0,
                coords: Coords::Small([0; 4]),
            };
        }
        let outer = rank.saturating_sub(1);
        let inner_len = shape.last().copied().unwrap_or(1);
        let inner_stride = strides.last().copied().unwrap_or(0);
        let coords = if outer <= 4 {
            Coords::Small([0; 4])
        } else {
            Coords::Large(vec![0; outer])
        };
        Self {
            shape,
            strides,
            outer_left: total / inner_len - 1,
            at: 0,
            inner: 0,
            inner_len,
            inner_stride,
            inner_back: inner_len.wrapping_mul(inner_stride),
            coords,
        }
    }

    #[cold]
    fn next_row(&mut self) -> Option<usize> {
        if self.outer_left == 0 {
            return None;
        }
        self.outer_left -= 1;
        self.at = self.at.wrapping_sub(self.inner_back);
        let coords = self.coords.as_mut_slice();
        for d in (0..self.shape.len() - 1).rev() {
            coords[d] += 1;
            if coords[d] < self.shape[d] {
                self.at = self.at.wrapping_add(self.strides[d]);
                break;
            }
            coords[d] = 0;
            self.at = self
                .at
                .wrapping_sub((self.shape[d] - 1).wrapping_mul(self.strides[d]));
        }
        let out = self.at;
        self.at = self.at.wrapping_add(self.inner_stride);
        self.inner = 1;
        Some(out)
    }
}

impl Iterator for Offsets<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.inner < self.inner_len {
            let out = self.at;
            self.at = self.at.wrapping_add(self.inner_stride);
            self.inner += 1;
            return Some(out);
        }
        self.next_row()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = (self.inner_len - self.inner) + self.outer_left * self.inner_len;
        (rem, Some(rem))
    }

    fn fold<B, F>(mut self, mut init: B, mut f: F) -> B
    where
        F: FnMut(B, usize) -> B,
    {
        if self.inner_len == 0 {
            return init;
        }
        let stride = self.inner_stride;
        let len = self.inner_len;
        loop {
            let mut at = self.at;
            for _ in self.inner..len {
                init = f(init, at);
                at = at.wrapping_add(stride);
            }
            self.at = at;
            self.inner = len;
            match self.next_row() {
                Some(first) => init = f(init, first),
                None => return init,
            }
        }
    }
}

impl ExactSizeIterator for Offsets<'_> {}
