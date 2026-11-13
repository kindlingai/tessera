//! A size fixed at lowering, or read at launch.

use graph::DimName;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extent {
    Fixed(usize),
    /// `mul` times the dimension's value, plus `add`.
    Dim {
        name: DimName,
        mul: usize,
        add: i64,
        /// The value at the plan's largest size.
        most: usize,
    },
}

impl Extent {
    /// The value buffers and estimates are sized for.
    pub fn most(&self) -> usize {
        match *self {
            Extent::Fixed(n) => n,
            Extent::Dim { most, .. } => most,
        }
    }

    pub fn runtime(&self) -> bool {
        matches!(self, Extent::Dim { .. })
    }

    /// The value once `size` gives the dimension.
    pub fn value(&self, size: &dyn Fn(DimName) -> Option<u32>) -> Option<usize> {
        Some(match *self {
            Extent::Fixed(n) => n,
            Extent::Dim { name, mul, add, .. } => {
                let at = i64::from(size(name)?) * mul as i64 + add;
                usize::try_from(at).ok()?
            }
        })
    }

    pub fn times(self, by: usize) -> Self {
        match self {
            Extent::Fixed(n) => Extent::Fixed(n * by),
            Extent::Dim {
                name,
                mul,
                add,
                most,
            } => Extent::Dim {
                name,
                mul: mul * by,
                add: add * by as i64,
                most: most * by,
            },
        }
    }

    pub fn less(self, n: usize) -> Self {
        match self {
            Extent::Fixed(all) => Extent::Fixed(all - n),
            Extent::Dim {
                name,
                mul,
                add,
                most,
            } => Extent::Dim {
                name,
                mul,
                add: add - n as i64,
                most: most - n,
            },
        }
    }
}
