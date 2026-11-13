//! A value crossing from one zone to another.

use crate::abi::{Launch, Source};
use crate::kernel::Kernel;
use crate::kernels::Map;
use crate::plan::Step;

/// Copies `len` elements into the step's zone.
#[derive(Clone, Copy, Debug)]
pub struct Move {
    pub len: usize,
}

impl Kernel for Move {
    fn name(&self) -> &'static str {
        "move"
    }

    /// The runner copies, so nothing compiles.
    fn source(&self) -> Source {
        Map(Vec::new()).source()
    }

    fn launches(&self, _: &Step) -> Vec<Launch> {
        Vec::new()
    }
}
