//! Copies new rows in after a plan's kept rows.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::extent::Extent;
use crate::kernel::Kernel;
use crate::kernels::Map;
use crate::plan::Step;

/// Copies `len` elements to `skip` elements into the kept rows.
#[derive(Clone, Copy, Debug)]
pub struct Append {
    pub skip: Extent,
    pub len: usize,
}

impl Kernel for Append {
    fn name(&self) -> &'static str {
        "append"
    }

    /// A plain copy, written at an offset.
    fn source(&self) -> Source {
        Map(Vec::new()).source()
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let out = Arg::offset(step.write, self.skip);
        vec![Launch {
            args: vec![step.read(0), out, Arg::count(self.len)],
            shape: Shape::Line(self.len),
        }]
    }

    /// The rows already kept cost nothing.
    fn work(&self, _: usize) -> f64 {
        self.len as f64
    }
}
