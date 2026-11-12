//! A narrow value as floats, for kernels that read floats.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::emit::{Text, float};
use crate::form::Stored;
use crate::kernel::Kernel;
use crate::plan::Step;
use graph::DType;
use sim::{Load, Ops};

/// Widens `len` elements of `form` to f32.
#[derive(Clone, Copy, Debug)]
pub struct Widen {
    pub len: usize,
    pub form: DType,
}

impl Kernel for Widen {
    fn name(&self) -> &'static str {
        "widen"
    }

    fn source(&self) -> Source {
        let load = self.form.load("x", "gid");
        Source::Made(Text {
            entry: format!("plan_widen_{}", self.form.name()),
            buffers: vec![("x".into(), self.form.ty(), false), float("out", true)],
            scalars: vec![("len", "u32")],
            body: format!(
                "    if (gid >= len) {{\n        return;\n    }}\n    out[gid] = {load};\n"
            ),
        })
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        vec![Launch {
            args: vec![step.read(0), step.out(), Arg::count(self.len)],
            shape: Shape::Line(self.len),
        }]
    }

    fn load(&self, _: usize) -> Option<Load> {
        let len = self.len as f64;
        let trip = Ops {
            int: 1.0,
            loads: 1.0,
            stores: 1.0,
            fresh: 1.0,
            ..Ops::default()
        };
        Some(Load {
            threads: self.len,
            group: 256,
            trips: 1.0,
            trip,
            reads: vec![self.form.bytes() as f64 * len],
            written: 4.0 * len,
            flight: 4.0,
        })
    }

    fn accepts(&self, _: usize, form: DType) -> bool {
        form == self.form
    }
}
