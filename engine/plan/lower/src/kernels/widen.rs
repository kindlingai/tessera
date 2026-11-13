//! A narrow value as floats, for kernels that read floats.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::form::Form as Stored;
use crate::kernel::Kernel;
use crate::plan::Step;
use codegen::prelude::*;
use sim::Load;

/// One thread per element, widened to f32.
#[kernel]
fn widen(
    #[form(form)] x: Buf<Dyn>,
    mut out: BufMut<F32>,
    len: u32,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= len {
        return;
    }
    out[gid() as u64] = x[gid() as u64];
}

/// Widens `len` elements of `form` to f32.
#[derive(Clone, Copy, Debug)]
pub struct Widen {
    pub len: usize,
    pub form: &'static dyn Stored,
}

impl Kernel for Widen {
    fn name(&self) -> &'static str {
        "widen"
    }

    fn source(&self) -> Source {
        let mut body = widen_ir(self.form);
        body.entry = format!("plan_widen_{}", self.form.name());
        Source::Made(body)
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        vec![Launch {
            args: vec![step.read(0), step.out(), Arg::count(self.len)],
            shape: Shape::Line(self.len),
        }]
    }

    fn load(&self, _: usize) -> Option<Load> {
        let len = self.len as f64;
        Some(Load {
            threads: self.len,
            group: 256,
            trips: 1.0,
            trip: widen_ir(self.form).trip(),
            reads: vec![self.form.bytes(self.len) as f64],
            written: 4.0 * len,
            flight: 4.0,
        })
    }

    fn accepts(&self, _: usize, form: &dyn Stored) -> bool {
        form.name() == self.form.name()
    }
}
