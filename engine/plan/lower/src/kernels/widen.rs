//! A narrow value as floats, for kernels that read floats.

use crate::Kernel;
use crate::abi::Source;
use crate::form::Form as Stored;
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
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    widen(self.form),
    crate = crate,
    name = "widen",
    threads = self.len,
    sizes = [],
    source = widen_source,
    load = widen_load,
    accepts = widen_accepts,
)]
pub struct Widen {
    pub len: usize,
    pub form: &'static dyn Stored,
}

fn widen_source(k: &Widen) -> Source {
    let entry = format!("plan_widen_{}", k.form.name());
    Source::named(widen_ir(k.form), entry)
}

fn widen_load(k: &Widen, _: usize) -> Option<Load> {
    let len = k.len as f64;
    Some(Load {
        threads: k.len,
        group: 256,
        trips: 1.0,
        trip: widen_ir(k.form).trip(),
        reads: vec![k.form.bytes(k.len) as f64],
        written: 4.0 * len,
        flight: 4.0,
    })
}

fn widen_accepts(k: &Widen, _: usize, form: &dyn Stored) -> bool {
    form.name() == k.form.name()
}
