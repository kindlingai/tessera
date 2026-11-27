//! A narrow value as floats, for kernels that read floats.

use crate::Kernel;
use crate::abi::{Arg, Launch, Shape, Source};
use crate::form::Load as Stored;
use crate::plan::Step;
use tessera_codegen::prelude::*;
use tessera_work::Load;

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
        ..Load::default()
    })
}

fn widen_accepts(k: &Widen, _: usize, form: &dyn Stored) -> bool {
    form.name() == k.form.name()
}

/// One thread per element, narrowed from f32.
#[kernel]
fn narrow(
    x: Buf<F32>,
    #[form(form)] mut out: BufMut<Dyn>,
    len: u32,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= len {
        return;
    }
    out[gid() as u64] = x[gid() as u64];
}

/// One thread per 32-element block, narrowed to OCP mxfp8.
#[kernel]
fn narrow_mxfp8(x: Buf<F32>, mut out: BufMut<U8>, blocks: u32) {
    if gid() >= blocks {
        return;
    }
    let src = gid() as u64 * 32;
    let mut amax = 0.0f32;
    for j in 0..32u32 {
        let v = f32::from_bits(x[src + j as u64].to_bits() & 0x7fff_ffffu32);
        amax = max(amax, v);
    }
    let bits = (amax * 0.002232142857f32).to_bits();
    let mut e = bits >> 23;
    if (bits & 0x7f_ffffu32) != 0 {
        e += 1;
    }
    if amax == 0.0f32 {
        e = 127;
    }
    if e < 1 {
        e = 1;
    }
    if e > 253 {
        e = 253;
    }
    let inv = f32::from_bits((254 - e) << 23);
    let dst = gid() as u64 * 33;
    out[dst] = e;
    for j in 0..32u32 {
        out[dst + 1 + j as u64] = to_e4m3(x[src + j as u64] * inv);
    }
}

/// Narrows `len` floats into `form`.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = crate,
    name = "narrow",
    sizes = [],
    work = self.len,
    launches = narrow_launches,
    shape = narrow_shape(self),
    source = narrow_source,
    load = narrow_load,
    emits = narrow_emits,
)]
pub(crate) struct Narrow {
    pub len: usize,
    pub form: &'static dyn Stored,
}

fn narrow_body(k: &Narrow) -> tessera_codegen::ir::Body {
    match k.form.name() {
        "mxfp8" => narrow_mxfp8_ir(),
        _ => narrow_ir(k.form),
    }
}

fn narrow_count(k: &Narrow) -> usize {
    match k.form.name() {
        "mxfp8" => k.len.div_ceil(32),
        _ => k.len,
    }
}

fn narrow_shape(k: &Narrow) -> Shape {
    Shape::Line(narrow_count(k))
}

fn narrow_launches(k: &Narrow, step: &Step) -> Vec<Launch> {
    vec![Launch {
        args: vec![step.read(0), step.out(), Arg::count(narrow_count(k))],
        shape: narrow_shape(k),
    }]
}

fn narrow_source(k: &Narrow) -> Source {
    let entry = format!("plan_narrow_{}", k.form.name());
    Source::named(narrow_body(k), entry)
}

fn narrow_load(k: &Narrow, _: usize) -> Option<Load> {
    let len = k.len as f64;
    Some(Load {
        threads: narrow_count(k),
        group: 256,
        trips: 1.0,
        trip: narrow_body(k).trip(),
        reads: vec![4.0 * len],
        written: k.form.bytes(k.len) as f64,
        flight: 4.0,
        ..Load::default()
    })
}

fn narrow_emits(k: &Narrow, form: &dyn Stored) -> bool {
    form.name() == k.form.name()
}
