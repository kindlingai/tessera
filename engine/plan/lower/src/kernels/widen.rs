//! A narrow value as floats, for kernels that read floats.

use crate::Kernel;
use crate::abi::Source;
use crate::form::Load as Stored;
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

macro_rules! cast_kernel {
    (
        $(#[$meta:meta])*
        $vis:vis struct $ty:ident,
        kernel = $kernel:ident,
        ir = $ir:ident,
        name = $name:literal,
        $hook:ident = $hook_fn:ident,
        load = $load_fn:ident,
        source = $src_fn:ident,
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Kernel)]
        #[launch(
            $kernel(self.form),
            crate = crate,
            name = $name,
            threads = self.len,
            sizes = [],
            source = $src_fn,
            load = $load_fn,
            load_key = crate::load_key((self.len, self.form.name())),
            $hook = $hook_fn,
        )]
        $vis struct $ty {
            pub len: usize,
            pub form: &'static dyn Stored,
        }

        fn $src_fn(k: &$ty) -> Source {
            Source::named($ir(k.form), format!("plan_{}_{}", $name, k.form.name()))
        }
    };
}

cast_kernel! {
    /// Widens `len` elements of `form` to f32.
    pub struct Widen,
    kernel = widen,
    ir = widen_ir,
    name = "widen",
    accepts = widen_accepts,
    load = widen_load,
    source = widen_source,
}

fn cast_load(len: usize, trip: crate::Ops, read: f64, written: f64) -> Option<Load> {
    Some(Load {
        threads: len,
        group: 256,
        trips: 1.0,
        trip,
        reads: vec![read],
        written,
        flight: 4.0,
        ..Load::default()
    })
}

fn widen_load(k: &Widen, _: usize) -> Option<Load> {
    cast_load(
        k.len,
        widen_ir(k.form).trip(),
        k.form.bytes(k.len) as f64,
        4.0 * k.len as f64,
    )
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

cast_kernel! {
    /// Narrows `len` floats into `form`.
    pub(crate) struct Narrow,
    kernel = narrow,
    ir = narrow_ir,
    name = "narrow",
    emits = narrow_emits,
    load = narrow_load,
    source = narrow_source,
}

fn narrow_load(k: &Narrow, _: usize) -> Option<Load> {
    cast_load(
        k.len,
        narrow_ir(k.form).trip(),
        4.0 * k.len as f64,
        k.form.bytes(k.len) as f64,
    )
}

fn narrow_emits(k: &Narrow, form: &dyn Stored) -> bool {
    form.name() == k.form.name()
}
