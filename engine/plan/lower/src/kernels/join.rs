//! Joins rows of two inputs along the last axis.

use crate::abi::Source;
use crate::form::Load as Stored;
use crate::{Extent, Kernel};
use tessera_codegen::prelude::*;

/// One thread per output element, from whichever side holds it.
#[kernel]
fn join(
    #[form(a_form)] a: Buf<Dyn>,
    #[form(b_form)] b: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    left: u32,
    right: u32,
    rows: u32,
    #[build] a_form: &'static dyn Stored,
    #[build] b_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    let width = left + right;
    let row = gid() / width;
    let col = gid() % width;
    if row >= rows {
        return;
    }
    let v = if col < left {
        a[(row * left + col) as u64]
    } else {
        b[(row * right + col - left) as u64]
    };
    out[gid() as u64] = v;
}

/// Each output row: a `left` row, then a `right` row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    join(self.a_form, self.b_form, self.out_form),
    crate = crate,
    threads = self.rows.times(self.left + self.right),
    sizes = [self.left, self.right],
    accepts = join_accepts,
    emits = join_emits,
    source = join_source,
)]
pub struct Join {
    pub left: usize,
    pub right: usize,
    /// The row count. Decode reads it at launch.
    pub rows: Extent,
    pub a_form: &'static dyn Stored,
    pub b_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

fn join_accepts(k: &Join, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.a_form.name(),
        1 => form.name() == k.b_form.name(),
        _ => !form.narrow(),
    }
}

fn join_emits(k: &Join, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn join_source(k: &Join) -> Source {
    let entry = match (k.a_form.name(), k.b_form.name(), k.out_form.name()) {
        ("f32", "f32", "f32") => "plan_join".into(),
        (a, b, o) => format!("plan_join_{a}_{b}_{o}"),
    };
    Source::named(join_ir(k.a_form, k.b_form, k.out_form), entry)
}
