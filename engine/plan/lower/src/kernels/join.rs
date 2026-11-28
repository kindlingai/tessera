//! Joins rows of two inputs along the last axis.

use crate::abi::Source;
use crate::form::Load as Stored;
use crate::{Extent, Kernel};
use tessera_codegen::prelude::*;

/// One thread per output element or quad.
#[kernel]
#[allow(clippy::too_many_arguments)]
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
    #[build] wide: bool,
) {
    if wide {
        let left4 = left / 4;
        let width4 = left4 + right / 4;
        let row = gid() / width4;
        let col4 = gid() % width4;
        if row >= rows {
            return;
        }
        if col4 < left4 {
            let v = a.load4(row as u64 * left as u64 + col4 as u64 * 4u64);
            out.store4(gid() as u64 * 4u64, F32x4::new(v.x, v.y, v.z, v.w));
        } else {
            let v = b.load4(row as u64 * right as u64 + (col4 - left4) as u64 * 4u64);
            out.store4(gid() as u64 * 4u64, F32x4::new(v.x, v.y, v.z, v.w));
        }
    } else {
        let width = left as u64 + right as u64;
        let row = gid() as u64 / width;
        let col = gid() as u64 % width;
        if row >= rows as u64 {
            return;
        }
        let v = if col < left as u64 {
            a[row * left as u64 + col]
        } else {
            b[row * right as u64 + (col - left as u64)]
        };
        out[gid() as u64] = v;
    }
}

/// Each output row: a `left` row, then a `right` row.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    join(self.a_form, self.b_form, self.out_form, self.wide()),
    crate = crate,
    threads = self.rows.times(self.thread_stride().unwrap_or(0)),
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

impl Join {
    fn wide(&self) -> bool {
        let float_form = |f: &dyn Stored| {
            matches!(
                f.dtype(),
                tessera_graph::DType::F32
                    | tessera_graph::DType::Bf16
                    | tessera_graph::DType::F16
                    | tessera_graph::DType::Fp8
            )
        };
        float_form(self.a_form)
            && float_form(self.b_form)
            && float_form(self.out_form)
            && self.left.is_multiple_of(4)
            && self.right.is_multiple_of(4)
    }

    fn thread_stride(&self) -> Option<usize> {
        if self.wide() {
            (self.left / 4).checked_add(self.right / 4)
        } else {
            self.left.checked_add(self.right)
        }
    }

    /// Whether widths, rows, and total threads fit in `u32`.
    pub fn fits(&self) -> bool {
        let max = u32::MAX as usize;
        if self.left > max || self.right > max || self.rows.most() > max {
            return false;
        }
        let Some(stride) = self.thread_stride() else {
            return false;
        };
        stride > 0
            && self
                .rows
                .most()
                .checked_mul(stride)
                .is_some_and(|threads| threads <= max)
    }
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
    let wide = k.wide();
    let suffix = if wide { "_x4" } else { "" };
    let entry = match (k.a_form.name(), k.b_form.name(), k.out_form.name()) {
        ("f32", "f32", "f32") => format!("plan_join{suffix}"),
        (a, b, o) => format!("plan_join{suffix}_{a}_{b}_{o}"),
    };
    Source::named(join_ir(k.a_form, k.b_form, k.out_form, wide), entry)
}
