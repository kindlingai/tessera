//! Writes rows into kept rows by slot and position.

use crate::Kernel;
use crate::abi::{Arg, Launch, Shape, Source};
use crate::form::{Load as Stored, plain, writable};
use crate::plan::Step;
use crate::tree::{T, Tree};
use tessera_codegen::kept::{self, Layout};
use tessera_codegen::prelude::*;

/// Row `r` lands in its slot's kept row.
#[kernel]
fn scatter(
    #[form(in_form)] x: Buf<Dyn>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    rows: u32,
    width: u32,
    #[build] in_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
    #[build] wide: bool,
) {
    if wide {
        let quads = width / 4;
        if gid() >= rows * quads {
            return;
        }
        let r = gid() / quads;
        let j = (gid() % quads) * 4;
        let dst = kept_row(slots[r as u64], positions[r as u64]) as u64 * width as u64 + j as u64;
        let v = x.load4(gid() as u64 * 4);
        out.store4(dst, F32x4::new(v.x, v.y, v.z, v.w));
    } else {
        if gid() >= rows * width {
            return;
        }
        let r = gid() / width;
        let j = gid() % width;
        let at = kept_row(slots[r as u64], positions[r as u64]) as u64 * width as u64 + j as u64;
        out[at] = x[gid() as u64];
    }
}

/// `rows` of `width` elements into `slots` of `context` rows.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = crate,
    name = "scatter",
    sizes = [],
    work = self.rows * self.width,
    accepts = scatter_accepts,
    emits = scatter_emits,
    source = scatter_source,
    launches = scatter_launches,
)]
pub(crate) struct Scatter {
    pub rows: usize,
    pub width: usize,
    pub slots: usize,
    pub context: usize,
    pub kept: &'static dyn Layout,
    pub sparse: bool,
    pub input: &'static dyn Stored,
    pub output: &'static dyn Stored,
    pub wide: bool,
}

impl Scatter {
    /// The scatter that runs append node `t` into kept `index`.
    pub(crate) fn at(tree: &Tree, t: T, index: u32) -> Option<Scatter> {
        let (_, kept, context) = tree.kept_layout(index)?;
        let node = tree.node(t);
        let all = &node.shape;
        let width: usize = all[2..].iter().product();
        let src = node.args[0];
        let input = plain(tree.node(src).form);
        let output = writable(node.form);
        let wide = width.is_multiple_of(4)
            && matches!(input.name(), "f32" | "bf16" | "f16" | "fp8")
            && matches!(output.name(), "f32" | "bf16" | "fp8");
        Some(Scatter {
            rows: tree.elements(src) / width,
            width,
            slots: all[0],
            context,
            kept,
            sparse: tree.is_sparse_kept(index),
            input,
            output,
            wide,
        })
    }
}

fn scatter_accepts(k: &Scatter, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.input.name(),
        _ => !form.narrow(),
    }
}

fn scatter_emits(k: &Scatter, form: &dyn Stored) -> bool {
    form.name() == k.output.name()
}

fn scatter_source(k: &Scatter) -> Source {
    let body = kept::attach(scatter_ir(k.input, k.output, k.wide), k.kept);
    let suffix = if k.wide { "_x4" } else { "" };
    let entry = match (k.input.name(), k.output.name()) {
        ("f32", "f32") => format!("plan_scatter{suffix}_{}", k.kept.name()),
        (a, b) => format!("plan_scatter{suffix}_{}_{a}_{b}", k.kept.name()),
    };
    Source::named(body, entry)
}

fn scatter_launches(k: &Scatter, step: &Step) -> Vec<Launch> {
    let sizes = [k.rows, k.width].map(Arg::count);
    let values = k.kept.values(k.slots, k.context);
    let kept = values.into_iter().map(|v| Arg::count(v as usize));
    let args = [step.read(0), step.read(1), step.read(2), step.out()];
    let table = step.kept_for(k.kept, !k.sparse);
    let threads = if k.wide {
        k.rows * (k.width / 4)
    } else {
        k.rows * k.width
    };
    vec![Launch {
        args: args
            .into_iter()
            .chain(table)
            .chain(sizes)
            .chain(kept)
            .collect(),
        shape: Shape::Line(threads),
    }]
}
