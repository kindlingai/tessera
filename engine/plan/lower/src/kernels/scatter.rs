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
) {
    if gid() >= rows * width {
        return;
    }
    let r = gid() / width;
    let j = gid() % width;
    let at = kept_row(slots[r as u64], positions[r as u64]) * width + j;
    out[at as u64] = x[gid() as u64];
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
    pub input: &'static dyn Stored,
    pub output: &'static dyn Stored,
}

impl Scatter {
    /// The scatter that runs append node `t` into kept `index`.
    pub(crate) fn at(tree: &Tree, t: T, index: u32) -> Option<Scatter> {
        let (_, context) = tree.kept_rows(index)?;
        let node = tree.node(t);
        let all = &node.shape;
        let width: usize = all[2..].iter().product();
        let src = node.args[0];
        Some(Scatter {
            rows: tree.elements(src) / width,
            width,
            slots: all[0],
            context,
            kept: tree.kept,
            input: plain(tree.node(src).form),
            output: writable(node.form),
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
    let body = kept::attach(scatter_ir(k.input, k.output), k.kept);
    let entry = match (k.input.name(), k.output.name()) {
        ("f32", "f32") => format!("plan_scatter_{}", k.kept.name()),
        (a, b) => format!("plan_scatter_{}_{a}_{b}", k.kept.name()),
    };
    Source::named(body, entry)
}

fn scatter_launches(k: &Scatter, step: &Step) -> Vec<Launch> {
    let sizes = [k.rows, k.width].map(Arg::count);
    let values = k.kept.values(k.slots, k.context);
    let kept = values.into_iter().map(|v| Arg::count(v as usize));
    let args = [step.read(0), step.read(1), step.read(2), step.out()];
    let table = step.kept(k.kept);
    vec![Launch {
        args: args
            .into_iter()
            .chain(table)
            .chain(sizes)
            .chain(kept)
            .collect(),
        shape: Shape::Line(k.rows * k.width),
    }]
}
