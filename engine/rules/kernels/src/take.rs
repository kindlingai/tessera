//! Entries of each row at given indices.

use tessera_codegen::prelude::*;
use tessera_graph::{Op, Select};
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::{Draft, Kernel, Source, T, Term, Tree};

/// One thread per output. Ids past the row read zero.
#[kernel]
fn take(
    #[form(x_form)] x: Buf<Dyn>,
    ids: Buf<U32>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    len: u32,
    width: u32,
    picks: u32,
    first: u32,
    #[build] x_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    if gid() < len {
        let idx = ids[gid() as u64];
        let row = (gid() / picks) as u64 * width as u64;
        out[gid() as u64] = if idx >= first && idx - first < width {
            x[row + (idx - first) as u64]
        } else {
            0.0f32
        };
    }
}

/// `picks` ids per row of `width`.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    take(self.x_form, self.out_form),
    threads = self.len,
    accepts = take_accepts,
    emits = take_emits,
    source = take_source,
)]
pub struct Take {
    pub len: usize,
    pub width: usize,
    pub picks: usize,
    pub first: usize,
    pub x_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

fn take_source(k: &Take) -> Source {
    let entry = match (k.x_form.name(), k.out_form.name()) {
        ("f32", "f32") => "plan_take".into(),
        (a, b) => format!("plan_take_{a}_{b}"),
    };
    Source::named(take_ir(k.x_form, k.out_form), entry)
}

fn take_accepts(k: &Take, input: usize, form: &dyn Stored) -> bool {
    if input == 0 {
        form.name() == k.x_form.name()
    } else {
        !form.narrow()
    }
}

fn take_emits(k: &Take, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::Select(Select::Take)) {
        return None;
    }
    let (x, ids) = (node.args[0], node.args[1]);
    let width = *tree.node(x).shape.last()?;
    let len = tree.elements(ids);
    let (first, _) = tree.row_share(x);
    let take = Take {
        len,
        width,
        picks: len / (tree.elements(x) / width).max(1),
        first,
        x_form: plain(tree.node(x).form),
        out_form: writable(node.form),
    };
    Some(vec![Draft::new(take, [x, ids], len)])
}

tessera_lower::rule!("take", propose);
