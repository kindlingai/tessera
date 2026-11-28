//! Scores a query cannot see, set to -inf.

use crate::WIDE;
use tessera_codegen::prelude::*;
use tessera_graph::Op;
use tessera_lower::form::{F32_FORM, Load as Stored, plain};
use tessera_lower::{Draft, Kernel, Source, T, Term, Tree};

/// Returns `val` if `key` lies inside `pos`'s window, else `-inf`.
#[kernel]
fn keep_score(val: f32, key: u32, pos: u32, back: u32, ahead: u32) -> f32 {
    let early = pos > back && key < pos - back;
    let late = key > pos && key - pos > ahead;
    if early || late {
        f32::NEG_INFINITY
    } else {
        val
    }
}

/// Masks `[.., queries, keys]`, four adjacent keys per thread when aligned.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn mask(
    #[form(x_form)] x: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    len: u32,
    queries: u32,
    keys: u32,
    past: u32,
    back: u32,
    ahead: u32,
    #[build] wide: bool,
    #[build] x_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    if wide {
        let quads = keys / WIDE;
        if gid() >= len / WIDE {
            return;
        }
        let key = (gid() % quads) * WIDE;
        let position = past + (gid() / quads) % queries;
        let base = gid() as u64 * WIDE as u64;
        let all_early = position > back && key + (WIDE - 1) < position - back;
        let all_late = key > position && key - position > ahead;
        let mut o = F32x4::zero();
        if all_early || all_late {
            o.x = f32::NEG_INFINITY;
            o.y = f32::NEG_INFINITY;
            o.z = f32::NEG_INFINITY;
            o.w = f32::NEG_INFINITY;
        } else {
            let v = x.load4(base);
            o.x = keep_score(v.x, key, position, back, ahead);
            o.y = keep_score(v.y, key + 1, position, back, ahead);
            o.z = keep_score(v.z, key + 2, position, back, ahead);
            o.w = keep_score(v.w, key + 3, position, back, ahead);
        }
        out.store4(base, o);
    } else {
        if gid() >= len {
            return;
        }
        let key = gid() % keys;
        let position = past + (gid() / keys) % queries;
        let early = position > back && key < position - back;
        let late = key > position && key - position > ahead;
        out[gid() as u64] = if early || late {
            f32::NEG_INFINITY
        } else {
            x[gid() as u64]
        };
    }
}

/// Keys before the first query, and how far queries see.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    mask(self.wide, self.x_form, self.out_form),
    name = "mask",
    threads = if self.wide { self.len / WIDE as usize } else { self.len },
    work = self.len,
    accepts = mask_accepts,
    emits = mask_emits,
    source = mask_source,
)]
pub(crate) struct Masked {
    pub len: usize,
    pub queries: usize,
    pub keys: usize,
    pub past: usize,
    pub back: usize,
    pub ahead: usize,
    pub wide: bool,
    pub x_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

/// Writable float forms that preserve `-inf` exactly.
pub(crate) fn inf_writable(form: &'static dyn Stored) -> &'static dyn Stored {
    match form.name() {
        "bf16" | "f16" if form.stores() => form,
        _ => F32_FORM,
    }
}

fn mask_accepts(k: &Masked, _: usize, form: &dyn Stored) -> bool {
    form.name() == k.x_form.name()
}

fn mask_emits(k: &Masked, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

/// Builds a named source for a masked kernel variant.
pub(crate) fn masked_source(
    base: &str,
    body: tessera_codegen::ir::Body,
    wide: bool,
    x_form: &'static dyn Stored,
    out_form: &'static dyn Stored,
) -> Source {
    let mode = if wide { "w" } else { "s" };
    let entry = format!("plan_{base}_{mode}_{}_{}", x_form.name(), out_form.name());
    Source::named(body, entry)
}

fn mask_source(k: &Masked) -> Source {
    let body = mask_ir(k.wide, k.x_form, k.out_form);
    masked_source("mask", body, k.wide, k.x_form, k.out_form)
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Op(Op::Mask(kind)) = node.term else {
        return None;
    };
    let shape = &node.shape;
    let [.., queries, keys] = shape[..] else {
        return None;
    };
    let (back, ahead) = kind.window(u32::MAX as usize);
    let len = tree.elements(at);
    let x = node.args[0];
    let masked = Masked {
        len,
        queries,
        keys,
        past: keys.saturating_sub(queries),
        back,
        ahead,
        wide: keys.is_multiple_of(WIDE as usize),
        x_form: plain(tree.node(x).form),
        out_form: inf_writable(node.form),
    };
    Some(vec![Draft::new(masked, [x], len)])
}

tessera_lower::rule!("mask", propose);
