//! Entries past each row's position, set to -inf.

use crate::WIDE;
use crate::mask::{inf_writable, masked_source};
use tessera_codegen::prelude::*;
use tessera_graph::Stage;
use tessera_lower::form::{Load as Stored, plain};
use tessera_lower::{Draft, Kernel, Source, T, Tree, sites};

/// Masks past each row's position, four columns per thread.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn causal(
    #[form(x_form)] x: Buf<Dyn>,
    positions: Buf<U32>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    len: u32,
    width: u32,
    per: u32,
    #[build] wide: bool,
    #[build] x_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    if wide {
        let quads = width / WIDE;
        if gid() >= len / WIDE {
            return;
        }
        let col = (gid() % quads) * WIDE;
        let limit = positions[(gid() / quads / per) as u64];
        let base = gid() as u64 * WIDE as u64;
        let mut o = F32x4::zero();
        if col > limit {
            o.x = f32::NEG_INFINITY;
            o.y = f32::NEG_INFINITY;
            o.z = f32::NEG_INFINITY;
            o.w = f32::NEG_INFINITY;
        } else {
            let v = x.load4(base);
            o.x = v.x;
            o.y = if col + 1 > limit {
                f32::NEG_INFINITY
            } else {
                v.y
            };
            o.z = if col + 2 > limit {
                f32::NEG_INFINITY
            } else {
                v.z
            };
            o.w = if col + 3 > limit {
                f32::NEG_INFINITY
            } else {
                v.w
            };
        }
        out.store4(base, o);
    } else {
        if gid() >= len {
            return;
        }
        out[gid() as u64] = if gid() % width > positions[(gid() / width / per) as u64] {
            f32::NEG_INFINITY
        } else {
            x[gid() as u64]
        };
    }
}

/// Rows of `width`, `per` rows to a position.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    causal(self.wide, self.x_form, self.out_form),
    name = "causal",
    threads = if self.wide { self.len / WIDE as usize } else { self.len },
    work = self.len,
    accepts = causal_accepts,
    emits = causal_emits,
    source = causal_source,
)]
pub(crate) struct Causal {
    pub len: usize,
    pub width: usize,
    pub per: usize,
    pub wide: bool,
    pub x_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

fn causal_accepts(k: &Causal, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.x_form.name(),
        _ => !form.narrow(),
    }
}

fn causal_emits(k: &Causal, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn causal_source(k: &Causal) -> Source {
    let body = causal_ir(k.wide, k.x_form, k.out_form);
    masked_source("causal", body, k.wide, k.x_form, k.out_form)
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    let (Stage::Causal, Some((_, positions))) = (&site.stage, site.operand) else {
        return None;
    };
    let rows = site.len / site.width;
    let kernel = Causal {
        len: site.len,
        width: site.width,
        per: rows / positions.max(1),
        wide: site.width.is_multiple_of(WIDE as usize),
        x_form: plain(tree.node(site.x).form),
        out_form: inf_writable(tree.node(at).form),
    };
    Some(vec![Draft::new(kernel, site.reads(), site.len)])
}

tessera_lower::rule!("causal", propose);
