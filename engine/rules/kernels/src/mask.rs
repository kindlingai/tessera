//! Scores a query cannot see, set to -inf.

use sensai_codegen::prelude::*;
use sensai_graph::Op;
use sensai_lower::{Draft, Kernel, T, Term, Tree};

/// One thread per score over `[.., queries, keys]`.
#[kernel]
fn mask(
    x: Buf<F32>,
    mut out: BufMut<F32>,
    len: u32,
    queries: u32,
    keys: u32,
    past: u32,
    back: u32,
    ahead: u32,
) {
    if gid() >= len {
        return;
    }
    let key = gid() % keys;
    let position = past + (gid() / keys) % queries;
    let early = position > back && key < position - back;
    let late = key > position && key - position > ahead;
    let mut v = x[gid() as u64];
    if early || late {
        v = f32::NEG_INFINITY;
    }
    out[gid() as u64] = v;
}

/// Keys before the first query, and how far queries see.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(mask, threads = self.len)]
pub(crate) struct Masked {
    pub len: usize,
    pub queries: usize,
    pub keys: usize,
    pub past: usize,
    pub back: usize,
    pub ahead: usize,
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
    let masked = Masked {
        len,
        queries,
        keys,
        past: keys.saturating_sub(queries),
        back,
        ahead,
    };
    Some(vec![Draft::new(masked, [node.args[0]], len)])
}

sensai_lower::rule!("mask", propose);
