//! Whole floats as indices, such as fixed id tables.

use tessera_codegen::prelude::*;
use tessera_graph::{DType, Op};
use tessera_lower::form::Load as Stored;
use tessera_lower::{Draft, Kernel, T, Term, Tree};

/// One thread per element. Negatives and NaN give zero.
#[kernel]
fn cast(
    #[form(form)] x: Buf<Dyn>,
    mut out: BufMut<U32>,
    len: u32,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= len {
        return;
    }
    let v = x[gid() as u64];
    let mut id = 0u32;
    if v > 0.0f32 {
        id = v as u32;
    }
    out[gid() as u64] = id;
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(cast(self.form), name = self.form.suffixed("cast"), threads = self.len, accepts = cast_accepts)]
pub struct Cast {
    pub len: usize,
    /// How the input is stored.
    pub form: &'static dyn Stored,
}

fn cast_accepts(k: &Cast, input: usize, form: &dyn Stored) -> bool {
    k.form.taken(input, 0, form)
}

/// One thread per element, integer id to float.
#[kernel]
fn cast_f32(x: Buf<U32>, mut out: BufMut<F32>, len: u32) {
    if gid() >= len {
        return;
    }
    out[gid() as u64] = x[gid() as u64] as f32;
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(cast_f32, threads = self.len)]
struct CastF32 {
    len: usize,
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let x = *node.args.first()?;
    let len = tree.elements(at);
    match node.term {
        Term::Op(Op::Cast(DType::U32)) => {
            let cast = Cast {
                len,
                form: tree.node(x).form,
            };
            Some(vec![Draft::new(cast, [x], len)])
        }
        Term::Op(Op::Cast(DType::F32)) if tree.node(x).form.dtype() == DType::U32 => {
            Some(vec![Draft::new(CastF32 { len }, [x], len)])
        }
        _ => None,
    }
}

tessera_lower::rule!("cast", propose);
