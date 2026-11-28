//! Whole floats as indices, such as fixed id tables.

use tessera_codegen::prelude::*;
use tessera_graph::{DType, Op};
use tessera_lower::form::{Load as Stored, writable};
use tessera_lower::{Draft, Kernel, Source, T, Term, Tree};

/// One thread per element or quad. Non-positives give zero.
#[kernel]
fn cast(
    #[form(form)] x: Buf<Dyn>,
    mut out: BufMut<U32>,
    len: u32,
    #[build] form: &'static dyn Stored,
    #[build] wide: bool,
) {
    if wide {
        if gid() >= len / 4 {
            return;
        }
        let base = (gid() * 4) as u64;
        let v = x.load4(base);
        out[base] = if v.x > 0.0f32 { v.x as u32 } else { 0u32 };
        out[base + 1u64] = if v.y > 0.0f32 { v.y as u32 } else { 0u32 };
        out[base + 2u64] = if v.z > 0.0f32 { v.z as u32 } else { 0u32 };
        out[base + 3u64] = if v.w > 0.0f32 { v.w as u32 } else { 0u32 };
    } else {
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
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    cast(self.form, self.wide),
    name = self.form.suffixed("cast"),
    threads = if self.wide { self.len / 4 } else { self.len },
    accepts = cast_accepts,
    source = cast_source,
)]
pub struct Cast {
    pub len: usize,
    /// How the input is stored.
    pub form: &'static dyn Stored,
    pub wide: bool,
}

fn cast_accepts(k: &Cast, input: usize, form: &dyn Stored) -> bool {
    k.form.taken(input, 0, form)
}

fn cast_source(k: &Cast) -> Source {
    let suffix = if k.wide { "_x4" } else { "" };
    let entry = match k.form.name() {
        "f32" => format!("plan_cast{suffix}"),
        name => format!("plan_cast{suffix}_{name}"),
    };
    Source::named(cast_ir(k.form, k.wide), entry)
}

/// One thread per element or quad, integer id to float.
#[kernel]
fn cast_f32(
    x: Buf<U32>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    len: u32,
    #[build] out_form: &'static dyn Stored,
    #[build] wide: bool,
) {
    if wide {
        if gid() >= len / 4 {
            return;
        }
        let base = (gid() * 4) as u64;
        let a = x[base] as f32;
        let b = x[base + 1u64] as f32;
        let c = x[base + 2u64] as f32;
        let d = x[base + 3u64] as f32;
        out.store4(base, F32x4::new(a, b, c, d));
    } else {
        if gid() >= len {
            return;
        }
        out[gid() as u64] = x[gid() as u64] as f32;
    }
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    cast_f32(self.out_form, self.wide),
    threads = if self.wide { self.len / 4 } else { self.len },
    emits = cast_f32_emits,
    source = cast_f32_source,
)]
struct CastF32 {
    len: usize,
    out_form: &'static dyn Stored,
    wide: bool,
}

fn cast_f32_emits(k: &CastF32, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn cast_f32_source(k: &CastF32) -> Source {
    let suffix = if k.wide { "_x4" } else { "" };
    let entry = match k.out_form.name() {
        "f32" => format!("plan_cast_f32{suffix}"),
        name => format!("plan_cast_f32{suffix}_{name}"),
    };
    Source::named(cast_f32_ir(k.out_form, k.wide), entry)
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let x = *node.args.first()?;
    let len = tree.elements(at);
    let plain_float = |f: &dyn Stored| {
        matches!(
            f.dtype(),
            DType::F32 | DType::Bf16 | DType::F16 | DType::Fp8
        )
    };
    match node.term {
        Term::Op(Op::Cast(DType::U32)) => {
            let form = tree.node(x).form;
            let wide = len.is_multiple_of(4) && plain_float(form);
            let cast = Cast { len, form, wide };
            Some(vec![Draft::new(cast, [x], len)])
        }
        Term::Op(Op::Cast(DType::F32)) if tree.node(x).form.dtype() == DType::U32 => {
            let out_form = writable(node.form);
            let wide = len.is_multiple_of(4) && plain_float(out_form);
            let cast = CastF32 {
                len,
                out_form,
                wide,
            };
            Some(vec![Draft::new(cast, [x], len)])
        }
        _ => None,
    }
}

tessera_lower::rule!("cast", propose);
