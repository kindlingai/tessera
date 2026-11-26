//! Fused MLA KV-prep and kept-cache scatter in one launch.

use tessera_codegen::kept::{self, Layout};
use tessera_codegen::prelude::*;
use tessera_graph::{DType, Op, Order, Pairs, Stage, View};
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::{Arg, Draft, Kernel, Launch, Needs, Shape, Source, Step, T, Term, Tree};

/// Elements one wide load brings per lane.
const WIDE: u32 = 4;

/// Rotation angle for one rotary pair at `pos`.
#[kernel]
pub(crate) fn rope_angle(pos: u32, pair: u32, width: u32, turned: u32, theta: f32) -> f32 {
    let turn = pow(theta, -((2 * pair) as f32) / width as f32);
    if pair < turned {
        pos as f32 * turn
    } else {
        0.0
    }
}

/// RMS-norms `0..latent`, rotates `latent..latent+rope`, scatters into kept.
#[kernel]
fn kv_scatter(
    #[form(x_form)] x: Buf<Dyn>,
    #[form(w_form)] w: Buf<Dyn>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    rows: u32,
    latent: u32,
    rope: u32,
    turned: u32,
    eps: f32,
    theta: f32,
    #[build] lanes: bool,
    #[build] adjacent: bool,
    #[build] x_form: &'static dyn Stored,
    #[build] w_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    let row = if lanes { gid() / LANES as u32 } else { gid() };
    if row >= rows {
        return;
    }
    let width = latent + rope;
    let in_at = row as u64 * width as u64;
    let slot = slots[row as u64];
    let pos = positions[row as u64];
    let out_at = kept_row(slot, pos) as u64 * width as u64;
    if lanes {
        let l = lane();
        let mut acc = 0.0f32;
        for i in (l * WIDE..latent).by(LANES as u32 * WIDE) {
            let v = x.load4(in_at + i as u64);
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        acc = sum32(acc);
        let inv = rsqrt(acc / latent as f32 + eps);
        for i in (l * WIDE..latent).by(LANES as u32 * WIDE) {
            let v = x.load4(in_at + i as u64);
            let g = w.load4(i as u64);
            let mut o = F32x4::zero();
            o.x = v.x * inv * g.x;
            o.y = v.y * inv * g.y;
            o.z = v.z * inv * g.z;
            o.w = v.w * inv * g.w;
            out.store4(out_at + i as u64, o);
        }
    } else {
        let mut acc = 0.0f32;
        for i in 0..latent {
            let v = x[in_at + i as u64];
            acc += v * v;
        }
        let inv = rsqrt(acc / latent as f32 + eps);
        for i in 0..latent {
            out[out_at + i as u64] = x[in_at + i as u64] * inv * w[i as u64];
        }
    }
    let pairs = rope / 2;
    let rope_in = in_at + latent as u64;
    let rope_out = out_at + latent as u64;
    let first = if lanes { lane() } else { 0u32 };
    let step = if lanes { LANES as u32 } else { 1u32 };
    for p in (first..pairs).by(step) {
        let angle = rope_angle(pos, p, rope, turned, theta);
        let c = cos(angle);
        let sn = sin(angle);
        let a = if adjacent { 2 * p } else { p };
        let b = if adjacent { a + 1 } else { pairs + p };
        let x0 = x[rope_in + a as u64];
        let x1 = x[rope_in + b as u64];
        out[rope_out + a as u64] = x0 * c - x1 * sn;
        out[rope_out + b as u64] = x0 * sn + x1 * c;
    }
}

/// Fused MLA KV norm, rotary key, and kept scatter.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.lanes { "mla_kv_scatter" } else { "mla_kv_scatter_plain" },
    order = if self.lanes { Order::PerConfig } else { Order::Fixed },
    sizes = [self.rows, self.latent, self.rope],
    work = self.rows * (self.latent + self.rope + usize::from(!self.lanes)),
    needs = if self.lanes { Needs::lanes() } else { Needs::default() },
    accepts = kv_accepts,
    emits = kv_emits,
    source = kv_source,
    launches = kv_launches,
    shape = Shape::Line(if self.lanes { self.rows * LANES } else { self.rows }),
)]
pub(crate) struct KvScatter {
    pub rows: usize,
    pub latent: usize,
    pub rope: usize,
    pub turned: usize,
    pub eps: f32,
    pub theta: f32,
    pub slots: usize,
    pub context: usize,
    pub lanes: bool,
    pub adjacent: bool,
    pub kept: &'static dyn Layout,
    pub x_form: &'static dyn Stored,
    pub w_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

fn kv_accepts(k: &KvScatter, input: usize, form: &dyn Stored) -> bool {
    crate::pair_accepts([k.x_form, k.w_form], input, form)
}

fn kv_emits(k: &KvScatter, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn kv_source(k: &KvScatter) -> Source {
    let body = kept::attach(
        kv_scatter_ir(k.lanes, k.adjacent, k.x_form, k.w_form, k.out_form),
        k.kept,
    );
    let mode = if k.lanes { "l" } else { "p" };
    let pair = if k.adjacent { "a" } else { "h" };
    let entry = format!(
        "plan_mla_kv_{mode}_{pair}_{}_{}_{}_{}",
        k.kept.name(),
        k.x_form.name(),
        k.w_form.name(),
        k.out_form.name(),
    );
    Source::named(body, entry)
}

fn kv_launches(k: &KvScatter, step: &Step) -> Vec<Launch> {
    let mut args = vec![
        step.read(0),
        step.read(1),
        step.read(2),
        step.read(3),
        step.out(),
    ];
    args.extend(step.kept(k.kept));
    args.extend([k.rows, k.latent, k.rope, k.turned].map(Arg::count));
    args.push(Arg::value(&k.eps));
    args.push(Arg::value(&k.theta));
    let kept = k.kept.values(k.slots, k.context);
    args.extend(kept.into_iter().map(|v| Arg::count(v as usize)));
    let shape = k.shape().expect("a line");
    vec![Launch { args, shape }]
}

/// Checks a last-axis slice of `parent` starting at `want_start`.
fn last_slice(tree: &Tree, t: T, want_start: usize) -> Option<(T, usize)> {
    let node = tree.node(t);
    let Term::Op(Op::View(View::Slice { axis, start, .. })) = &node.term else {
        return None;
    };
    let parent = node.args[0];
    let rank = tree.node(parent).shape.len();
    if *axis as usize + 1 != rank {
        return None;
    }
    let off = start.value(&tree.g.dims, &|n| tree.size(n))? as usize;
    if off != want_start {
        return None;
    }
    let width = *node.shape.last()?;
    Some((parent, width))
}

/// Matches MLA KV-prep concatenated and appended to a kept cache.
fn find_site(tree: &Tree, at: T) -> Option<(KvScatter, [T; 4])> {
    let node = tree.node(at);
    let Term::Op(Op::Append(index)) = node.term else {
        return None;
    };
    let (_, context) = tree.kept_rows(index)?;
    let [mut src, slots, positions] = node.args[..] else {
        return None;
    };
    if tree.node(slots).form.dtype() != DType::U32
        || tree.node(positions).form.dtype() != DType::U32
    {
        return None;
    }
    while matches!(tree.node(src).term, Term::Op(Op::View(View::Reshape(_)))) {
        src = tree.node(src).args[0];
    }
    let joined = tree.node(src);
    if joined.term != Term::Op(Op::View(View::Concat)) {
        return None;
    }
    let [left_t, right_t] = joined.args[..] else {
        return None;
    };
    let left = tree.node(left_t);
    let Term::Stage(Stage::RmsNorm { eps }) = left.term else {
        return None;
    };
    let [left_slice, norm_w] = left.args[..] else {
        return None;
    };
    let (kv_src, latent) = last_slice(tree, left_slice, 0)?;
    let right = tree.node(right_t);
    let Term::Stage(Stage::Rope {
        theta,
        pairs,
        turned,
    }) = right.term
    else {
        return None;
    };
    let [right_slice, rope_pos] = right.args[..] else {
        return None;
    };
    if rope_pos != positions {
        return None;
    }
    let (kv_src2, rope) = last_slice(tree, right_slice, latent)?;
    if kv_src != kv_src2 || latent == 0 || rope == 0 || !rope.is_multiple_of(2) {
        return None;
    }
    let width = latent + rope;
    let kv_node = tree.node(kv_src);
    if kv_node.shape.last() != Some(&width)
        || node.shape[2..].iter().product::<usize>() != width
        || !kv_node.form.dtype().is_float()
        || !tree.node(norm_w).form.dtype().is_float()
        || !node.form.dtype().is_float()
    {
        return None;
    }
    let rows = tree.elements(kv_src) / width;
    if rows == 0 || tree.elements(slots) != rows || tree.elements(positions) != rows {
        return None;
    }
    let kv = KvScatter {
        rows,
        latent,
        rope,
        turned: turned.map_or(rope / 2, |t| t as usize),
        eps,
        theta,
        slots: node.shape[0],
        context,
        lanes: true,
        adjacent: pairs == Pairs::Adjacent,
        kept: tree.kept,
        x_form: plain(kv_node.form),
        w_form: plain(tree.node(norm_w).form),
        out_form: writable(node.form),
    };
    Some((kv, [kv_src, norm_w, slots, positions]))
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (kv, reads) = find_site(tree, at)?;
    let out = kv.rows * (kv.latent + kv.rope);
    let mut drafts = Vec::new();
    let aligned = kv.latent.is_multiple_of(WIDE as usize) && kv.rope.is_multiple_of(WIDE as usize);
    if aligned {
        drafts.push(Draft::new(kv, reads, out).choice(0));
    }
    let plain = KvScatter { lanes: false, ..kv };
    drafts.push(Draft::new(plain, reads, out).choice(1));
    Some(drafts)
}

tessera_lower::rule!("mla_kv_scatter", propose);
