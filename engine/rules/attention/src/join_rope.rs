//! Fused partial-RoPE and concat for MLA queries and partial heads.

use crate::kv_prep::{rope_angle, rope_angle_inline};
use tessera_codegen::prelude::*;
use tessera_graph::{DType, Op, Pairs, Stage, View};
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::{Arg, Draft, Home, Kernel, Launch, Shape, Source, Step, T, Term, Tree};

/// Elements one wide load brings per thread.
const WIDE: u32 = 4;

/// Copies a plain slice and rotates a rotary slice.
#[kernel]
fn join_rope(
    #[form(plain_form)] plain_buf: Buf<Dyn>,
    #[form(rot_form)] rot_buf: Buf<Dyn>,
    pos: Buf<U32>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    positions: u32,
    heads: u32,
    plain_w: u32,
    rope_w: u32,
    plain_stride: u32,
    plain_off: u32,
    rot_stride: u32,
    rot_off: u32,
    turned: u32,
    theta: f32,
    #[build] wide: bool,
    #[build] adjacent: bool,
    #[build] plain_swap: bool,
    #[build] rope_first: bool,
    #[build] plain_form: &'static dyn Stored,
    #[build] rot_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    let plain_tasks = if wide { plain_w / WIDE } else { plain_w };
    let pairs = rope_w / 2;
    let tasks = plain_tasks + pairs;
    if gid() >= positions * heads * tasks {
        return;
    }
    let task = gid() % tasks;
    let row_head = gid() / tasks;
    let h = row_head % heads;
    let p = row_head / heads;
    let out_base = row_head as u64 * (plain_w + rope_w) as u64;
    if task < plain_tasks {
        let plain_row = if plain_swap {
            h * positions + p
        } else {
            row_head
        };
        let src_at = plain_row as u64 * plain_stride as u64 + plain_off as u64;
        let out_off = if rope_first { rope_w as u64 } else { 0u64 };
        let dst_at = out_base + out_off;
        if wide {
            let k = (task * WIDE) as u64;
            let v = plain_buf.load4(src_at + k);
            out.store4(dst_at + k, v);
        } else {
            out[dst_at + task as u64] = plain_buf[src_at + task as u64];
        }
    } else {
        let pair = task - plain_tasks;
        let angle = rope_angle(pos[p as u64], pair, rope_w, turned, theta);
        let c = cos(angle);
        let sn = sin(angle);
        let a = if adjacent { 2 * pair } else { pair };
        let b = if adjacent { a + 1 } else { pairs + pair };
        let src_at = row_head as u64 * rot_stride as u64 + rot_off as u64;
        let out_off = if rope_first { 0u64 } else { plain_w as u64 };
        let dst_at = out_base + out_off;
        let x0 = rot_buf[src_at + a as u64];
        let x1 = rot_buf[src_at + b as u64];
        out[dst_at + a as u64] = x0 * c - x1 * sn;
        out[dst_at + b as u64] = x0 * sn + x1 * c;
    }
}

/// Fused concat of an unrotated slice and a rotated slice.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "join_rope",
    sizes = [self.positions, self.heads, self.plain_w, self.rope_w],
    work = self.positions * self.heads * (self.plain_w + self.rope_w),
    accepts = jr_accepts,
    emits = jr_emits,
    source = jr_source,
    launches = jr_launches,
    shape = Shape::Line(jr_threads(self)),
)]
pub(crate) struct JoinRope {
    pub positions: usize,
    pub heads: usize,
    pub plain_w: usize,
    pub rope_w: usize,
    pub plain_stride: usize,
    pub plain_off: usize,
    pub rot_stride: usize,
    pub rot_off: usize,
    pub turned: usize,
    pub theta: f32,
    pub wide: bool,
    pub adjacent: bool,
    pub plain_swap: bool,
    pub rope_first: bool,
    pub plain_form: &'static dyn Stored,
    pub rot_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

fn jr_accepts(k: &JoinRope, input: usize, form: &dyn Stored) -> bool {
    crate::pair_accepts([k.plain_form, k.rot_form], input, form)
}

fn jr_emits(k: &JoinRope, form: &dyn Stored) -> bool {
    form.name() == k.out_form.name()
}

fn jr_source(k: &JoinRope) -> Source {
    let body = join_rope_ir(
        k.wide,
        k.adjacent,
        k.plain_swap,
        k.rope_first,
        k.plain_form,
        k.rot_form,
        k.out_form,
    );
    let flags = usize::from(k.wide)
        | (usize::from(k.adjacent) << 1)
        | (usize::from(k.plain_swap) << 2)
        | (usize::from(k.rope_first) << 3);
    let entry = format!(
        "plan_join_rope_{flags}_{}_{}_{}",
        k.plain_form.name(),
        k.rot_form.name(),
        k.out_form.name(),
    );
    Source::named(body, entry)
}

fn jr_threads(k: &JoinRope) -> usize {
    let plain_tasks = if k.wide {
        k.plain_w / WIDE as usize
    } else {
        k.plain_w
    };
    k.positions * k.heads * (plain_tasks + k.rope_w / 2)
}

fn jr_launches(k: &JoinRope, step: &Step) -> Vec<Launch> {
    let mut args = vec![step.read(0), step.read(1), step.read(2), step.out()];
    let counts = [
        k.positions,
        k.heads,
        k.plain_w,
        k.rope_w,
        k.plain_stride,
        k.plain_off,
        k.rot_stride,
        k.rot_off,
        k.turned,
    ];
    args.extend(counts.map(Arg::count));
    args.push(Arg::value(&k.theta));
    let shape = k.shape().expect("a line");
    vec![Launch { args, shape }]
}

/// Whether `t` is a non-weight, non-append float tensor.
fn plain_float(tree: &Tree, t: T) -> bool {
    let n = tree.node(t);
    n.form.dtype().is_float()
        && !matches!(
            n.term,
            Term::Leaf(Home::Weight(_) | Home::Part(..)) | Term::Op(Op::Append(_))
        )
}

/// Peels a single-reader last-axis slice when present.
fn peel_slice(tree: &Tree, t: T) -> (T, usize, usize) {
    let width = *tree.node(t).shape.last().unwrap_or(&0);
    if tree.root(t) {
        return (t, width, 0);
    }
    let Term::Op(Op::View(View::Slice { axis, start, .. })) = &tree.node(t).term else {
        return (t, width, 0);
    };
    let parent = tree.node(t).args[0];
    let rank = tree.node(parent).shape.len();
    if *axis as usize + 1 != rank || !plain_float(tree, parent) {
        return (t, width, 0);
    }
    let Some(off) = start.value(&tree.g.dims, &|n| tree.size(n)) else {
        return (t, width, 0);
    };
    let stride = *tree.node(parent).shape.last().unwrap_or(&0);
    (parent, stride, off as usize)
}

/// Unwraps the plain side: slice, swap permute, or direct.
fn peel_plain(tree: &Tree, t: T, positions: usize, heads: usize) -> (T, usize, usize, bool) {
    let width = *tree.node(t).shape.last().unwrap_or(&0);
    let node = tree.node(t);
    if !tree.root(t)
        && let Term::Op(Op::View(View::Permute(order))) = &node.term
        && order[..] == [1, 0, 2]
        && node.shape[..] == [positions, heads, width]
        && plain_float(tree, node.args[0])
    {
        return (node.args[0], width, 0, true);
    }
    let (src, stride, off) = peel_slice(tree, t);
    (src, stride, off, false)
}

/// Matches `concat(plain, rope)` or `concat(rope, plain)`.
fn find_site(tree: &Tree, at: T) -> Option<(JoinRope, [T; 3])> {
    let node = tree.node(at);
    if node.term != Term::Op(Op::View(View::Concat)) || !node.form.dtype().is_float() {
        return None;
    }
    let [a, b] = node.args[..] else {
        return None;
    };
    if !plain_float(tree, a) || !plain_float(tree, b) {
        return None;
    }
    let is_rope = |t: T| matches!(tree.node(t).term, Term::Stage(Stage::Rope { .. }));
    let (plain_t, rope_t, rope_first) = match (is_rope(a), is_rope(b)) {
        (false, true) => (a, b, false),
        (true, false) => (b, a, true),
        _ => return None,
    };
    if tree.root(rope_t) {
        return None;
    }
    let rope_node = tree.node(rope_t);
    let Term::Stage(Stage::Rope {
        theta,
        pairs,
        turned,
    }) = rope_node.term
    else {
        return None;
    };
    let [rot_in, pos_t] = rope_node.args[..] else {
        return None;
    };
    if tree.node(pos_t).form.dtype() != DType::U32 {
        return None;
    }
    let plain_w = *tree.node(plain_t).shape.last()?;
    let rope_w = *rope_node.shape.last()?;
    if plain_w == 0 || rope_w == 0 || !rope_w.is_multiple_of(2) {
        return None;
    }
    let positions = tree.elements(pos_t);
    let total_rows = tree.elements(rope_t) / rope_w;
    if positions == 0 || total_rows == 0 || !total_rows.is_multiple_of(positions) {
        return None;
    }
    let heads = total_rows / positions;
    let (rot_src, rot_stride, rot_off) = peel_slice(tree, rot_in);
    let (plain_src, plain_stride, plain_off, plain_swap) =
        peel_plain(tree, plain_t, positions, heads);
    let w = WIDE as usize;
    let wide = plain_w.is_multiple_of(w)
        && plain_stride.is_multiple_of(w)
        && plain_off.is_multiple_of(w)
        && rope_w.is_multiple_of(w);
    let kernel = JoinRope {
        positions,
        heads,
        plain_w,
        rope_w,
        plain_stride,
        plain_off,
        rot_stride,
        rot_off,
        turned: turned.map_or(rope_w / 2, |t| t as usize),
        theta,
        wide,
        adjacent: pairs == Pairs::Adjacent,
        plain_swap,
        rope_first,
        plain_form: plain(tree.node(plain_src).form),
        rot_form: plain(tree.node(rot_src).form),
        out_form: writable(node.form),
    };
    Some((kernel, [plain_src, rot_src, pos_t]))
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (kernel, reads) = find_site(tree, at)?;
    Some(vec![Draft::new(kernel, reads, tree.elements(at))])
}

tessera_lower::rule!("join_rope", propose);
