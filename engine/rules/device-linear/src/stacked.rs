//! Stacked BF16 matmul and optional RoPE tail fusion.

use super::{THREADS, WARPS, WIDE_COLS, WIDE_THREADS, cuda_needs, cuda_source_with, trip};
use std::sync::OnceLock;
use tessera_device::{Backend, flatten_for};
use tessera_graph::{DType, DimExpr, Op, Order, Pairs, Stage, View};
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, Draft, Kernel, Launch, Load, Needs, Ops, Shape, Source, Step, T, Term, Tree,
};

const STACKED: &str = include_str!("stacked.cu");

tessera_lower::knob!("fast_stacked_wide", "tiled64_min_rows", 1, 4096);
tessera_lower::knob!("fast_stacked_wide", "min_blocks", 1, 16);
tessera_lower::knob!("fast_stacked_wide", "w64_min_blocks", 1, 16);
tessera_lower::knob!("fast_stacked_lanes", "tiled64_min_rows", 1, 4096);
tessera_lower::knob!("fast_stacked_lanes", "min_blocks", 1, 16);
tessera_lower::knob!("fast_stacked_lanes", "tc_min_blocks", 1, 16);
tessera_lower::knob!("fast_stacked_lanes", "l64_min_blocks", 1, 16);

pub(crate) fn stacked_source() -> &'static str {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    CUDA.get_or_init(|| flatten_for(Backend::Cuda, "stacked.kernel", STACKED))
}

fn stacked_source_with<'a>(
    entry: &'static str,
    defs: impl IntoIterator<Item = (&'a str, usize)>,
) -> Source {
    tessera_device::cuda_source_with(stacked_source(), entry, defs)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RopeTail {
    width: usize,
    rot_stride: usize,
    rot_off: usize,
    turned: usize,
    adjacent: bool,
    rope_first: bool,
    theta: f32,
}

/// Fast CUDA stacked BF16 matmul, transposed or row-major.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.wide { "fast_stacked_wide" } else { "fast_stacked_lanes" },
    sizes = [
        self.batches, self.rows, self.n, self.k,
        self.x_stride[0], self.x_stride[1], self.o_stride[0], self.o_stride[1],
    ],
    work = self.len * self.k,
    accepts = stacked_accepts,
    emits = super::f32_only,
    needs = stacked_needs(self),
    order = Order::PerConfig,
    source = stacked_source_for(self),
    launches = stacked_launches,
    load = stacked_load,
    load_key = tessera_lower::load_key((
        (self.len, self.batches, self.rows, self.n, self.k, self.block, self.first, self.step),
        (self.x_stride, self.o_stride, self.wide, self.tiled64_min_rows, self.min_blocks),
        (
            self.rope.map(|r| (
                r.width, r.rot_stride, r.rot_off, r.turned, r.adjacent,
                r.rope_first, r.theta.to_bits(),
            )),
            self.form.name(),
        ),
    )),
    unproven,
)]
pub(crate) struct FastStacked {
    pub len: usize,
    pub batches: usize,
    pub rows: usize,
    pub n: usize,
    pub k: usize,
    pub block: usize,
    pub first: usize,
    pub step: usize,
    pub x_stride: [usize; 2],
    pub o_stride: [usize; 2],
    pub wide: bool,
    pub tiled64_min_rows: usize,
    pub min_blocks: Option<u32>,
    pub rope: Option<RopeTail>,
    pub form: &'static dyn Stored,
}

impl FastStacked {
    pub(crate) fn wide_64(&self) -> bool {
        self.wide
            && self.rows >= self.tiled64_min_rows
            && self.n.is_multiple_of(64)
            && self.k.is_multiple_of(32)
    }

    pub(crate) fn lanes_64(&self) -> bool {
        !self.wide
            && self.rows >= self.tiled64_min_rows
            && self.n.is_multiple_of(64)
            && self.k.is_multiple_of(32)
    }

    pub(crate) fn tiled_64(&self) -> bool {
        self.wide_64() || self.lanes_64()
    }

    fn lanes_tc(&self) -> bool {
        !self.wide && self.n.is_multiple_of(16) && self.k.is_multiple_of(128)
    }
}

fn stacked_accepts(k: &FastStacked, input: usize, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == k.form.name(),
        3 => form.name() == "u32",
        _ => form.name() == "f32",
    }
}

pub(crate) fn stacked_needs(k: &FastStacked) -> Needs {
    if k.tiled_64() {
        return Needs {
            bf16_tile: true,
            ..cuda_needs(64 * 18 * 4 * 2 + 64 * 40 * 2)
        };
    }
    if !(k.wide || k.lanes_tc()) {
        return cuda_needs(0);
    }
    let group_bytes = if k.wide {
        16 * 18 * 8 + 32 * 34 * 2
    } else {
        4 * 24 * 36 * 4
    };
    Needs {
        group_threads: WIDE_THREADS,
        bf16_tile: true,
        ..cuda_needs(group_bytes)
    }
}

pub(crate) fn stacked_source_for(k: &FastStacked) -> Source {
    match (k.wide, k.tiled_64(), k.lanes_tc()) {
        (true, true, _) => stacked_source_with(
            "tessera_stacked_wide_64",
            k.min_blocks.map(|v| ("ST_WIDE64_BLK", v as usize)),
        ),
        (false, true, _) => stacked_source_with(
            "tessera_stacked_lanes_64",
            k.min_blocks.map(|v| ("ST_LANES64_BLK", v as usize)),
        ),
        (true, false, _) => stacked_source_with(
            "tessera_stacked_wide",
            k.min_blocks.map(|v| ("ST_WIDE_BLK", v as usize)),
        ),
        (false, false, true) => cuda_source_with(
            "tessera_stacked_lanes_tc",
            k.min_blocks.map(|v| ("ST_TC_BLK", v as usize)),
        ),
        (false, false, false) => cuda_source_with(
            "tessera_stacked_lanes",
            k.min_blocks.map(|v| ("ST_LANES_BLK", v as usize)),
        ),
    }
}

fn stacked_load(k: &FastStacked, _: usize) -> Option<Load> {
    let (tiled, tc) = (k.tiled_64(), k.lanes_tc());
    let rtiles = k.rows.div_ceil(match (tiled, k.wide || tc) {
        (true, _) => 64,
        (false, true) => 16,
        (false, false) => 8,
    });
    let r = k.rows.min(8);
    let (groups, group, trips, mut trip_ops, flight) = if tiled {
        (
            (k.n / 64) * k.batches * rtiles,
            THREADS,
            k.k as f64 / 32.0,
            Ops::default(),
            1024.0,
        )
    } else if tc {
        (
            (k.n / 16) * k.batches * rtiles,
            128,
            k.k as f64 / 128.0,
            Ops::default(),
            1024.0,
        )
    } else if k.wide {
        (
            (k.n / WIDE_COLS) * k.batches * rtiles,
            WIDE_THREADS,
            k.k as f64 / 32.0,
            Ops::default(),
            512.0,
        )
    } else {
        (
            (k.batches * k.n).div_ceil(WARPS) * rtiles,
            THREADS,
            k.k as f64 / 256.0,
            trip(8 * r, 8, 1 + 2 * r, 16 + 32 * r),
            512.0,
        )
    };
    trip_ops.fresh = f64::from(u8::from(tiled || tc));
    let mut reads = vec![
        (2 * k.batches * rtiles * k.n * k.k) as f64,
        (4 * k.batches * k.rows * k.k) as f64,
    ];
    if let Some(r) = k.rope {
        reads.push((4 * k.batches * k.rows * r.width) as f64);
        reads.push((4 * k.rows) as f64);
    }
    Some(Load {
        threads: groups * group,
        group,
        trips,
        trip: trip_ops,
        reads,
        written: (4 * k.len) as f64,
        flight,
        ..Load::default()
    })
}

fn stacked_launches(k: &FastStacked, step: &Step) -> Vec<Launch> {
    let ([xb, xr], [ob, or]) = (k.x_stride, k.o_stride);
    let dims = [k.rows, k.n, k.k, k.block, k.first, k.step, xb, xr, ob, or];
    if k.wide {
        let (across, down, threads) = if k.wide_64() {
            (k.n / 64, k.batches * k.rows.div_ceil(64), THREADS)
        } else {
            (
                k.n / WIDE_COLS,
                k.batches * k.rows.div_ceil(16),
                WIDE_THREADS,
            )
        };
        let (rot, pos, mat_off, rope_out_off, r) = match k.rope {
            Some(r) => {
                let (m_off, r_off) = if r.rope_first { (r.width, 0) } else { (0, k.n) };
                (step.read(2), step.read(3), m_off, r_off, r)
            }
            None => (
                step.read(0),
                step.read(0),
                0,
                0,
                RopeTail {
                    width: 0,
                    rot_stride: 0,
                    rot_off: 0,
                    turned: 0,
                    adjacent: false,
                    rope_first: false,
                    theta: 1.0,
                },
            ),
        };
        let mut args = vec![
            step.read(0),
            step.read(1),
            rot,
            pos,
            step.out(),
            Arg::count(k.batches),
        ];
        args.extend(dims.map(Arg::count));
        args.extend([
            Arg::count(mat_off),
            Arg::count(r.width),
            Arg::count(r.rot_stride),
            Arg::count(r.rot_off),
            Arg::count(rope_out_off),
            Arg::count(r.turned),
            Arg::count(usize::from(r.adjacent)),
            Arg::value(&r.theta),
        ]);
        let shape = Shape::Grid {
            across,
            down,
            threads,
        };
        return vec![Launch { args, shape }];
    }
    let tc = k.lanes_tc() || k.lanes_64();
    let bn = k.batches * k.n;
    let (across, down, threads) = if k.lanes_64() {
        (k.n / 64, k.batches * k.rows.div_ceil(64), THREADS)
    } else if k.lanes_tc() {
        (k.n / 16, k.batches * k.rows.div_ceil(16), WIDE_THREADS)
    } else {
        (bn.div_ceil(WARPS), k.rows.div_ceil(8), THREADS)
    };
    let shape = Shape::Grid {
        across,
        down,
        threads,
    };
    let mut args = vec![step.read(0), step.read(1), step.out()];
    args.extend((!tc).then(|| Arg::count(bn)));
    args.extend(dims.map(Arg::count));
    vec![Launch { args, shape }]
}

fn peel_input(tree: &Tree, x: T, batches: usize, rows: usize, k: usize) -> (T, [usize; 2]) {
    let is_swap = |t: T| matches!(&tree.node(t).term, Term::Op(Op::View(View::Permute(p))) if p[..] == [1, 0, 2]);
    if !is_swap(x) || tree.root(x) || tree.node(tree.node(x).args[0]).form.name() != "f32" {
        return (x, [rows * k, k]);
    }
    let inner = tree.node(x).args[0];
    let slice = matches!(
        &tree.node(inner).term,
        Term::Op(Op::View(View::Slice {
            axis: 2,
            start: DimExpr::Const(0),
            ..
        }))
    );
    if slice && !tree.root(inner) {
        let base = tree.node(inner).args[0];
        if let &[r, b, fk] = tree.node(base).shape.as_slice()
            && (r, b) == (rows, batches)
            && fk.is_multiple_of(4)
            && tree.node(base).form.name() == "f32"
        {
            return (base, [fk, batches * fk]);
        }
    }
    (inner, [k, batches * k])
}

/// Peels an optional single-reader last-axis slice from `t`.
fn peel_rot_slice(tree: &Tree, t: T) -> Option<(T, usize, usize)> {
    if tree.root(t) {
        return None;
    }
    let n = tree.node(t);
    let Term::Op(Op::View(View::Slice {
        axis: 2,
        start: off,
        ..
    })) = &n.term
    else {
        return None;
    };
    let parent = n.args[0];
    if tree.node(parent).form.name() != "f32" {
        return None;
    }
    let off = off.value(&tree.g.dims, &|d| tree.size(d))? as usize;
    let stride = *tree.node(parent).shape.last()?;
    Some((parent, off, stride))
}

/// Matches `concat(permute(matmul), rope)` or `concat(rope, permute(matmul))`.
fn match_rope_concat(tree: &Tree, at: T) -> Option<(T, RopeTail, T, T)> {
    let outer = tree.node(at);
    if outer.term != Term::Op(Op::View(View::Concat)) || outer.form.name() != "f32" {
        return None;
    }
    let [a, b] = outer.args[..] else {
        return None;
    };
    let is_rope = |t: T| matches!(tree.node(t).term, Term::Stage(Stage::Rope { .. }));
    let (plain_t, rope_t, rope_first) = match (is_rope(a), is_rope(b)) {
        (false, true) => (a, b, false),
        (true, false) => (b, a, true),
        _ => return None,
    };
    if tree.root(plain_t) || tree.root(rope_t) {
        return None;
    }
    let perm = tree.node(plain_t);
    let Term::Op(Op::View(View::Permute(order))) = &perm.term else {
        return None;
    };
    if order[..] != [1, 0, 2] || perm.form.name() != "f32" {
        return None;
    }
    let mat = perm.args[0];
    if tree.root(mat) || tree.node(mat).term != Term::Op(Op::Matmul) {
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
    if rope_node.form.name() != "f32"
        || tree.node(rot_in).form.name() != "f32"
        || tree.node(pos_t).form.dtype() != DType::U32
    {
        return None;
    }
    let &[r_rows, r_heads, rope_w] = rope_node.shape.as_slice() else {
        return None;
    };
    if rope_w == 0
        || !rope_w.is_multiple_of(2)
        || tree.elements(pos_t) != r_rows
        || perm.shape[..2] != [r_rows, r_heads]
    {
        return None;
    }
    let (rot_src, rot_off, rot_stride) =
        peel_rot_slice(tree, rot_in).unwrap_or((rot_in, 0, rope_w));
    let tail = RopeTail {
        width: rope_w,
        rot_stride,
        rot_off,
        turned: turned.map_or(rope_w / 2, |t| t as usize),
        adjacent: pairs == Pairs::Adjacent,
        rope_first,
        theta,
    };
    Some((mat, tail, rot_src, pos_t))
}

pub(crate) fn check_stacked(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    let outer = tree.node(at);
    let (node, out_swap, rope_match) =
        if let Some((mat, tail, rot, pos)) = match_rope_concat(tree, at) {
            (tree.node(mat), true, Some((tail, rot, pos)))
        } else {
            match &outer.term {
                Term::Op(Op::View(View::Permute(p))) if p[..] == [1, 0, 2] => {
                    let inner = outer.args[0];
                    if tree.root(inner) || tree.node(inner).term != Term::Op(Op::Matmul) {
                        return None;
                    }
                    (tree.node(inner), true, None)
                }
                Term::Op(Op::Matmul) => (outer, false, None),
                _ => return None,
            }
        };
    let (x, mut w) = (node.args[0], node.args[1]);
    let (&[batches, n, k], &[xb, rows, xk]) =
        (tree.node(w).shape.as_slice(), tree.node(x).shape.as_slice())
    else {
        return None;
    };
    if xb != batches || xk != k || rows == 0 {
        return None;
    }
    if [outer.form, node.form, tree.node(x).form]
        .iter()
        .any(|f| f.name() != "f32")
    {
        return Some(Err("stacked requires f32 input and output".into()));
    }
    let wide =
        matches!(&tree.node(w).term, Term::Op(Op::View(View::Permute(p))) if p[..] == [0, 2, 1]);
    if rope_match.is_some() && !wide {
        return None;
    }
    let (x, x_stride) = peel_input(tree, x, batches, rows, k);
    let out_w = n + rope_match.map_or(0, |(r, _, _)| r.width);
    let o_stride = if out_swap {
        [out_w, batches * out_w]
    } else {
        [rows * out_w, out_w]
    };
    w = if wide { tree.node(w).args[0] } else { w };
    let mut first_row = 0usize;
    if let Term::Op(Op::View(View::Slice { axis: 1, start, .. })) = &tree.node(w).term {
        let Some(offset) = start.value(&tree.g.dims, &|d| tree.size(d)) else {
            return Some(Err("slice start is dynamic".into()));
        };
        (first_row, w) = (offset as usize, tree.node(w).args[0]);
    }
    if !matches!(&tree.node(w).term, Term::Op(Op::View(View::Reshape(_)))) {
        return Some(Err("weight is not a reshaped 2D leaf".into()));
    }
    let leaf = tree.node(w).args[0];
    let (&[_, block_rows, width], &[_, stored_w]) = (
        tree.node(w).shape.as_slice(),
        tree.node(leaf).shape.as_slice(),
    ) else {
        return Some(Err("reshaped weight is not 3D over 2D leaf".into()));
    };
    let form = tree.node(leaf).form;
    if !tree.leaf(leaf) || stored_w != width || form.name() != "bf16" {
        return Some(Err(
            "stacked requires direct bf16 leaf with matching width".into()
        ));
    }
    let (block, first) = (block_rows * width, first_row * width);
    if ![block, first, width].iter().all(|v| v.is_multiple_of(8)) {
        return Some(Err(format!(
            "strides ({width}, {block}, {first}) not multiple of 8"
        )));
    }
    let ok = if wide {
        n > 0 && n.is_multiple_of(WIDE_COLS) && k > 0 && k.is_multiple_of(32)
    } else {
        n > 0 && k > 0 && k.is_multiple_of(8)
    };
    if !ok {
        return Some(Err(format!(
            "stacked shape n={n} k={k} wide={wide} unsupported"
        )));
    }
    let (rope, reads) = match rope_match {
        Some((tail, rot, pos)) => (Some(tail), vec![x, leaf, rot, pos]),
        None => (None, vec![x, leaf]),
    };
    let len = batches * rows * out_w;
    let rule = if wide {
        "fast_stacked_wide"
    } else {
        "fast_stacked_lanes"
    };
    let tiled64_min_rows = tree.knobs.get(rule, "tiled64_min_rows").unwrap_or(32);
    let mut k = FastStacked {
        len,
        batches,
        rows,
        n,
        k,
        block,
        first,
        step: width,
        x_stride,
        o_stride,
        wide,
        tiled64_min_rows,
        min_blocks: None,
        rope,
        form,
    };
    let blk_knob = match (wide, k.tiled_64(), k.lanes_tc()) {
        (true, true, _) => "w64_min_blocks",
        (false, true, _) => "l64_min_blocks",
        (false, false, true) => "tc_min_blocks",
        _ => "min_blocks",
    };
    k.min_blocks = tree
        .knobs
        .get(rule, blk_knob)
        .or_else(|| tree.knobs.get(rule, "min_blocks"))
        .map(|v| v as u32);
    Some(Ok(vec![Draft::new(k, reads, len)]))
}
