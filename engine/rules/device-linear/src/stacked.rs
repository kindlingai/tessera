//! Stacked BF16 matmul and optional RoPE tail fusion.

use super::{
    THREADS, WARPS, WIDE_COLS, WIDE_THREADS, cuda_module, cuda_needs, cuda_source_with, io_defs,
    trip, x_accepts,
};
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

cuda_module!(pub(crate) stacked_source, stacked_source_with, "stacked.kernel", STACKED);

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
    emits = stacked_emits,
    needs = stacked_needs(self),
    order = Order::PerConfig,
    source = stacked_source_for(self),
    launches = stacked_launches,
    load = stacked_load,
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
    pub x_bf16: bool,
    pub y_bf16: bool,
    pub o_bf16: bool,
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
        0 => x_accepts(k.x_bf16, form),
        1 => form.name() == k.form.name(),
        2 => x_accepts(k.o_bf16, form),
        3 => form.name() == "u32",
        _ => form.name() == "f32",
    }
}

fn stacked_emits(k: &FastStacked, form: &dyn Stored) -> bool {
    x_accepts(k.y_bf16, form)
}

pub(crate) fn stacked_needs(k: &FastStacked) -> Needs {
    let xp = if k.x_bf16 { 1 } else { 2 };
    if k.tiled_64() {
        return Needs {
            bf16_tile: true,
            ..cuda_needs(64 * 18 * 4 * xp + 64 * 40 * 2)
        };
    }
    if !(k.wide || k.lanes_tc()) {
        return cuda_needs(0);
    }
    let group_bytes = if k.wide {
        16 * 18 * 4 * xp + 32 * 34 * 2
    } else {
        (xp * 2 * 16 * 36 + 4 * 8 * 36) * 4
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
            io_defs(
                k.min_blocks.map(|v| ("ST_WIDE64_BLK", v as usize)),
                k.x_bf16,
                k.y_bf16,
                k.o_bf16,
            ),
        ),
        (false, true, _) => stacked_source_with(
            "tessera_stacked_lanes_64",
            io_defs(
                k.min_blocks.map(|v| ("ST_LANES64_BLK", v as usize)),
                k.x_bf16,
                k.y_bf16,
                false,
            ),
        ),
        (true, false, _) => stacked_source_with(
            "tessera_stacked_wide",
            io_defs(
                k.min_blocks.map(|v| ("ST_WIDE_BLK", v as usize)),
                k.x_bf16,
                k.y_bf16,
                k.o_bf16,
            ),
        ),
        (false, false, true) => cuda_source_with(
            "tessera_stacked_lanes_tc",
            io_defs(
                k.min_blocks.map(|v| ("ST_TC_BLK", v as usize)),
                k.x_bf16,
                k.y_bf16,
                false,
            ),
        ),
        (false, false, false) => cuda_source_with(
            "tessera_stacked_lanes",
            io_defs(
                k.min_blocks.map(|v| ("ST_LANES_BLK", v as usize)),
                k.x_bf16,
                k.y_bf16,
                false,
            ),
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
    let x_elem = if k.x_bf16 { 2 } else { 4 };
    let o_elem = if k.o_bf16 { 2 } else { 4 };
    let y_elem = if k.y_bf16 { 2 } else { 4 };
    let mut reads = vec![
        (2 * k.batches * rtiles * k.n * k.k) as f64,
        (x_elem * k.batches * k.rows * k.k) as f64,
    ];
    if let Some(r) = k.rope {
        reads.push((o_elem * k.batches * k.rows * r.width) as f64);
        reads.push((4 * k.rows) as f64);
    }
    Some(Load {
        threads: groups * group,
        group,
        trips,
        trip: trip_ops,
        reads,
        written: (y_elem * k.len) as f64,
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
    let want = tree.node(x).form.name();
    let is_swap = |t: T| matches!(&tree.node(t).term, Term::Op(Op::View(View::Permute(p))) if p[..] == [1, 0, 2]);
    if !is_swap(x) || tree.root(x) || tree.node(tree.node(x).args[0]).form.name() != want {
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
            && tree.node(base).form.name() == want
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
    if tree.node(parent).form.name() != n.form.name() {
        return None;
    }
    let off = off.value(&tree.g.dims, &|d| tree.size(d))? as usize;
    let stride = *tree.node(parent).shape.last()?;
    Some((parent, off, stride))
}

/// Matches `concat(permute(matmul), rope)` or `concat(rope, permute(matmul))`.
fn match_rope_concat(tree: &Tree, at: T) -> Option<(T, RopeTail, T, T)> {
    let plain = |f: &dyn Stored| matches!(f.name(), "f32" | "bf16");
    let outer = tree.node(at);
    if outer.term != Term::Op(Op::View(View::Concat)) || !plain(outer.form) {
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
    if order[..] != [1, 0, 2] || !plain(perm.form) {
        return None;
    }
    let mat = perm.args[0];
    if tree.root(mat) || tree.node(mat).term != Term::Op(Op::Matmul) {
        return None;
    }
    let rope = tree.node(rope_t);
    let (
        &Term::Stage(Stage::Rope {
            theta,
            pairs,
            turned,
        }),
        &[rot_in, pos_t],
        &[r_rows, r_heads, rope_w],
    ) = (&rope.term, rope.args.as_slice(), rope.shape.as_slice())
    else {
        return None;
    };
    let ok = plain(rope.form)
        && plain(tree.node(rot_in).form)
        && tree.node(pos_t).form.dtype() == DType::U32
        && rope_w > 0
        && rope_w.is_multiple_of(2)
        && tree.elements(pos_t) == r_rows
        && perm.shape[..2] == [r_rows, r_heads];
    if !ok {
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
    if !matches!(outer.form.name(), "f32" | "bf16") || !matches!(node.form.name(), "f32" | "bf16") {
        return Some(Err("stacked requires f32 or bf16 output".into()));
    }
    let y_bf16 = outer.form.name() == "bf16";
    let x_name = tree.node(x).form.name();
    let x_bf16 = match x_name {
        "f32" => false,
        "bf16" => true,
        _ => {
            return Some(Err(format!(
                "stacked input form {x_name} (requires f32 or bf16)"
            )));
        }
    };
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
    let (rope, o_bf16, reads) = match rope_match {
        Some((tail, rot, pos)) => (
            Some(tail),
            tree.node(rot).form.name() == "bf16",
            vec![x, leaf, rot, pos],
        ),
        None => (None, false, vec![x, leaf]),
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
        x_bf16,
        y_bf16,
        o_bf16,
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
