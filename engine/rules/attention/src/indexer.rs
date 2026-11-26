//! Fused Lightning Indexer scoring over kept or whole keys.

use tessera_codegen::kept::{self, Dense, Layout};
use tessera_codegen::prelude::*;
use tessera_graph::{Act, DType, Mask, Op, Order, Select, Stage, View};
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::{
    Arg, Draft, Extent, Kernel, Launch, Needs, Shape, Source, Step, T, Term, Tree,
};

/// Elements one wide load brings per lane.
const WIDE: u32 = 4;

/// SIMD groups in one warp-parallel block.
const WARPS: u32 = 8;

/// Threads in one warp-parallel block.
const THREADS: u32 = WARPS * LANES as u32;

/// Widest key dimension for register-held warp scoring.
const MAX_DIM: usize = 256;

/// Per-lane register slots for `MAX_DIM` floats.
const MAX_SLOTS: usize = MAX_DIM / LANES;

/// Unroll bound for per-lane wide-load loops.
const MAX_QUADS: u32 = (MAX_SLOTS / WIDE as usize) as u32;

/// Cap on active warps so `gid` fits `u32`.
const MAX_WARPS: usize = 1 << 20;

/// Scores one query row against one key row per warp.
#[kernel]
fn score_warp(
    #[form(q_form)] q: Buf<Dyn>,
    #[form(k_form)] k: Buf<Dyn>,
    #[form(w_form)] weights: Buf<Dyn>,
    row: u32,
    k_at: u64,
    heads: u32,
    #[build] dim: u32,
    #[build] q_form: &'static dyn Stored,
    #[build] k_form: &'static dyn Stored,
    #[build] w_form: &'static dyn Stored,
) -> f32 {
    let l = lane();
    let quads = (dim + LANES as u32 * WIDE - 1) / (LANES as u32 * WIDE);
    let mut kr = [0.0f32; MAX_SLOTS];
    for s in (0..quads).unroll(MAX_QUADS) {
        let d = s * (LANES as u32 * WIDE) + l * WIDE;
        let b = (s * WIDE) as usize;
        if d < dim {
            let kv = k.load4(k_at + d as u64);
            kr[b] = kv.x;
            kr[b + 1] = kv.y;
            kr[b + 2] = kv.z;
            kr[b + 3] = kv.w;
        }
    }
    let w_at = row as u64 * heads as u64;
    let mut score = 0.0f32;
    for h in 0..heads {
        let q_at = (row * heads + h) as u64 * dim as u64;
        let mut dot = 0.0f32;
        for s in (0..quads).unroll(MAX_QUADS) {
            let d = s * (LANES as u32 * WIDE) + l * WIDE;
            let b = (s * WIDE) as usize;
            if d < dim {
                let qv = q.load4(q_at + d as u64);
                dot += qv.x * kr[b] + qv.y * kr[b + 1] + qv.z * kr[b + 2] + qv.w * kr[b + 3];
            }
        }
        dot = sum32(dot);
        if l == 0 {
            score += max(dot, 0.0) * weights[w_at + h as u64];
        }
    }
    score
}

/// Scores one query row against one key row per thread.
#[kernel]
fn score_plain(
    #[form(q_form)] q: Buf<Dyn>,
    #[form(k_form)] k: Buf<Dyn>,
    #[form(w_form)] weights: Buf<Dyn>,
    row: u32,
    k_at: u64,
    heads: u32,
    dim: u32,
    #[build] q_form: &'static dyn Stored,
    #[build] k_form: &'static dyn Stored,
    #[build] w_form: &'static dyn Stored,
) -> f32 {
    let w_at = row as u64 * heads as u64;
    let mut score = 0.0f32;
    for h in 0..heads {
        let q_at = (row * heads + h) as u64 * dim as u64;
        let mut dot = 0.0f32;
        for d in 0..dim {
            dot += q[q_at + d as u64] * k[k_at + d as u64];
        }
        score += max(dot, 0.0) * weights[w_at + h as u64];
    }
    score
}

/// Generates matching warp and plain indexer kernels for one layout.
macro_rules! indexer_pair {
    (
        warp = $warp:ident,
        plain = $plain:ident,
        bufs: [$($buf:ident : Buf<$bty:ty>),*],
        scalars: [$($sc:ident : u32),*],
        [$row:ident, $j:ident, $from:ident, $to:ident, $out_keys:ident, $masked:ident],
        bounds: { $($bounds:tt)* },
        key_row: { $($krow:tt)* } $(,)?
    ) => {
        #[kernel]
        fn $warp(
            #[form(q_form)] q: Buf<Dyn>,
            #[form(k_form)] k: Buf<Dyn>,
            #[form(w_form)] weights: Buf<Dyn>,
            $($buf: Buf<$bty>,)*
            #[form(out_form)] mut out: BufMut<Dyn>,
            rows: u32,
            $out_keys: u32,
            heads: u32,
            per_row: u32,
            $masked: u32,
            $($sc: u32,)*
            #[build] dim: u32,
            #[build] q_form: &'static dyn Stored,
            #[build] k_form: &'static dyn Stored,
            #[build] w_form: &'static dyn Stored,
            #[build] out_form: &'static dyn Stored,
        ) {
            let wid = gid() / LANES as u32;
            if wid >= rows * per_row {
                return;
            }
            let $row = wid / per_row;
            let start = wid % per_row;
            let l = lane();
            $($bounds)*
            for $j in (start..$out_keys).by(per_row) {
                let out_at = $row as u64 * $out_keys as u64 + $j as u64;
                if $j >= $from && $j < $to {
                    let k_at = ($($krow)*) as u64 * dim as u64;
                    let score =
                        score_warp(q, k, weights, $row, k_at, heads, dim, q_form, k_form, w_form);
                    if l == 0 {
                        out[out_at] = score;
                    }
                } else if l == 0 {
                    out[out_at] = f32::NEG_INFINITY;
                }
            }
        }

        #[kernel]
        fn $plain(
            #[form(q_form)] q: Buf<Dyn>,
            #[form(k_form)] k: Buf<Dyn>,
            #[form(w_form)] weights: Buf<Dyn>,
            $($buf: Buf<$bty>,)*
            #[form(out_form)] mut out: BufMut<Dyn>,
            rows: u32,
            $out_keys: u32,
            heads: u32,
            dim: u32,
            $masked: u32,
            $($sc: u32,)*
            #[build] q_form: &'static dyn Stored,
            #[build] k_form: &'static dyn Stored,
            #[build] w_form: &'static dyn Stored,
            #[build] out_form: &'static dyn Stored,
        ) {
            if gid() >= rows * $out_keys {
                return;
            }
            let $row = gid() / $out_keys;
            let $j = gid() % $out_keys;
            $($bounds)*
            let mut score = f32::NEG_INFINITY;
            if $j >= $from && $j < $to {
                let k_at = ($($krow)*) as u64 * dim as u64;
                score = score_plain(q, k, weights, $row, k_at, heads, dim, q_form, k_form, w_form);
            }
            out[gid() as u64] = score;
        }
    };
}

indexer_pair!(
    warp = rows_warp,
    plain = rows_plain,
    bufs: [slots: Buf<U32>, positions: Buf<U32>],
    scalars: [live_keys: u32],
    [row, j, from, to, out_keys, masked],
    bounds: {
        let slot = slots[row as u64];
        let from = 0u32;
        let mut to = live_keys;
        if masked == 1 {
            let pos = positions[row as u64];
            if pos < live_keys {
                to = pos + 1;
            }
        }
    },
    key_row: { kept_row(slot, j) },
);

indexer_pair!(
    warp = whole_warp,
    plain = whole_plain,
    bufs: [],
    scalars: [queries: u32, past: u32, back: u32, ahead: u32],
    [row, j, from, to, out_keys, masked],
    bounds: {
        let slot = row / queries;
        let mut from = 0u32;
        let mut to = out_keys;
        if masked == 1 {
            let pos = past + row % queries;
            if pos > back {
                from = pos - back;
            }
            if out_keys > pos && out_keys - pos - 1 > ahead {
                to = pos + ahead + 1;
            }
        }
    },
    key_row: { slot * out_keys + j },
);

/// Sizes and storage forms for one Lightning Indexer launch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sizes {
    pub rows: usize,
    pub seqs: usize,
    pub queries: usize,
    pub keys: Extent,
    pub context: usize,
    pub kept: &'static dyn Layout,
    pub heads: usize,
    pub dim: usize,
    pub ragged: bool,
    pub masked: bool,
    pub window: (usize, usize),
    pub q_form: &'static dyn Stored,
    pub k_form: &'static dyn Stored,
    pub w_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

impl Sizes {
    fn out_keys(&self) -> usize {
        self.keys.most()
    }

    fn out_len(&self) -> usize {
        self.rows * self.out_keys()
    }

    fn per_row(&self) -> usize {
        (MAX_WARPS / self.rows.max(1)).clamp(1, self.out_keys().max(1))
    }

    fn form_tag(&self) -> String {
        format!(
            "{}_{}_{}_{}",
            self.q_form.name(),
            self.k_form.name(),
            self.w_form.name(),
            self.out_form.name(),
        )
    }

    fn accepts(&self, input: usize, form: &dyn Stored) -> bool {
        match input {
            0 => form.name() == self.q_form.name(),
            1 => form.name() == self.k_form.name(),
            2 => form.name() == self.w_form.name(),
            _ => !form.narrow(),
        }
    }

    fn emits(&self, form: &dyn Stored) -> bool {
        form.name() == self.out_form.name()
    }

    fn buffers(&self, step: &Step) -> Vec<Arg> {
        let mut args: Vec<Arg> = (0..step.reads.len())
            .map(|i| step.read(i))
            .chain([step.out()])
            .collect();
        if self.ragged {
            args.extend(step.kept_for(self.kept, true));
        }
        args
    }

    fn kept_scalars(&self) -> impl Iterator<Item = Arg> {
        self.kept
            .values(self.seqs, self.context)
            .into_iter()
            .map(|v| Arg::count(v as usize))
    }

    fn whole_tail(&self) -> [Arg; 4] {
        let past = self.out_keys().saturating_sub(self.queries);
        [self.queries, past, self.window.0, self.window.1].map(Arg::count)
    }

    fn push_tail(&self, args: &mut Vec<Arg>) {
        if self.ragged {
            args.push(Arg::size(self.keys));
            args.extend(self.kept_scalars());
        } else {
            args.extend(self.whole_tail());
        }
    }
}

/// Warp-parallel fused Lightning Indexer kernel.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "indexer_warp",
    order = Order::PerConfig,
    sizes = [self.0.rows, self.0.out_keys(), self.0.heads, self.0.dim],
    work = self.0.out_len() * self.0.heads * self.0.dim,
    needs = warp_needs(),
    accepts = warp_accepts,
    emits = warp_emits,
    source = warp_source,
    launches = warp_launches,
    shape = Shape::grid(
        ((self.0.rows * self.0.per_row()).div_ceil(WARPS as usize), 1),
        THREADS as usize,
    ),
)]
pub(crate) struct IndexerWarp(pub Sizes);

fn warp_needs() -> Needs {
    Needs {
        lanes: LANES,
        group_threads: THREADS as usize,
        ..Needs::default()
    }
}

fn warp_accepts(k: &IndexerWarp, input: usize, form: &dyn Stored) -> bool {
    k.0.accepts(input, form)
}

fn warp_emits(k: &IndexerWarp, form: &dyn Stored) -> bool {
    k.0.emits(form)
}

fn warp_source(k: &IndexerWarp) -> Source {
    let s = &k.0;
    let tag = s.form_tag();
    if s.ragged {
        let ir = rows_warp_ir(s.dim as u32, s.q_form, s.k_form, s.w_form, s.out_form);
        let body = kept::attach(ir, s.kept);
        let name = format!("plan_idx_rows_w_{}_{}_{tag}", s.kept.name(), s.dim);
        return Source::named(body, name);
    }
    let ir = whole_warp_ir(s.dim as u32, s.q_form, s.k_form, s.w_form, s.out_form);
    Source::named(ir, format!("plan_idx_whole_w_{}_{tag}", s.dim))
}

fn warp_launches(k: &IndexerWarp, step: &Step) -> Vec<Launch> {
    let s = &k.0;
    let (per_row, shape) = (s.per_row(), k.shape().expect("a grid"));
    let mut args = s.buffers(step);
    let head = [
        s.rows,
        s.out_keys(),
        s.heads,
        per_row,
        usize::from(s.masked),
    ];
    args.extend(head.map(Arg::count));
    s.push_tail(&mut args);
    vec![Launch { args, shape }]
}

/// Lane-agnostic fused Lightning Indexer kernel.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "indexer",
    sizes = [self.0.rows, self.0.out_keys(), self.0.heads, self.0.dim],
    work = self.0.out_len() * self.0.heads * (self.0.dim + 1),
    accepts = plain_accepts,
    emits = plain_emits,
    source = plain_source,
    launches = plain_launches,
)]
pub(crate) struct IndexerPlain(pub Sizes);

fn plain_accepts(k: &IndexerPlain, input: usize, form: &dyn Stored) -> bool {
    k.0.accepts(input, form)
}

fn plain_emits(k: &IndexerPlain, form: &dyn Stored) -> bool {
    k.0.emits(form)
}

fn plain_source(k: &IndexerPlain) -> Source {
    let s = &k.0;
    let tag = s.form_tag();
    if s.ragged {
        let ir = rows_plain_ir(s.q_form, s.k_form, s.w_form, s.out_form);
        let body = kept::attach(ir, s.kept);
        let name = format!("plan_idx_rows_{}_{tag}", s.kept.name());
        return Source::named(body, name);
    }
    let ir = whole_plain_ir(s.q_form, s.k_form, s.w_form, s.out_form);
    Source::named(ir, format!("plan_idx_whole_{tag}"))
}

fn plain_launches(k: &IndexerPlain, step: &Step) -> Vec<Launch> {
    let s = &k.0;
    let mut args = s.buffers(step);
    let head = [s.rows, s.out_keys(), s.heads, s.dim, usize::from(s.masked)];
    args.extend(head.map(Arg::count));
    s.push_tail(&mut args);
    vec![Launch {
        args,
        shape: Shape::Line(s.out_len()),
    }]
}

/// Optional masking node above the score projection.
enum FoundMask {
    None,
    Causal(T),
    Fixed(Mask),
}

/// Strips an optional mask and trailing reshape above `matmul_rows`.
fn strip_top(tree: &Tree, at: T) -> Option<(FoundMask, T)> {
    let node = tree.node(at);
    let (mask, mut cur) = match &node.term {
        Term::Stage(Stage::Causal) => (FoundMask::Causal(node.args[1]), node.args[0]),
        Term::Op(Op::Mask(kind)) => (FoundMask::Fixed(*kind), node.args[0]),
        Term::Op(Op::Matmul) => return Some((FoundMask::None, at)),
        _ => return None,
    };
    if matches!(tree.node(cur).term, Term::Op(Op::View(View::Reshape(_)))) {
        cur = tree.node(cur).args[0];
    }
    Some((mask, cur))
}

/// Checks a permutation swapping only the last two axes.
fn swaps_last(order: &[u32], rank: usize) -> bool {
    if order.len() != rank || rank < 3 {
        return false;
    }
    let prefix = (0..rank - 2).all(|i| order[i] == i as u32);
    prefix && order[rank - 2] == (rank - 1) as u32 && order[rank - 1] == (rank - 2) as u32
}

/// Matches the Lightning Indexer scoring chain at `at`.
fn find_site(tree: &Tree, at: T) -> Option<(Sizes, Vec<T>)> {
    let (mask, scores_mm) = strip_top(tree, at)?;
    let mm_node = tree.node(scores_mm);
    if mm_node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (hits_perm, by_head) = (mm_node.args[0], mm_node.args[1]);
    let perm_node = tree.node(hits_perm);
    let Term::Op(Op::View(View::Permute(order))) = &perm_node.term else {
        return None;
    };
    let rank = perm_node.shape.len();
    if !matches!(rank, 3 | 4) || !swaps_last(order, rank) {
        return None;
    }
    let lead = &perm_node.shape[..rank - 2];
    let (out_keys, heads) = (perm_node.shape[rank - 2], perm_node.shape[rank - 1]);
    let w_shape = &tree.node(by_head).shape;
    if w_shape.len() != rank
        || &w_shape[..rank - 2] != lead
        || w_shape[rank - 2] != 1
        || w_shape[rank - 1] != heads
    {
        return None;
    }
    let hits_relu = perm_node.args[0];
    if tree.node(hits_relu).term != Term::Stage(Stage::Act(Act::Relu)) {
        return None;
    }
    let hits_mm = tree.node(hits_relu).args[0];
    let hits_node = tree.node(hits_mm);
    if hits_node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (q, k_src) = (hits_node.args[0], hits_node.args[1]);
    let q_node = tree.node(q);
    if q_node.shape.len() != rank
        || &q_node.shape[..rank - 2] != lead
        || q_node.shape[rank - 2] != heads
    {
        return None;
    }
    let dim = q_node.shape[rank - 1];
    let rows: usize = lead.iter().product();
    if tree.elements(at) != rows * out_keys {
        return None;
    }
    let out_node = tree.node(at);
    if !out_node.form.dtype().is_float()
        || !q_node.form.dtype().is_float()
        || !tree.node(by_head).form.dtype().is_float()
    {
        return None;
    }
    if rank == 3 {
        return find_ragged(
            tree,
            mask,
            [q, k_src, by_head, at],
            [rows, out_keys, heads, dim],
        );
    }
    find_whole(
        tree,
        mask,
        [q, k_src, by_head, at],
        [lead[0], lead[1], out_keys, heads, dim],
    )
}

/// Finishes a stepped ragged match over kept keys.
fn find_ragged(
    tree: &Tree,
    mask: FoundMask,
    [q, k_src, by_head, at]: [T; 4],
    [rows, out_keys, heads, dim]: [usize; 4],
) -> Option<(Sizes, Vec<T>)> {
    let gather = tree.node(k_src);
    if gather.term != Term::Op(Op::Select(Select::Gather)) {
        return None;
    }
    let (k_append, slots) = (gather.args[0], gather.args[1]);
    let k_node = tree.node(k_append);
    let Term::Op(Op::Append(ki)) = k_node.term else {
        return None;
    };
    let &[seqs, k_keys, k_dim] = &k_node.shape[..] else {
        return None;
    };
    if k_keys != out_keys || k_dim != dim || !k_node.form.dtype().is_float() {
        return None;
    }
    if tree.node(q).extents[0].runtime() || tree.elements(slots) != rows {
        return None;
    }
    if tree.node(slots).form.dtype() != DType::U32 {
        return None;
    }
    let (_, context) = tree.kept_rows(ki)?;
    let keys = k_node.extents[1];
    let (masked, positions) = match mask {
        FoundMask::Causal(pos) => {
            if tree.elements(pos) != rows || tree.node(pos).form.dtype() != DType::U32 {
                return None;
            }
            (true, pos)
        }
        FoundMask::None if !keys.runtime() => (false, slots),
        _ => return None,
    };
    let sizes = Sizes {
        rows,
        seqs,
        queries: 1,
        keys,
        context,
        kept: tree.kept,
        heads,
        dim,
        ragged: true,
        masked,
        window: (0, 0),
        q_form: plain(tree.node(q).form),
        k_form: plain(k_node.form),
        w_form: plain(tree.node(by_head).form),
        out_form: writable(tree.node(at).form),
    };
    Some((sizes, vec![q, k_append, by_head, slots, positions]))
}

/// Finishes a whole-sequence match over `[seqs, keys, dim]` keys.
fn find_whole(
    tree: &Tree,
    mask: FoundMask,
    [q, k_src, by_head, at]: [T; 4],
    [seqs, queries, out_keys, heads, dim]: [usize; 5],
) -> Option<(Sizes, Vec<T>)> {
    let k_node = tree.node(k_src);
    if matches!(k_node.term, Term::Op(Op::Append(_))) || !k_node.form.dtype().is_float() {
        return None;
    }
    let &[k_seqs, k_keys, k_dim] = &k_node.shape[..] else {
        return None;
    };
    if k_seqs != seqs || k_keys != out_keys || k_dim != dim {
        return None;
    }
    let q_ext = &tree.node(q).extents;
    if q_ext[0].runtime() || q_ext[1].runtime() || k_node.extents[1].runtime() {
        return None;
    }
    let (masked, window) = match mask {
        FoundMask::Fixed(kind) => (true, kind.window(u32::MAX as usize)),
        FoundMask::None => (false, (0, 0)),
        FoundMask::Causal(_) => return None,
    };
    let sizes = Sizes {
        rows: seqs * queries,
        seqs,
        queries,
        keys: Extent::Fixed(out_keys),
        context: out_keys,
        kept: &Dense,
        heads,
        dim,
        ragged: false,
        masked,
        window,
        q_form: plain(tree.node(q).form),
        k_form: plain(k_node.form),
        w_form: plain(tree.node(by_head).form),
        out_form: writable(tree.node(at).form),
    };
    Some((sizes, vec![q, k_src, by_head]))
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (sizes, reads) = find_site(tree, at)?;
    let out = sizes.out_len();
    let mut drafts = Vec::new();
    let fits_warp =
        sizes.dim.is_multiple_of(WIDE as usize) && sizes.dim <= MAX_DIM && sizes.rows <= MAX_WARPS;
    if fits_warp {
        drafts.push(Draft::new(IndexerWarp(sizes), reads.iter().copied(), out).choice(0));
    }
    if out <= u32::MAX as usize {
        drafts.push(Draft::new(IndexerPlain(sizes), reads, out).choice(1));
    }
    (!drafts.is_empty()).then_some(drafts)
}

tessera_lower::rule!("indexer", propose);
