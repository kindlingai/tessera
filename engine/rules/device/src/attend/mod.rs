//! Tensor-Core MLA, GQA, and Lightning Indexer rules for CUDA.

use crate::linear::{CUDA_LACKS, f32_only};
use crate::{Backend, Options, flatten};
use std::sync::OnceLock;
use tessera_graph::{Act, DType, Op, Order, Select, Stage, View};
use tessera_lower::form::Load as Stored;
use tessera_lower::kept::{self, Layout};
use tessera_lower::{
    Arg, ByBackend, Draft, Extent, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Term,
    Tree,
};

const KERNEL: &str = include_str!("mla.cu");
const MLA_WIDTH: usize = 576;
const GQA_WIDTH: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttendKind {
    Mla,
    Gqa,
}

impl AttendKind {
    fn width(self) -> usize {
        match self {
            Self::Mla => MLA_WIDTH,
            Self::Gqa => GQA_WIDTH,
        }
    }

    fn stride(self) -> usize {
        self.width() + 4
    }

    fn smem(self) -> usize {
        if self == Self::Mla { 47_104 } else { 18_432 }
    }

    fn threads(self) -> usize {
        if self == Self::Mla { 256 } else { 128 }
    }

    fn tile_keys(self) -> usize {
        if self == Self::Mla { 32 } else { 16 }
    }

    fn names(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::Mla => ("fast_mla_attend", "fast_mla_reduce", "tessera_mla_reduce"),
            Self::Gqa => ("fast_gqa_attend", "fast_gqa_reduce", "tessera_gqa_reduce"),
        }
    }

    fn kv_entry(self, kv: &str) -> Option<&'static str> {
        match (self, kv) {
            (Self::Mla, "fp8") => Some("tessera_mla_attend_fp8"),
            (Self::Mla, "bf16") => Some("tessera_mla_attend_bf16"),
            (Self::Mla, "f32") => Some("tessera_mla_attend_f32"),
            (Self::Gqa, "fp8") => Some("tessera_gqa_attend_fp8"),
            (Self::Gqa, "bf16") => Some("tessera_gqa_attend_bf16"),
            (Self::Gqa, "f32") => Some("tessera_gqa_attend_f32"),
            _ => None,
        }
    }
}

fn attend_entry(entry: &'static str) -> Source {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    let text = *CUDA.get_or_init(|| {
        let options = Options {
            backend: Backend::Cuda,
            remote_kv: false,
            marks: false,
        };
        let flat = flatten("mla.kernel", KERNEL, &|_| None, &options)
            .expect("attention kernel flattens")
            .text;
        Box::leak(flat.into_boxed_str())
    });
    Source::Fixed {
        text: ByBackend::from_fn(|_| text),
        entry,
    }
}

fn split_count(kind: AttendKind, rows: usize, hgroups: usize) -> usize {
    let groups = rows * hgroups;
    match kind {
        AttendKind::Mla if groups <= 8 => 16,
        AttendKind::Mla if groups <= 16 => 8,
        AttendKind::Mla => 4,
        AttendKind::Gqa if groups <= 16 => 16,
        AttendKind::Gqa if groups <= 32 => 8,
        AttendKind::Gqa => 4,
    }
}

/// Grouped-Q split-K MLA or GQA over dense or paged KV.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.kind.names().0,
    sizes = [self.rows, self.heads, self.kv_heads, self.keys.most(), self.picks, self.splits],
    work = self.rows * self.heads * self.picks * self.kind.width(),
    accepts = attend_accepts,
    emits = f32_only,
    needs = attend_needs(self),
    order = Order::PerConfig,
    source = attend_entry(self.entry),
    launches = attend_launches,
    load = attend_load,
    unproven,
)]
struct SplitAttend {
    kind: AttendKind,
    rows: usize,
    heads: usize,
    kv_heads: usize,
    keys: Extent,
    context: usize,
    kept: &'static dyn Layout,
    picks: usize,
    from_list: bool,
    scale: f32,
    window: (usize, usize),
    splits: usize,
    kv_form: &'static dyn Stored,
    entry: &'static str,
}

impl SplitAttend {
    fn hgroups(&self) -> usize {
        match self.kind {
            AttendKind::Mla => self.heads / 16,
            AttendKind::Gqa => self.kv_heads * (self.heads / self.kv_heads).div_ceil(16),
        }
    }
}

fn attend_accepts(k: &SplitAttend, input: usize, form: &dyn Stored) -> bool {
    match (k.kind, input) {
        (_, 0) => form.name() == "f32",
        (AttendKind::Mla, 1) | (AttendKind::Gqa, 1 | 2) => form.name() == k.kv_form.name(),
        _ => form.name() == "u32",
    }
}

fn attend_needs(k: &SplitAttend) -> Needs {
    Needs {
        lanes: 32,
        group_bytes: k.kind.smem(),
        group_threads: k.kind.threads(),
        compiler: true,
        lacks: CUDA_LACKS,
        bf16_tile: true,
        ..Needs::default()
    }
}

fn attend_load(k: &SplitAttend, _: usize) -> Option<Load> {
    let w = k.kind.width();
    let threads = k.kind.threads();
    let groups = k.rows * k.hgroups() * k.splits;
    let seen = k.picks.min(k.keys.most()).max(1);
    let q_bytes = (4 * k.rows * k.heads * w) as f64;
    let kv_mul = match k.kind {
        AttendKind::Mla => 1,
        AttendKind::Gqa => 2,
    };
    let kv_bytes = (kv_mul * k.kv_form.bytes(k.rows * seen * k.kv_heads * w)) as f64;
    let out_bytes = (4 * k.rows * k.heads * k.splits * k.kind.stride()) as f64;
    Some(Load {
        threads: groups * threads,
        group: threads,
        trips: seen.div_ceil(k.kind.tile_keys() * k.splits) as f64,
        reads: vec![kv_bytes, q_bytes],
        written: out_bytes,
        flight: 1024.0,
        ..Load::default()
    })
}

fn attend_launches(k: &SplitAttend, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: k.splits,
        down: k.rows * k.hgroups(),
        threads: k.kind.threads(),
    };
    let mut args = vec![
        step.read(0),
        step.read(1),
        step.read(2),
        step.read(3),
        step.read(4),
        step.out(),
        Arg::value(&k.scale),
        Arg::count(k.rows),
        Arg::size(k.keys),
        Arg::count(k.context),
        Arg::count(k.heads),
    ];
    match k.kind {
        AttendKind::Mla => {
            args.push(Arg::count(k.picks));
            args.push(Arg::count(usize::from(k.from_list)));
        }
        AttendKind::Gqa => args.push(Arg::count(k.kv_heads)),
    }
    let page_shift = k.kept.page().map_or(0, |p| p.trailing_zeros() as usize);
    let table = step.kept(k.kept).unwrap_or_else(|| step.read(3));
    args.extend([
        Arg::count(k.window.0),
        Arg::count(k.window.1),
        Arg::count(k.splits),
        table,
        Arg::count(page_shift),
    ]);
    vec![Launch { args, shape }]
}

/// Combines split-K MLA or GQA partials into final outputs.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.kind.names().1,
    sizes = [self.lists, self.splits],
    work = self.lists * self.splits * self.kind.width(),
    accepts = red_accepts,
    emits = f32_only,
    needs = red_needs(self),
    order = Order::PerConfig,
    source = attend_entry(self.kind.names().2),
    launches = red_launches,
    load = red_load,
    unproven,
)]
struct SplitReduce {
    kind: AttendKind,
    lists: usize,
    splits: usize,
}

fn red_accepts(_: &SplitReduce, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn red_needs(r: &SplitReduce) -> Needs {
    let (group_bytes, group_threads) = match r.kind {
        AttendKind::Mla => (4 * 32 * 3 + 4, 160),
        AttendKind::Gqa => (0, 128),
    };
    Needs {
        lanes: 32,
        group_bytes,
        group_threads,
        compiler: true,
        lacks: CUDA_LACKS,
        ..Needs::default()
    }
}

fn red_load(r: &SplitReduce, _: usize) -> Option<Load> {
    let (blocks, group) = match r.kind {
        AttendKind::Mla => (r.lists, 160),
        AttendKind::Gqa => (r.lists.div_ceil(4), 128),
    };
    Some(Load {
        threads: blocks * group,
        group,
        trips: r.splits as f64,
        reads: vec![(4 * r.lists * r.splits * r.kind.stride()) as f64],
        written: (4 * r.lists * r.kind.width()) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn red_launches(r: &SplitReduce, step: &Step) -> Vec<Launch> {
    let (across, threads) = match r.kind {
        AttendKind::Mla => (r.lists, 160),
        AttendKind::Gqa => (r.lists.div_ceil(4), 128),
    };
    let shape = Shape::Grid {
        across,
        down: 1,
        threads,
    };
    let args = vec![
        step.read(0),
        step.out(),
        Arg::count(r.lists),
        Arg::count(r.splits),
    ];
    vec![Launch { args, shape }]
}

fn kept_stride(kept: &'static dyn Layout, context: usize) -> Option<(usize, &'static dyn Layout)> {
    if kept.name() == "dense" {
        return Some((context, kept));
    }
    let page = kept.page()?;
    Some((context.div_ceil(page), kept))
}

fn kept_context(tree: &Tree, k: T, v: T, most: usize) -> Option<(usize, &'static dyn Layout)> {
    match (&tree.node(k).term, &tree.node(v).term) {
        (Term::Op(Op::Append(ki)), Term::Op(Op::Append(vi))) => {
            let ctx = tree.kept_rows(*ki)?.1;
            if tree.kept_rows(*vi)?.1 != ctx {
                return None;
            }
            kept_stride(tree.kept, ctx)
        }
        (Term::Op(Op::Append(_)), _) | (_, Term::Op(Op::Append(_))) => None,
        _ => Some((most, kept::DENSE)),
    }
}

fn propose(tree: &Tree, at: T, kind: AttendKind) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Op(Op::Attend(a)) = &node.term else {
        return None;
    };
    if !a.ragged || node.form.name() != "f32" {
        return None;
    }
    let (q, k, v) = (node.args[0], node.args[1], node.args[2]);
    if tree.node(q).form.name() != "f32" {
        return None;
    }
    let kv_form = tree.node(k).form;
    if tree.node(v).form.name() != kv_form.name() {
        return None;
    }
    let entry = kind.kv_entry(kv_form.name())?;
    let (&[rows, heads, width], &[_, most, kv_heads, kw]) =
        (&tree.node(q).shape[..], &tree.node(k).shape[..])
    else {
        return None;
    };
    if width != kind.width() || kw != width || tree.node(v).shape.last() != Some(&width) {
        return None;
    }
    if rows == 0 || rows > crate::FAST_MAX_ROWS || kv_heads == 0 || !heads.is_multiple_of(kv_heads)
    {
        return None;
    }
    let [slots, positions] = [node.args[3], node.args[4]];
    let (reads, picks, hgroups) = match kind {
        AttendKind::Mla => {
            if k != v || kv_heads != 1 || heads < 16 || !heads.is_multiple_of(16) {
                return None;
            }
            let (ids, picks) = match node.args.get(5) {
                Some(&ids) => (ids, *tree.node(ids).shape.last()?),
                None => (positions, most),
            };
            ([q, k, ids, slots, positions], picks, heads / 16)
        }
        AttendKind::Gqa => {
            let gqa = heads / kv_heads;
            if a.picked || gqa < 4 || (gqa > 16 && !gqa.is_multiple_of(16)) {
                return None;
            }
            (
                [q, k, v, slots, positions],
                most,
                kv_heads * gqa.div_ceil(16),
            )
        }
    };
    let (context, kept) = kept_context(tree, k, v, most)?;
    let splits = split_count(kind, rows, hgroups);
    let attend = SplitAttend {
        kind,
        rows,
        heads,
        kv_heads,
        keys: tree.node(k).extents[1],
        context,
        kept,
        picks,
        from_list: a.picked,
        scale: a.scale,
        window: a.mask.window(most),
        splits,
        kv_form,
        entry,
    };
    let lists = rows * heads;
    let reduce = SplitReduce {
        kind,
        lists,
        splits,
    };
    Some(vec![
        Draft::new(attend, reads, lists * splits * kind.stride()),
        Draft::new(reduce, [Read::Draft(0)], lists * width),
    ])
}

fn propose_mla(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    propose(tree, at, AttendKind::Mla)
}

fn propose_gqa(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    propose(tree, at, AttendKind::Gqa)
}

tessera_lower::rule!("fast_mla_attend", propose_mla);
tessera_lower::rule!("fast_gqa_attend", propose_gqa);

const IDX_SM_WORDS: usize = 4608;
const IDX_SMEM: usize = IDX_SM_WORDS * 8 + 128 * 4;
const IDX_QT_SMEM: usize = 2 * 32 * 136 * 2 + 16;
const IDX_MAX_HEADS: usize = 128;

fn idx_kv_entry(kv: &str, qt: bool) -> Option<&'static str> {
    match (kv, qt) {
        ("fp8", false) => Some("tessera_indexer_fp8"),
        ("bf16", false) => Some("tessera_indexer_bf16"),
        ("f32", false) => Some("tessera_indexer_f32"),
        ("fp8", true) => Some("tessera_indexer_qt_fp8"),
        ("bf16", true) => Some("tessera_indexer_qt_bf16"),
        ("f32", true) => Some("tessera_indexer_qt_f32"),
        _ => None,
    }
}

fn idx_tile_keys(rows: usize, qt: bool) -> usize {
    if qt {
        1024
    } else if rows <= 4 {
        64
    } else {
        256
    }
}

/// Tensor-Core Lightning Indexer over dense or paged keys.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "fast_indexer",
    sizes = [self.rows, self.out_keys, self.heads, self.dim],
    work = self.rows * self.out_keys * self.heads * self.dim,
    accepts = idx_accepts,
    emits = f32_only,
    needs = idx_needs(self),
    order = Order::PerConfig,
    source = attend_entry(self.entry),
    launches = idx_launches,
    load = idx_load,
    unproven,
)]
struct FastIndexer {
    rows: usize,
    out_keys: usize,
    keys: Extent,
    context: usize,
    kept: &'static dyn Layout,
    heads: usize,
    dim: usize,
    masked: bool,
    qt: bool,
    k_form: &'static dyn Stored,
    entry: &'static str,
}

fn idx_accepts(k: &FastIndexer, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 | 2 => form.name() == "f32",
        1 => form.name() == k.k_form.name(),
        _ => form.name() == "u32",
    }
}

fn idx_needs(k: &FastIndexer) -> Needs {
    Needs {
        lanes: 32,
        group_bytes: if k.qt { IDX_QT_SMEM } else { IDX_SMEM },
        group_threads: if k.qt { 256 } else { 128 },
        compiler: true,
        lacks: CUDA_LACKS,
        bf16_tile: true,
        ..Needs::default()
    }
}

fn idx_load(k: &FastIndexer, _: usize) -> Option<Load> {
    let tk = idx_tile_keys(k.rows, k.qt);
    let blocks_x = k.out_keys.div_ceil(tk);
    let blocks_y = if k.qt { k.rows.div_ceil(8) } else { k.rows };
    let group = if k.qt { 256 } else { 128 };
    let q_bytes = (4 * k.rows * k.heads * k.dim) as f64;
    let k_bytes = k.k_form.bytes(blocks_y * k.out_keys * k.dim) as f64;
    let out_bytes = (4 * k.rows * k.out_keys) as f64;
    Some(Load {
        threads: blocks_x * blocks_y * group,
        group,
        trips: tk.div_ceil(if k.qt { 32 } else { 64 }) as f64 * (k.heads / 16) as f64,
        reads: vec![k_bytes, q_bytes],
        written: out_bytes,
        flight: 1024.0,
        ..Load::default()
    })
}

fn idx_launches(k: &FastIndexer, step: &Step) -> Vec<Launch> {
    let tk = idx_tile_keys(k.rows, k.qt);
    let shape = Shape::Grid {
        across: k.out_keys.div_ceil(tk),
        down: if k.qt { k.rows.div_ceil(8) } else { k.rows },
        threads: if k.qt { 256 } else { 128 },
    };
    let page_shift = k.kept.page().map_or(0, |p| p.trailing_zeros() as usize);
    let table = step.kept(k.kept).unwrap_or_else(|| step.read(3));
    let args = vec![
        step.read(0),
        step.read(1),
        step.read(2),
        step.read(3),
        step.read(4),
        step.out(),
        Arg::count(k.rows),
        Arg::count(k.out_keys),
        Arg::size(k.keys),
        Arg::count(k.context),
        Arg::count(k.heads),
        Arg::count(k.dim),
        Arg::count(usize::from(k.masked)),
        Arg::count(tk),
        table,
        Arg::count(page_shift),
    ];
    vec![Launch { args, shape }]
}

/// Matches stepped ragged Lightning Indexer over dense or paged keys.
fn propose_indexer(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    if tree.kept.name() != "dense" && tree.kept.page().is_none() {
        return None;
    }
    let node = tree.node(at);
    let (masked_pos, mut cur) = match &node.term {
        Term::Stage(Stage::Causal) => (Some(node.args[1]), node.args[0]),
        Term::Op(Op::Matmul) => (None, at),
        _ => return None,
    };
    if matches!(tree.node(cur).term, Term::Op(Op::View(View::Reshape(_)))) {
        cur = tree.node(cur).args[0];
    }
    let mm_node = tree.node(cur);
    if mm_node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (hits_perm, by_head) = (mm_node.args[0], mm_node.args[1]);
    let perm_node = tree.node(hits_perm);
    let Term::Op(Op::View(View::Permute(order))) = &perm_node.term else {
        return None;
    };
    if perm_node.shape.len() != 3 || order.as_slice() != [0, 2, 1] {
        return None;
    }
    let (rows, out_keys, heads) = (perm_node.shape[0], perm_node.shape[1], perm_node.shape[2]);
    let w_node = tree.node(by_head);
    if w_node.shape != [rows, 1, heads] || w_node.form.name() != "f32" {
        return None;
    }
    let hits_relu = perm_node.args[0];
    if tree.node(hits_relu).term != Term::Stage(Stage::Act(Act::Relu)) {
        return None;
    }
    let hits_node = tree.node(tree.node(hits_relu).args[0]);
    if hits_node.term != Term::Op(Op::Matmul) {
        return None;
    }
    let (q, k_src) = (hits_node.args[0], hits_node.args[1]);
    let q_node = tree.node(q);
    if q_node.shape.len() != 3
        || q_node.shape[0] != rows
        || q_node.shape[1] != heads
        || q_node.form.name() != "f32"
        || q_node.extents[0].runtime()
    {
        return None;
    }
    let dim = q_node.shape[2];
    if node.form.name() != "f32" || tree.elements(at) != rows * out_keys {
        return None;
    }
    let stride = dim / 2 + 2;
    if rows == 0
        || !(16..=IDX_MAX_HEADS).contains(&heads)
        || !heads.is_multiple_of(16)
        || dim < 16
        || !dim.is_multiple_of(16)
        || heads * stride > IDX_SM_WORDS
        || 64 * stride > IDX_SM_WORDS
    {
        return None;
    }
    let gather = tree.node(k_src);
    if gather.term != Term::Op(Op::Select(Select::Gather)) {
        return None;
    }
    let (k_append, slots) = (gather.args[0], gather.args[1]);
    let k_node = tree.node(k_append);
    let Term::Op(Op::Append(ki)) = k_node.term else {
        return None;
    };
    let &[_, k_keys, k_dim] = &k_node.shape[..] else {
        return None;
    };
    if k_keys != out_keys || k_dim != dim {
        return None;
    }
    let qt = rows >= 8 && heads == 32 && dim == 128;
    let entry = idx_kv_entry(k_node.form.name(), qt)?;
    if tree.elements(slots) != rows || tree.node(slots).form.dtype() != DType::U32 {
        return None;
    }
    let (_, raw_ctx) = tree.kept_rows(ki)?;
    let (context, kept) = kept_stride(tree.kept, raw_ctx)?;
    let keys = k_node.extents[1];
    let (masked, positions) = match masked_pos {
        Some(pos) => {
            if tree.elements(pos) != rows || tree.node(pos).form.dtype() != DType::U32 {
                return None;
            }
            (true, pos)
        }
        None if !keys.runtime() => (false, slots),
        None => return None,
    };
    let kernel = FastIndexer {
        rows,
        out_keys,
        keys,
        context,
        kept,
        heads,
        dim,
        masked,
        qt,
        k_form: k_node.form,
        entry,
    };
    let reads = [q, k_append, by_head, slots, positions];
    Some(vec![Draft::new(kernel, reads, rows * out_keys)])
}

tessera_lower::rule!("fast_indexer", propose_indexer);
