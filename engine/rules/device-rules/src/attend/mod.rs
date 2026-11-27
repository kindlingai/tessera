//! Tensor-Core MLA, GQA, and Lightning Indexer rules for CUDA.

#[cfg(test)]
mod attend_test;

use crate::flatten_cuda;
use std::sync::OnceLock;
use tessera_device::{CUDA_LACKS, cuda_rule, f32_only};
use tessera_graph::{Act, DType, Op, Order, Select, Stage, View};
use tessera_lower::form::Load as Stored;
use tessera_lower::kept::{self, Layout};
use tessera_lower::{
    Arg, ByBackend, Draft, Extent, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Term,
    Tree,
};

const KERNEL: &str = include_str!("mla.cu");
const MLA_WIDTH: usize = 576;
const MLA_LATENT: usize = 512;
const GQA_WIDTH: usize = 128;
const RING_SHIFT: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

    fn kv_entry(self, kv: &str, out_width: usize) -> Option<&'static str> {
        match (self, kv, out_width) {
            (Self::Mla, "fp8", MLA_LATENT) => Some("tessera_mla_attend_512_fp8"),
            (Self::Mla, "bf16", MLA_LATENT) => Some("tessera_mla_attend_512_bf16"),
            (Self::Mla, "f32", MLA_LATENT) => Some("tessera_mla_attend_512_f32"),
            (Self::Mla, "fp8", _) => Some("tessera_mla_attend_fp8"),
            (Self::Mla, "bf16", _) => Some("tessera_mla_attend_bf16"),
            (Self::Mla, "f32", _) => Some("tessera_mla_attend_f32"),
            (Self::Gqa, "fp8", _) => Some("tessera_gqa_attend_fp8"),
            (Self::Gqa, "bf16", _) => Some("tessera_gqa_attend_bf16"),
            (Self::Gqa, "f32", _) => Some("tessera_gqa_attend_f32"),
            _ => None,
        }
    }
}

fn attend_entry(
    entry: &'static str,
    defs: impl IntoIterator<Item = (&'static str, usize)>,
) -> Source {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    let base = *CUDA.get_or_init(|| flatten_cuda("mla.kernel", KERNEL));
    let text = tessera_device::cuda_prefixed(base, defs);
    Source::Fixed {
        text: ByBackend::from_fn(|_| text),
        entry,
    }
}

tessera_lower::knob!("fast_mla_attend", "splits", 1, 32);
tessera_lower::knob!("fast_mla_attend", "min_blocks", 1, 16);
tessera_lower::knob!("fast_mla_attend", "reduce_min_blocks", 1, 16);
tessera_lower::knob!("fast_mla_attend", "reduce_threads", 32, 1024);
tessera_lower::knob!("fast_gqa_attend", "splits", 1, 32);
tessera_lower::knob!("fast_gqa_attend", "min_blocks", 1, 16);
tessera_lower::knob!("fast_gqa_attend", "reduce_min_blocks", 1, 16);
tessera_lower::knob!("fast_indexer", "min_blocks", 1, 16);
tessera_lower::knob!("fast_indexer", "qt_min_rows", 1, 4096);
tessera_lower::knob!("fast_indexer", "qt_min_blocks", 1, 16);
tessera_lower::knob!("fast_indexer", "tile_keys", 16, 4096);
tessera_lower::knob!("fast_indexer", "small_rows", 1, 4096);
tessera_lower::knob!("fast_indexer", "small_tile_keys", 16, 4096);
tessera_lower::knob!("fast_indexer", "large_tile_keys", 16, 4096);
tessera_lower::knob!("fast_indexer", "qt_tile_keys", 32, 4096);

fn split_count(kind: AttendKind, rows: usize, hgroups: usize) -> usize {
    let groups = rows * hgroups;
    match kind {
        AttendKind::Mla if groups <= 8 => 16,
        AttendKind::Mla if groups <= 16 => 8,
        AttendKind::Mla if groups <= 64 => 4,
        AttendKind::Mla if groups <= 128 => 2,
        AttendKind::Mla => 1,
        AttendKind::Gqa if groups <= 16 => 16,
        AttendKind::Gqa if groups <= 32 => 8,
        AttendKind::Gqa if groups <= 64 => 4,
        AttendKind::Gqa if groups <= 128 => 2,
        AttendKind::Gqa => 1,
    }
}

/// Grouped-Q split-K MLA or GQA over KV.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.kind.names().0,
    sizes = [self.rows, self.heads, self.kv_heads, self.keys.most(), self.picks, self.splits],
    work = self.rows * self.heads * self.picks * self.kind.width(),
    accepts = attend_accepts,
    emits = f32_only,
    needs = attend_needs(self),
    order = Order::PerConfig,
    source = attend_entry(
        self.entry,
        self.min_blocks.map(|v| {
            let def = match self.kind {
                AttendKind::Mla => "MLA_MIN_BLOCKS",
                AttendKind::Gqa => "GQA_MIN_BLOCKS",
            };
            (def, v as usize)
        }),
    ),
    launches = attend_launches,
    load = attend_load, load_key = tessera_lower::load_key(((self.kind, self.rows, self.heads, self.kv_heads, self.out_width, self.keys), (self.context, self.kept.name(), self.picks, self.from_list, self.has_sinks), (self.scale.to_bits(), self.window, self.splits, self.min_blocks, self.kv_form.name(), self.entry))),
    unproven,
)]
struct SplitAttend {
    kind: AttendKind,
    rows: usize,
    heads: usize,
    kv_heads: usize,
    out_width: usize,
    keys: Extent,
    context: usize,
    kept: &'static dyn Layout,
    picks: usize,
    from_list: bool,
    has_sinks: bool,
    scale: f32,
    window: (usize, usize),
    splits: usize,
    min_blocks: Option<u32>,
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

    fn stride(&self) -> usize {
        self.out_width + 4
    }
}

fn attend_accepts(k: &SplitAttend, input: usize, form: &dyn Stored) -> bool {
    match (k.kind, input) {
        (_, 0) | (_, 5) => form.name() == "f32",
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
    let out_stride = match k.splits {
        1 => k.out_width,
        s => s * k.stride(),
    };
    let out_bytes = (4 * k.rows * k.heads * out_stride) as f64;
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
        step.read(5),
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
    let page_shift = page_shift(k.kept);
    let table = step
        .kept_for(k.kept, k.kind == AttendKind::Gqa)
        .unwrap_or_else(|| step.read(3));
    args.extend([
        Arg::count(k.window.0),
        Arg::count(k.window.1),
        Arg::count(k.splits),
        table,
        Arg::count(page_shift),
        Arg::count(usize::from(k.has_sinks)),
    ]);
    if k.kind == AttendKind::Mla {
        args.push(Arg::KeptHost(step.reads[1]));
    }
    vec![Launch { args, shape }]
}

/// Combines split-K MLA or GQA partials into final outputs.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.kind.names().1,
    sizes = [self.lists, self.splits, self.out_width],
    work = self.lists * self.splits * self.out_width,
    accepts = red_accepts,
    emits = f32_only,
    needs = red_needs(self),
    order = Order::PerConfig,
    source = reduce_source(self),
    launches = red_launches,
    load = red_load, load_key = tessera_lower::load_key((self.kind, self.lists, self.splits, self.out_width, self.threads, self.custom_threads, self.min_blocks)),
    unproven,
)]
struct SplitReduce {
    kind: AttendKind,
    lists: usize,
    splits: usize,
    out_width: usize,
    threads: usize,
    custom_threads: bool,
    min_blocks: Option<u32>,
}

fn reduce_source(r: &SplitReduce) -> Source {
    let mut defs = Vec::new();
    if let Some(mb) = r.min_blocks {
        let def = match r.kind {
            AttendKind::Mla => "MLA_REDUCE_MIN_BLOCKS",
            AttendKind::Gqa => "GQA_REDUCE_MIN_BLOCKS",
        };
        defs.push((def, mb as usize));
    }
    if r.kind == AttendKind::Mla && r.custom_threads {
        defs.push(("MLA_REDUCE_THREADS", r.threads));
    }
    attend_entry(r.kind.names().2, defs)
}

fn red_accepts(_: &SplitReduce, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn red_needs(r: &SplitReduce) -> Needs {
    let group_bytes = match r.kind {
        AttendKind::Mla => 4 * 32 * 3 + 4,
        AttendKind::Gqa => 0,
    };
    Needs {
        lanes: 32,
        group_bytes,
        group_threads: r.threads,
        compiler: true,
        lacks: CUDA_LACKS,
        ..Needs::default()
    }
}

fn red_load(r: &SplitReduce, _: usize) -> Option<Load> {
    let blocks = match r.kind {
        AttendKind::Mla => r.lists,
        AttendKind::Gqa => r.lists.div_ceil(4),
    };
    Some(Load {
        threads: blocks * r.threads,
        group: r.threads,
        trips: r.splits as f64,
        reads: vec![(4 * r.lists * r.splits * (r.out_width + 4)) as f64],
        written: (4 * r.lists * r.out_width) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn red_launches(r: &SplitReduce, step: &Step) -> Vec<Launch> {
    let across = match r.kind {
        AttendKind::Mla => r.lists,
        AttendKind::Gqa => r.lists.div_ceil(4),
    };
    let shape = Shape::Grid {
        across,
        down: 1,
        threads: r.threads,
    };
    let mut args = vec![
        step.read(0),
        step.out(),
        Arg::count(r.lists),
        Arg::count(r.splits),
    ];
    if r.kind == AttendKind::Mla {
        args.push(Arg::count(r.out_width));
    }
    vec![Launch { args, shape }]
}

fn page_shift(kept: &'static dyn Layout) -> usize {
    if kept.name() == "ring" {
        RING_SHIFT
    } else {
        kept.page().map_or(0, |p| p.trailing_zeros() as usize)
    }
}

fn kept_stride(kept: &'static dyn Layout, context: usize) -> Option<(usize, &'static dyn Layout)> {
    if kept.name() == "dense" || kept.name() == "ring" {
        return Some((context, kept));
    }
    let page = kept.page()?;
    Some((context.div_ceil(page), kept))
}

fn kept_context(tree: &Tree, k: T, v: T, most: usize) -> Option<(usize, &'static dyn Layout)> {
    match (&tree.node(k).term, &tree.node(v).term) {
        (Term::Op(Op::Append(ki)), Term::Op(Op::Append(vi))) => {
            let (_, layout, ctx) = tree.kept_layout(*ki)?;
            let (_, vlayout, vctx) = tree.kept_layout(*vi)?;
            if vlayout != layout || vctx != ctx {
                return None;
            }
            kept_stride(layout, ctx)
        }
        (Term::Op(Op::Append(_)), _) | (_, Term::Op(Op::Append(_))) => None,
        _ => Some((most, kept::DENSE)),
    }
}

/// Declines the site with a reason when `cond` is false.
macro_rules! require {
    (f32 $tree:expr, $($what:literal $t:expr),+) => {
        $(
            let form = $tree.node($t).form.name();
            require!(form == "f32", "{} form {} (requires f32)", $what, form);
        )+
    };
    ($cond:expr, $($why:tt)+) => {
        if !$cond {
            return Some(Err(format!($($why)+)));
        }
    };
}

fn check(tree: &Tree, at: T, kind: AttendKind) -> Option<Result<Vec<Draft>, String>> {
    let outer = tree.node(at);
    let (attend_t, out_width) = if kind == AttendKind::Mla
        && let Term::Op(Op::View(View::Slice { axis: 2, start, .. })) = &outer.term
        && outer.shape.last() == Some(&MLA_LATENT)
        && start.value(&tree.g.dims, &|d| tree.size(d)) == Some(0)
        && !tree.root(outer.args[0])
        && matches!(tree.node(outer.args[0]).term, Term::Op(Op::Attend(_)))
    {
        (outer.args[0], MLA_LATENT)
    } else {
        (at, kind.width())
    };
    let node = tree.node(attend_t);
    let Term::Op(Op::Attend(a)) = &node.term else {
        return None;
    };
    let (q, k, v) = (node.args[0], node.args[1], node.args[2]);
    // MLA reads one latent cache as keys and values.
    let own = match kind {
        AttendKind::Mla => k == v,
        AttendKind::Gqa => !a.picked,
    };
    let (&[rows, heads, width], &[_, most, kv_heads, kw]) =
        (&tree.node(q).shape[..], &tree.node(k).shape[..])
    else {
        return None;
    };
    if !a.ragged || !own {
        return None;
    }
    require!(f32 tree, "output" at, "attend" attend_t, "query" q);
    let (kv_form, vf) = (tree.node(k).form, tree.node(v).form.name());
    let kv = kv_form.name();
    require!(vf == kv, "key form {kv} and value form {vf} differ");
    let entry = kind.kv_entry(kv, out_width);
    require!(entry.is_some(), "kv form {kv} (supports fp8, bf16, f32)");
    let want = kind.width();
    let v_width = tree.node(v).shape.last() == Some(&want);
    require!(
        width == want && kw == want && v_width,
        "head width {width} (requires {want})"
    );
    require!(rows > 0, "rows={rows} (requires at least 1)");
    let groups = kv_heads > 0 && heads.is_multiple_of(kv_heads);
    require!(
        groups,
        "heads={heads} not a multiple of kv_heads={kv_heads}"
    );
    let [slots, positions] = [node.args[3], node.args[4]];
    let sinks = match a.sinks_arg() {
        Some(si) => {
            let s = node.args[si];
            require!(f32 tree, "sinks" s);
            s
        }
        None => q,
    };
    let (reads, picks, hgroups) = match kind {
        AttendKind::Mla => {
            require!(kv_heads == 1, "kv_heads={kv_heads} (requires 1)");
            let fits = heads >= 16 && heads.is_multiple_of(16);
            require!(fits, "heads={heads} (requires a multiple of 16)");
            let (ids, picks) = match a.ids_arg() {
                Some(i) => {
                    let ids = node.args[i];
                    (ids, *tree.node(ids).shape.last()?)
                }
                None => (positions, most),
            };
            ([q, k, ids, slots, positions, sinks], picks, heads / 16)
        }
        AttendKind::Gqa => {
            let gqa = heads / kv_heads;
            let fits = gqa >= 4 && (gqa <= 16 || gqa.is_multiple_of(16));
            require!(fits, "{gqa} heads per kv head (requires 4..=16 or n*16)");
            let hgroups = kv_heads * gqa.div_ceil(16);
            ([q, k, v, slots, positions, sinks], most, hgroups)
        }
    };
    let cache = kept_context(tree, k, v, most);
    require!(
        cache.is_some(),
        "keys and values share no dense, ring, or paged cache"
    );
    let (context, kept) = cache?;
    let entry = entry?;
    let rule = kind.names().0;
    let splits = tree
        .knobs
        .get(rule, "splits")
        .unwrap_or_else(|| split_count(kind, rows, hgroups));
    let min_blocks = tree.knobs.get(rule, "min_blocks").map(|v| v as u32);
    let reduce_min_blocks = tree.knobs.get(rule, "reduce_min_blocks").map(|v| v as u32);
    let (red_threads, custom_threads) = match (kind, tree.knobs.get(rule, "reduce_threads")) {
        (AttendKind::Mla, Some(t)) => {
            require!(
                t.is_multiple_of(32),
                "reduce_threads={t} (requires a multiple of 32)"
            );
            (t, true)
        }
        (AttendKind::Mla, None) => (160, false),
        (AttendKind::Gqa, _) => (128, false),
    };
    let attend = SplitAttend {
        kind,
        rows,
        heads,
        kv_heads,
        out_width,
        keys: tree.node(k).extents[1],
        context,
        kept,
        picks,
        from_list: a.picked,
        has_sinks: a.sinks,
        scale: a.scale,
        window: a.mask.window(most),
        splits,
        min_blocks,
        kv_form,
        entry,
    };
    let lists = rows * heads;
    if splits == 1 {
        return Some(Ok(vec![Draft::new(attend, reads, lists * out_width)]));
    }
    let stride = attend.stride();
    let reduce = SplitReduce {
        kind,
        lists,
        splits,
        out_width,
        threads: red_threads,
        custom_threads,
        min_blocks: reduce_min_blocks,
    };
    Some(Ok(vec![
        Draft::new(attend, reads, lists * splits * stride),
        Draft::new(reduce, [Read::Draft(0)], lists * out_width),
    ]))
}

fn check_mla(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    check(tree, at, AttendKind::Mla)
}

fn check_gqa(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    check(tree, at, AttendKind::Gqa)
}

cuda_rule!("fast_mla_attend", check = check_mla);
cuda_rule!("fast_gqa_attend", check = check_gqa);

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

/// Tensor-Core Lightning Indexer over dense, ring, or paged keys.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "fast_indexer",
    sizes = [self.rows, self.out_keys, self.heads, self.dim],
    work = self.rows * self.out_keys * self.heads * self.dim,
    accepts = idx_accepts,
    emits = f32_only,
    needs = idx_needs(self),
    order = Order::PerConfig,
    source = attend_entry(
        self.entry,
        match self.qt {
            true => self.qt_min_blocks.map(|v| ("IDX_QT_MIN_BLOCKS", v as usize)),
            false => self.min_blocks.map(|v| ("IDX_MIN_BLOCKS", v as usize)),
        },
    ),
    launches = idx_launches,
    load = idx_load, load_key = tessera_lower::load_key(((self.rows, self.out_keys, self.keys, self.context, self.kept.name(), self.heads), (self.dim, self.masked, self.qt, self.tile_keys, self.min_blocks, self.qt_min_blocks), (self.k_form.name(), self.entry))),
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
    tile_keys: usize,
    min_blocks: Option<u32>,
    qt_min_blocks: Option<u32>,
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
    let tk = k.tile_keys;
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
    let tk = k.tile_keys;
    let shape = Shape::Grid {
        across: k.out_keys.div_ceil(tk),
        down: if k.qt { k.rows.div_ceil(8) } else { k.rows },
        threads: if k.qt { 256 } else { 128 },
    };
    let page_shift = page_shift(k.kept);
    let table = step.kept_for(k.kept, true).unwrap_or_else(|| step.read(3));
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

/// Matches stepped ragged Lightning Indexer over keys.
fn check_indexer(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
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
    if w_node.shape != [rows, 1, heads] || rows == 0 {
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
    let &[q_rows, q_heads, dim] = &q_node.shape[..] else {
        return None;
    };
    if q_rows != rows || q_heads != heads || tree.elements(at) != rows * out_keys {
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
    let per_row = |t: T| tree.elements(t) == rows && tree.node(t).form.dtype() == DType::U32;
    if k_keys != out_keys || k_dim != dim || !per_row(slots) || !masked_pos.is_none_or(per_row) {
        return None;
    }
    let (_, raw_ctx) = tree.kept_rows(ki)?;
    require!(f32 tree, "output" at, "query" q, "head weight" by_head);
    require!(!q_node.extents[0].runtime(), "query rows are runtime-sized");
    let fits = (16..=IDX_MAX_HEADS).contains(&heads) && heads.is_multiple_of(16);
    require!(fits, "heads={heads} (requires n*16, max {IDX_MAX_HEADS})");
    let fits = dim >= 16 && dim.is_multiple_of(16);
    require!(fits, "dim={dim} (requires a multiple of 16)");
    let fits = heads.max(64) * (dim / 2 + 2) <= IDX_SM_WORDS;
    require!(fits, "heads={heads}, dim={dim} exceed shared memory");
    let kf = k_node.form.name();
    let qt_min_rows = tree.knobs.get("fast_indexer", "qt_min_rows").unwrap_or(8);
    let qt = rows >= qt_min_rows && heads == 32 && dim == 128;
    let tile_keys = if let Some(v) = tree.knobs.get("fast_indexer", "tile_keys") {
        v
    } else if qt {
        tree.knobs
            .get("fast_indexer", "qt_tile_keys")
            .unwrap_or(1024)
    } else {
        let small = tree.knobs.get("fast_indexer", "small_rows").unwrap_or(4);
        if rows <= small {
            tree.knobs
                .get("fast_indexer", "small_tile_keys")
                .unwrap_or(64)
        } else {
            tree.knobs
                .get("fast_indexer", "large_tile_keys")
                .unwrap_or(256)
        }
    };
    let align = if qt { 32 } else { 16 };
    require!(
        tile_keys >= align && tile_keys.is_multiple_of(align),
        "tile_keys={tile_keys} (requires a multiple of {align})"
    );
    let entry = idx_kv_entry(kf, qt);
    require!(entry.is_some(), "key form {kf} (supports fp8, bf16, f32)");
    let paged = kept_stride(tree.kept, raw_ctx);
    let layout = tree.kept.name();
    require!(
        paged.is_some(),
        "kv layout {layout} is neither dense, ring, nor paged"
    );
    let ((context, kept), entry) = (paged?, entry?);
    let keys = k_node.extents[1];
    let (masked, positions) = match masked_pos {
        Some(pos) => (true, pos),
        None => {
            require!(!keys.runtime(), "runtime key count needs a causal mask");
            (false, slots)
        }
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
        tile_keys,
        min_blocks: tree
            .knobs
            .get("fast_indexer", "min_blocks")
            .map(|v| v as u32),
        qt_min_blocks: tree
            .knobs
            .get("fast_indexer", "qt_min_blocks")
            .map(|v| v as u32),
        k_form: k_node.form,
        entry,
    };
    let reads = [q, k_append, by_head, slots, positions];
    Some(Ok(vec![Draft::new(kernel, reads, rows * out_keys)]))
}

cuda_rule!("fast_indexer", check = check_indexer);
