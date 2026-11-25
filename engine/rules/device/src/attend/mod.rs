//! Tensor-Core grouped-Q split-K MLA and GQA attention rules for CUDA.

use crate::linear::{CUDA_LACKS, f32_only};
use crate::{Backend, Options, flatten};
use std::sync::OnceLock;
use tessera_graph::{Op, Order};
use tessera_lower::form::Load as Stored;
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
            AttendKind::Mla => MLA_WIDTH,
            AttendKind::Gqa => GQA_WIDTH,
        }
    }

    fn stride(self) -> usize {
        self.width() + 4
    }

    fn smem(self) -> usize {
        match self {
            AttendKind::Mla => 42_240,
            AttendKind::Gqa => 18_432,
        }
    }

    fn attend_name(self) -> &'static str {
        match self {
            AttendKind::Mla => "fast_mla_attend",
            AttendKind::Gqa => "fast_gqa_attend",
        }
    }

    fn reduce_name(self) -> &'static str {
        match self {
            AttendKind::Mla => "fast_mla_reduce",
            AttendKind::Gqa => "fast_gqa_reduce",
        }
    }

    fn reduce_entry(self) -> &'static str {
        match self {
            AttendKind::Mla => "tessera_mla_reduce",
            AttendKind::Gqa => "tessera_gqa_reduce",
        }
    }

    fn kv_entry(self, kv: &str) -> Option<&'static str> {
        match (self, kv) {
            (AttendKind::Mla, "fp8") => Some("tessera_mla_attend_fp8"),
            (AttendKind::Mla, "bf16") => Some("tessera_mla_attend_bf16"),
            (AttendKind::Mla, "f32") => Some("tessera_mla_attend_f32"),
            (AttendKind::Gqa, "fp8") => Some("tessera_gqa_attend_fp8"),
            (AttendKind::Gqa, "bf16") => Some("tessera_gqa_attend_bf16"),
            (AttendKind::Gqa, "f32") => Some("tessera_gqa_attend_f32"),
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

/// Grouped-Q split-K MLA or GQA attention over dense KV.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.kind.attend_name(),
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
        group_threads: 128,
        compiler: true,
        lacks: CUDA_LACKS,
        bf16_tile: true,
        ..Needs::default()
    }
}

fn attend_load(k: &SplitAttend, _: usize) -> Option<Load> {
    let w = k.kind.width();
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
        threads: groups * 128,
        group: 128,
        trips: seen.div_ceil(16 * k.splits) as f64,
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
        threads: 128,
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
    args.extend([
        Arg::count(k.window.0),
        Arg::count(k.window.1),
        Arg::count(k.splits),
    ]);
    vec![Launch { args, shape }]
}

/// Combines split-K MLA or GQA partials into final outputs.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.kind.reduce_name(),
    sizes = [self.lists, self.splits],
    work = self.lists * self.splits * self.kind.width(),
    accepts = red_accepts,
    emits = f32_only,
    needs = red_needs(self),
    order = Order::PerConfig,
    source = attend_entry(self.kind.reduce_entry()),
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

fn dense_context(tree: &Tree, k: T, v: T, most: usize) -> Option<usize> {
    match (&tree.node(k).term, &tree.node(v).term) {
        (Term::Op(Op::Append(ki)), Term::Op(Op::Append(vi))) => {
            if tree.kept.name() != "dense" {
                return None;
            }
            let ctx = tree.kept_rows(*ki)?.1;
            (tree.kept_rows(*vi)?.1 == ctx).then_some(ctx)
        }
        (Term::Op(Op::Append(_)), _) | (_, Term::Op(Op::Append(_))) => None,
        _ => Some(most),
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
    if rows == 0 || rows > 32 || kv_heads == 0 || !heads.is_multiple_of(kv_heads) {
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
    let context = dense_context(tree, k, v, most)?;
    let splits = split_count(kind, rows, hgroups);
    let attend = SplitAttend {
        kind,
        rows,
        heads,
        kv_heads,
        keys: tree.node(k).extents[1],
        context,
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
