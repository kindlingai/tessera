//! Native Metal GQA split-K decode and FlashAttention-2 prefill.

use tessera_codegen::kept::{Dense, Layout};
use tessera_graph::{Mask, Op, Order};
use tessera_lower::Backend;
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, Draft, Extent, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Term, Tree,
};

const THREADS: usize = 256;
const BQ: usize = 16;
const BK: usize = 32;
const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];
const SOURCE: &str = include_str!("attn_metal.metal");
/// Low default guess so Metal prefers native attention kernels.
const GUESS: f64 = 1.0e-12;

const KV_FORMS: [&str; 4] = ["f32", "bf16", "f16", "fp8"];
const DEC_ENTRIES: [(usize, [&str; 4]); 3] = [
    (
        64,
        [
            "am_dec_64_f32",
            "am_dec_64_bf16",
            "am_dec_64_f16",
            "am_dec_64_fp8",
        ],
    ),
    (
        128,
        [
            "am_dec_128_f32",
            "am_dec_128_bf16",
            "am_dec_128_f16",
            "am_dec_128_fp8",
        ],
    ),
    (
        256,
        [
            "am_dec_256_f32",
            "am_dec_256_bf16",
            "am_dec_256_f16",
            "am_dec_256_fp8",
        ],
    ),
];
const PF_ENTRIES: [(usize, [&str; 4]); 3] = [
    (
        64,
        [
            "am_pf_64_f32",
            "am_pf_64_bf16",
            "am_pf_64_f16",
            "am_pf_64_fp8",
        ],
    ),
    (
        128,
        [
            "am_pf_128_f32",
            "am_pf_128_bf16",
            "am_pf_128_f16",
            "am_pf_128_fp8",
        ],
    ),
    (
        256,
        [
            "am_pf_256_f32",
            "am_pf_256_bf16",
            "am_pf_256_f16",
            "am_pf_256_fp8",
        ],
    ),
];

fn fixed(entry: &'static str) -> Source {
    Source::metal(SOURCE, entry)
}

fn find_entry(
    table: &[(usize, [&'static str; 4])],
    width: usize,
    form: &str,
) -> Option<&'static str> {
    let fi = KV_FORMS.iter().position(|f| *f == form)?;
    let (_, names) = table.iter().find(|(w, _)| *w == width)?;
    Some(names[fi])
}

fn split_kv(rows: usize, kv_heads: usize, max_keys: usize) -> usize {
    if max_keys < 512 {
        return 1;
    }
    let goal = if max_keys >= 8192 { 128usize } else { 64usize };
    let max_splits = (max_keys / 256).max(1);
    goal.div_ceil((rows * kv_heads).max(1)).clamp(1, max_splits)
}

fn kept_params(
    layout: &'static dyn Layout,
    context: usize,
) -> Option<(usize, usize, usize, usize)> {
    if let Some(page) = layout.page() {
        if !page.is_power_of_two() {
            return None;
        }
        let pages = context.div_ceil(page);
        Some((1, pages, page.trailing_zeros() as usize, page - 1))
    } else if layout.name() == "dense" {
        Some((0, context, 0, 0))
    } else {
        None
    }
}

fn pf_smem(width: usize) -> usize {
    let stride = width + 8;
    (BQ * stride + BK * stride) * 2 + BQ * 36 * 6 + BQ * 16
}

fn dec_smem(width: usize) -> usize {
    6 * width * 2 + (8 * width + 16) * 4
}

#[derive(Clone, Copy, Debug)]
struct Spec {
    rows: usize,
    keys: Extent,
    context: usize,
    heads: usize,
    kv_heads: usize,
    width: usize,
    scale: f32,
    kept: &'static dyn Layout,
    kv_form: &'static dyn Stored,
    prefill: bool,
    splits: usize,
    entry: &'static str,
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.0.prefill { "attn_metal_pf" } else { "attn_metal_dec" },
    sizes = [self.0.rows, self.0.keys.most(), self.0.heads, self.0.width],
    work = self.0.rows * self.0.heads * self.0.keys.most() * self.0.width,
    guess = GUESS,
    accepts = attn_accepts,
    emits = f32_only,
    needs = attn_needs(&self.0),
    order = Order::PerConfig,
    source = fixed(self.0.entry),
    launches = attn_launches,
    load = attn_load,
)]
struct MetalAttn(Spec);

fn f32_only<K>(_: &K, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn attn_accepts(k: &MetalAttn, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == "f32",
        1 | 2 => form.name() == k.0.kv_form.name(),
        _ => !form.narrow(),
    }
}

fn attn_needs(s: &Spec) -> Needs {
    Needs {
        lanes: 32,
        group_bytes: if s.prefill {
            pf_smem(s.width)
        } else {
            dec_smem(s.width)
        },
        group_threads: THREADS,
        tile: s.prefill,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn attn_load(k: &MetalAttn, _: usize) -> Option<Load> {
    let s = &k.0;
    let keys = s.keys.most() as f64;
    let kv_eb = s.kv_form.element_bytes();
    if s.prefill {
        let blocks = s.rows.div_ceil(BQ);
        let groups = blocks * s.heads;
        let kv_bytes = 2.0 * kv_eb * keys * (s.kv_heads * s.width) as f64 * blocks as f64;
        Some(Load {
            threads: groups * THREADS,
            group: THREADS,
            trips: keys / BK as f64,
            reads: vec![(4 * s.rows * s.heads * s.width) as f64, kv_bytes],
            written: (4 * s.rows * s.heads * s.width) as f64,
            flight: 256.0,
            group_bytes: pf_smem(s.width) as f64,
            ..Load::default()
        })
    } else {
        let groups = s.rows * s.kv_heads * s.splits;
        let g_chunks = (s.heads / s.kv_heads).div_ceil(6);
        let kv_bytes =
            2.0 * kv_eb * keys * (s.kv_heads * s.width) as f64 * (s.rows * g_chunks) as f64;
        Some(Load {
            threads: groups * THREADS,
            group: THREADS,
            trips: keys / (8 * s.splits) as f64,
            reads: vec![(4 * s.rows * s.heads * s.width) as f64, kv_bytes],
            written: (4 * s.rows * s.heads * s.splits * (s.width + 2)) as f64,
            flight: 256.0,
            group_bytes: dec_smem(s.width) as f64,
            ..Load::default()
        })
    }
}

fn attn_launches(k: &MetalAttn, step: &Step) -> Vec<Launch> {
    let s = &k.0;
    let (kept_mode, kept_param, page_shift, page_mask) =
        kept_params(s.kept, s.context).expect("checked kept layout");
    let page_table = step
        .kept(s.kept)
        .into_iter()
        .next()
        .unwrap_or_else(|| step.read(3));
    let across = if s.prefill {
        s.rows.div_ceil(BQ) * s.heads
    } else {
        s.rows * s.kv_heads * s.splits
    };
    let args = vec![
        step.read(0),
        step.read(1),
        step.read(2),
        step.read(3),
        step.read(4),
        page_table,
        step.out(),
        Arg::value(&s.scale),
        Arg::count(s.rows),
        Arg::size(s.keys),
        Arg::count(s.heads),
        Arg::count(s.kv_heads),
        Arg::count(s.splits),
        Arg::count(kept_mode),
        Arg::count(kept_param),
        Arg::count(page_shift),
        Arg::count(page_mask),
    ];
    vec![Launch {
        args,
        shape: Shape::Grid {
            across,
            down: 1,
            threads: THREADS,
        },
    }]
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "attn_metal_reduce",
    source = fixed("am_dec_reduce"),
    order = Order::PerConfig,
    sizes = [self.lists, self.splits, self.width],
    work = self.lists * self.splits * self.width,
    guess = GUESS,
    emits = f32_only,
    accepts = f32_only_in,
    needs = combine_needs(),
    load = combine_load,
    launches = combine_launches,
)]
struct MetalAttnCombine {
    lists: usize,
    splits: usize,
    width: usize,
}

fn f32_only_in(_: &MetalAttnCombine, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn combine_needs() -> Needs {
    Needs {
        lacks: LACKS,
        compiler: true,
        lanes: 32,
        group_threads: THREADS,
        ..Needs::default()
    }
}

fn combine_load(c: &MetalAttnCombine, _: usize) -> Option<Load> {
    let part = 4 * c.lists * c.splits * (c.width + 2);
    let out = 4 * c.lists * c.width;
    Some(Load {
        threads: c.lists * THREADS,
        group: THREADS,
        trips: c.splits as f64,
        reads: vec![part as f64],
        written: out as f64,
        flight: 128.0,
        ..Load::default()
    })
}

fn combine_launches(c: &MetalAttnCombine, step: &Step) -> Vec<Launch> {
    let args = vec![
        step.read(0),
        step.out(),
        Arg::count(c.lists),
        Arg::count(c.splits),
        Arg::count(c.width),
    ];
    let shape = Shape::Grid {
        across: c.lists,
        down: 1,
        threads: THREADS,
    };
    vec![Launch { args, shape }]
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Op(Op::Attend(a)) = &node.term else {
        return None;
    };
    if !a.ragged || a.picked || a.sinks || a.mask != Mask::Causal {
        return None;
    }
    if node.form.name() != "f32" {
        return None;
    }
    let (q, k, v, slots, positions) = (
        node.args[0],
        node.args[1],
        node.args[2],
        node.args[3],
        node.args[4],
    );
    if tree.node(q).form.name() != "f32" {
        return None;
    }
    let kv_form = tree.node(k).form;
    if tree.node(v).form.name() != kv_form.name() {
        return None;
    }
    let (&[rows, heads, width], &[_, most, kv_heads, _]) =
        (&tree.node(q).shape[..], &tree.node(k).shape[..])
    else {
        return None;
    };
    if tree.node(v).shape.last() != Some(&width) || kv_heads == 0 || !heads.is_multiple_of(kv_heads)
    {
        return None;
    }
    let (kept, context): (&'static dyn Layout, usize) =
        match (&tree.node(k).term, &tree.node(v).term) {
            (Term::Op(Op::Append(ki)), Term::Op(Op::Append(vi))) => {
                let (_, layout, ctx) = tree.kept_layout(*ki)?;
                let (_, vlayout, vctx) = tree.kept_layout(*vi)?;
                if vlayout != layout || vctx != ctx {
                    return None;
                }
                (layout, ctx)
            }
            (Term::Op(Op::Append(_)), _) | (_, Term::Op(Op::Append(_))) => return None,
            _ => (&Dense, most),
        };
    kept_params(kept, context)?;

    let prefill = rows > 16;
    // Dense width-128 f32/bf16 prefill defers to flash_sunk_metal.
    if prefill && kept.name() == "dense" && width == 128 && matches!(kv_form.name(), "f32" | "bf16")
    {
        return None;
    }
    let table = if prefill { &PF_ENTRIES } else { &DEC_ENTRIES };
    let entry = find_entry(table, width, kv_form.name())?;
    let keys = tree.node(k).extents[1];
    let splits = if prefill {
        1
    } else {
        split_kv(rows, kv_heads, keys.most())
    };
    let spec = Spec {
        rows,
        keys,
        context,
        heads,
        kv_heads,
        width,
        scale: a.scale,
        kept,
        kv_form,
        prefill,
        splits,
        entry,
    };
    let out_len = rows * heads * width;
    if splits == 1 {
        Some(vec![Draft::new(
            MetalAttn(spec),
            [q, k, v, slots, positions],
            out_len,
        )])
    } else {
        let lists = rows * heads;
        let part_len = lists * splits * (width + 2);
        let combine = MetalAttnCombine {
            lists,
            splits,
            width,
        };
        Some(vec![
            Draft::new(
                MetalAttn(spec),
                [
                    Read::Node(q),
                    Read::Node(k),
                    Read::Node(v),
                    Read::Node(slots),
                    Read::Node(positions),
                ],
                part_len,
            ),
            Draft::new(combine, [Read::Draft(0)], out_len),
        ])
    }
}

fn why(_: &Tree, _: T) -> Option<String> {
    None
}

tessera_lower::rule!("attn_metal", propose, why = why, lacks = LACKS);
