//! Native Metal grouped-Q split-K MLA attention kernels.

use crate::indexer_metal::{k_fmt_idx, kept_params, plain_fmt};
use tessera_codegen::kept::{Dense, Layout};
use tessera_graph::{Op, Order, View};
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::{
    Arg, Draft, Extent, Kernel, Launch, Load, Needs, Read, Shape, Source, Step, T, Term, Tree,
};

const MMA_THREADS: usize = 512;
const GEN_THREADS: usize = 256;
const MLA_WIDTH: usize = 576;
const MLA_LATENT: usize = 512;
const MAX_GEN_WIDTH: usize = 1024;
const SOURCE: &str = concat!(include_str!("kv.metal"), include_str!("mla_metal.metal"));
/// Low default guess so Metal prefers native MLA attention.
const GUESS: f64 = 1.0e-12;

fn select_entry(
    width: usize,
    out_width: usize,
    kv_form: &dyn Stored,
) -> Option<(&'static str, usize, usize, usize)> {
    let fi = k_fmt_idx(kv_form)?;
    if width == MLA_WIDTH && out_width == MLA_LATENT {
        const NAMES: [&str; 4] = [
            "mla_mma_512_f32",
            "mla_mma_512_bf16",
            "mla_mma_512_f16",
            "mla_mma_512_fp8",
        ];
        let smem = if fi == 0 { 24_576 } else { 28_672 };
        return Some((NAMES[fi], 16, MMA_THREADS, smem));
    }
    if width == MLA_WIDTH && out_width == MLA_WIDTH {
        const NAMES: [&str; 4] = [
            "mla_mma_576_f32",
            "mla_mma_576_bf16",
            "mla_mma_576_f16",
            "mla_mma_576_fp8",
        ];
        let smem = if fi == 0 { 24_576 } else { 28_672 };
        return Some((NAMES[fi], 16, MMA_THREADS, smem));
    }
    if width <= 256 {
        const NAMES: [&str; 4] = [
            "mla_gen_256_f32",
            "mla_gen_256_bf16",
            "mla_gen_256_f16",
            "mla_gen_256_fp8",
        ];
        return Some((NAMES[fi], 8, GEN_THREADS, 5_120));
    }
    if width <= MAX_GEN_WIDTH {
        const NAMES: [&str; 4] = [
            "mla_gen_1024_f32",
            "mla_gen_1024_bf16",
            "mla_gen_1024_f16",
            "mla_gen_1024_fp8",
        ];
        return Some((NAMES[fi], 8, GEN_THREADS, 17_408));
    }
    None
}

fn split_kv(rows: usize, hgroups: usize, seen: usize) -> usize {
    if seen < 256 {
        return 1;
    }
    let groups = (rows * hgroups).max(1);
    let goal: usize = if seen >= 4096 { 64 } else { 32 };
    let max_splits = (seen / 128).clamp(1, 32);
    goal.div_ceil(groups).clamp(1, max_splits)
}

#[derive(Clone, Copy, Debug)]
struct Spec {
    rows: usize,
    keys: Extent,
    context: usize,
    heads: usize,
    width: usize,
    out_width: usize,
    picks: usize,
    from_list: bool,
    has_sinks: bool,
    scale: f32,
    window: (usize, usize),
    splits: usize,
    group_heads: usize,
    threads: usize,
    smem: usize,
    kept: &'static dyn Layout,
    q_form: &'static dyn Stored,
    kv_form: &'static dyn Stored,
    out_form: &'static dyn Stored,
    entry: &'static str,
}

impl Spec {
    fn hgroups(&self) -> usize {
        self.heads.div_ceil(self.group_heads)
    }

    fn tile_keys(&self) -> usize {
        if self.group_heads >= 16 {
            if self.kv_form.name() == "f32" { 8 } else { 16 }
        } else {
            4
        }
    }
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "mla_metal",
    sizes = [self.0.rows, self.0.heads, self.0.width, self.0.out_width, self.0.keys.most(), self.0.picks, self.0.splits],
    work = self.0.rows * self.0.heads * self.0.picks.min(self.0.keys.most()) * self.0.width,
    guess = GUESS,
    accepts = mla_accepts,
    emits = mla_emits,
    needs = mla_needs(&self.0),
    order = Order::PerConfig,
    source = Source::metal(SOURCE, self.0.entry),
    launches = mla_launches,
    load = mla_load,
)]
struct MetalMla(Spec);

fn mla_accepts(k: &MetalMla, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.0.q_form.name(),
        1 => form.name() == k.0.kv_form.name(),
        2 if !k.0.from_list => true,
        5 if !k.0.has_sinks => true,
        _ => !form.narrow(),
    }
}

fn mla_emits(k: &MetalMla, form: &dyn Stored) -> bool {
    if k.0.splits > 1 {
        form.name() == "f32"
    } else {
        form.name() == k.0.out_form.name()
    }
}

fn mla_needs(s: &Spec) -> Needs {
    Needs::metal(s.threads, s.smem)
}

fn mla_load(k: &MetalMla, _: usize) -> Option<Load> {
    let s = &k.0;
    let hgroups = s.hgroups();
    let groups = s.rows * hgroups * s.splits;
    let seen = s.picks.min(s.keys.most()).max(1);
    let q_bytes = s.q_form.bytes(s.rows * s.heads * s.width) as f64;
    let kv_bytes = s.kv_form.bytes(s.rows * hgroups * seen * s.width) as f64;
    let out_bytes = if s.splits == 1 {
        s.out_form.bytes(s.rows * s.heads * s.out_width) as f64
    } else {
        (4 * s.rows * s.heads * s.splits * (s.out_width + 2)) as f64
    };
    Some(Load {
        threads: groups * s.threads,
        group: s.threads,
        trips: seen.div_ceil(s.tile_keys() * s.splits) as f64,
        reads: vec![kv_bytes, q_bytes],
        written: out_bytes,
        flight: 512.0,
        ..Load::default()
    })
}

fn mla_launches(k: &MetalMla, step: &Step) -> Vec<Launch> {
    let s = &k.0;
    let (kept_mode, kept_param, page_shift, page_mask) =
        kept_params(s.kept, s.context).expect("verified layout");
    let page_table = step.kept_for(s.kept, false).unwrap_or_else(|| step.read(3));
    let q_fmt = plain_fmt(s.q_form).expect("verified q_form");
    let out_fmt = if s.splits == 1 {
        plain_fmt(s.out_form).expect("verified out_form")
    } else {
        0
    };
    let flags = kept_mode
        | (usize::from(s.from_list) << 2)
        | (usize::from(s.has_sinks) << 3)
        | (q_fmt << 4)
        | (out_fmt << 6);
    let args = vec![
        step.read(0),
        step.read(1),
        step.read(2),
        step.read(3),
        step.read(4),
        step.read(5),
        page_table,
        step.out(),
        Arg::value(&s.scale),
        Arg::count(s.rows),
        Arg::size(s.keys),
        Arg::count(s.heads),
        Arg::count(s.width),
        Arg::count(s.out_width),
        Arg::count(s.picks),
        Arg::count(s.window.0),
        Arg::count(s.window.1),
        Arg::count(s.splits),
        Arg::count(kept_param),
        Arg::count(page_shift),
        Arg::count(page_mask),
        Arg::count(flags),
    ];
    let shape = Shape::Grid {
        across: s.splits,
        down: s.rows * s.hgroups(),
        threads: s.threads,
    };
    vec![Launch { args, shape }]
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "mla_metal_reduce",
    source = Source::metal(SOURCE, "mla_reduce"),
    order = Order::PerConfig,
    sizes = [self.lists, self.splits, self.out_width],
    work = self.lists * self.splits * self.out_width,
    guess = GUESS,
    accepts = red_accepts,
    emits = red_emits,
    needs = red_needs(),
    load = red_load,
    launches = red_launches,
)]
struct MetalMlaReduce {
    lists: usize,
    splits: usize,
    out_width: usize,
    out_form: &'static dyn Stored,
}

fn red_accepts(_: &MetalMlaReduce, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn red_emits(r: &MetalMlaReduce, form: &dyn Stored) -> bool {
    form.name() == r.out_form.name()
}

fn red_needs() -> Needs {
    Needs::metal(GEN_THREADS, 0)
}

fn red_load(r: &MetalMlaReduce, _: usize) -> Option<Load> {
    let part = 4 * r.lists * r.splits * (r.out_width + 2);
    let out = r.out_form.bytes(r.lists * r.out_width);
    Some(Load {
        threads: r.lists * GEN_THREADS,
        group: GEN_THREADS,
        trips: r.splits as f64,
        reads: vec![part as f64],
        written: out as f64,
        flight: 128.0,
        ..Load::default()
    })
}

fn red_launches(r: &MetalMlaReduce, step: &Step) -> Vec<Launch> {
    let out_fmt = plain_fmt(r.out_form).expect("verified out_form");
    let args = vec![
        step.read(0),
        step.out(),
        Arg::count(r.lists),
        Arg::count(r.splits),
        Arg::count(r.out_width),
        Arg::count(out_fmt),
    ];
    let shape = Shape::Grid {
        across: r.lists,
        down: 1,
        threads: GEN_THREADS,
    };
    vec![Launch { args, shape }]
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let outer = tree.node(at);
    let (attend_t, sliced_out) = if let Term::Op(Op::View(View::Slice { axis: 2, start, .. })) =
        &outer.term
        && start.value(&tree.g.dims, &|d| tree.size(d)) == Some(0)
        && !tree.root(outer.args[0])
        && matches!(tree.node(outer.args[0]).term, Term::Op(Op::Attend(_)))
    {
        (outer.args[0], *outer.shape.last()?)
    } else {
        (at, 0)
    };
    let node = tree.node(attend_t);
    let Term::Op(Op::Attend(a)) = &node.term else {
        return None;
    };
    let (q, k, v) = (node.args[0], node.args[1], node.args[2]);
    if !a.ragged || k != v {
        return None;
    }
    let (&[rows, heads, width], &[_, most, kv_heads, kw]) =
        (&tree.node(q).shape[..], &tree.node(k).shape[..])
    else {
        return None;
    };
    if kv_heads != 1 || kw != width || rows == 0 || heads == 0 || !width.is_multiple_of(4) {
        return None;
    }
    let out_width = if sliced_out > 0 { sliced_out } else { width };
    if out_width == 0 || out_width > width || !out_width.is_multiple_of(4) {
        return None;
    }
    let q_form = plain(tree.node(q).form);
    let kv_form = plain(tree.node(k).form);
    let out_form = writable(outer.form);
    plain_fmt(q_form)?;
    plain_fmt(out_form)?;
    let (entry, group_heads, threads, smem) = select_entry(width, out_width, kv_form)?;
    let (kept, context): (&'static dyn Layout, usize) = match &tree.node(k).term {
        Term::Op(Op::Append(ki)) => {
            let (_, layout, ctx) = tree.kept_layout(*ki)?;
            (layout, ctx)
        }
        _ => (&Dense, most),
    };
    kept_params(kept, context)?;
    let [slots, positions] = [node.args[3], node.args[4]];
    let (ids, picks) = match a.ids_arg() {
        Some(i) => {
            let ids = node.args[i];
            (ids, *tree.node(ids).shape.last()?)
        }
        None => (positions, most),
    };
    let sinks = match a.sinks_arg() {
        Some(si) => {
            let s = node.args[si];
            if tree.node(s).form.name() != "f32" {
                return None;
            }
            s
        }
        None => q,
    };
    let keys = tree.node(k).extents[1];
    let hgroups = heads.div_ceil(group_heads);
    let seen = picks.min(keys.most());
    let splits = split_kv(rows, hgroups, seen);
    let spec = Spec {
        rows,
        keys,
        context,
        heads,
        width,
        out_width,
        picks,
        from_list: a.picked,
        has_sinks: a.sinks,
        scale: a.scale,
        window: a.mask.window(most),
        splits,
        group_heads,
        threads,
        smem,
        kept,
        q_form,
        kv_form,
        out_form,
        entry,
    };
    let lists = rows * heads;
    let out_len = lists * out_width;
    let reads = [q, k, ids, slots, positions, sinks];
    if splits == 1 {
        Some(vec![Draft::new(MetalMla(spec), reads, out_len)])
    } else {
        let part_len = lists * splits * (out_width + 2);
        let reduce = MetalMlaReduce {
            lists,
            splits,
            out_width,
            out_form,
        };
        Some(vec![
            Draft::new(MetalMla(spec), reads, part_len),
            Draft::new(reduce, [Read::Draft(0)], out_len),
        ])
    }
}

tessera_lower::rule!("mla_metal", propose);
