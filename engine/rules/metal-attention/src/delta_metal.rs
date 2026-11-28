//! Native Metal Gated DeltaNet and causal conv1d over ragged rows.

use tessera_graph::{Op, Order};
use tessera_lower::Backend;
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, ByBackend, Draft, Extent, Keeping, Kernel, KernelRef, Launch, Needs, Read, Shape, Source,
    Step, T, Term, Tree, sites,
};

const DM_THREADS: usize = 128;
const DM_DEC_THREADS: usize = 256;
const CM_THREADS: usize = 256;
const CM_DEC_THREADS: usize = 64;
const CM_BR: usize = 16;
const MAX_TAPS: usize = 16;
const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];
const SOURCE: &str = include_str!("delta_metal.metal");
/// Low default guess so Metal prefers native recurrence kernels.
const GUESS: f64 = 1.0e-13;

fn fixed(entry: &'static str) -> Source {
    Source::Fixed {
        text: ByBackend([SOURCE, "", ""]),
        entry,
    }
}

fn f32_only<K>(_: &K, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn delta_accepts(_: &MetalDeltaRows, input: usize, form: &dyn Stored) -> bool {
    match input {
        5 | 6 => !form.narrow(),
        _ => form.name() == "f32",
    }
}

fn delta_smem(key_width: usize, dec_f4: bool) -> usize {
    let chunk = if dec_f4 { 16 } else { 8 };
    (chunk * key_width * 2 + 64) * 4
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "delta_metal_rows",
    sizes = [self.site.seqs, self.site.rows.most(), self.site.heads, self.site.width],
    work = self.site.out() * self.site.key_width * 4,
    guess = GUESS,
    accepts = delta_accepts,
    emits = f32_only,
    needs = delta_needs(self),
    order = Order::PerConfig,
    source = fixed(self.entry),
    launches = delta_launches,
    commit = delta_commit,
)]
struct MetalDeltaRows {
    site: sites::Delta,
    keeping: Keeping,
    entry: &'static str,
    dec_f4: bool,
}

fn delta_needs(k: &MetalDeltaRows) -> Needs {
    Needs {
        lanes: 32,
        group_bytes: delta_smem(k.site.key_width, k.dec_f4),
        group_threads: if k.dec_f4 { DM_DEC_THREADS } else { DM_THREADS },
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn delta_launches(k: &MetalDeltaRows, step: &Step) -> Vec<Launch> {
    let d = &k.site;
    let (across, threads) = if k.dec_f4 {
        (d.seqs * d.heads * d.width.div_ceil(128), DM_DEC_THREADS)
    } else {
        (d.seqs * d.heads * d.width.div_ceil(32), DM_THREADS)
    };
    let args = vec![
        step.read(0),
        step.read(1),
        step.read(2),
        step.read(3),
        step.read(4),
        step.read(5),
        step.read(6),
        step.read(7),
        step.out(),
        k.keeping.keep(step),
        Arg::value(&d.scale),
        Arg::count(d.seqs),
        Arg::count(d.heads),
        Arg::count(d.kv_heads),
        Arg::count(d.width),
        Arg::size(k.keeping.rows(d.rows)),
        Arg::count(usize::from(k.keeping != Keeping::Peeks)),
    ];
    vec![Launch {
        args,
        shape: Shape::Grid {
            across,
            down: 1,
            threads,
        },
    }]
}

fn delta_commit(k: &MetalDeltaRows) -> Option<KernelRef> {
    k.keeping.twin(|keeping| MetalDeltaRows { keeping, ..*k })
}

fn propose_delta(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Delta = tree.site(at)?;
    let (idx, [slots, positions]) = site.state?;
    let dec_f4 = site.rows.most() <= 16 && site.key_width == 128 && site.width.is_multiple_of(16);
    let entry = match (site.key_width, dec_f4) {
        (128, true) => "dm_dec_128",
        (128, false) => "dm_delta_128",
        (64, _) => "dm_delta_64",
        _ => return None,
    };
    if !site.width.is_multiple_of(8) || site.width == 0 || site.kv_heads == 0 {
        return None;
    }
    if tree.node(at).form.name() != "f32" {
        return None;
    }
    for r in site.reads {
        if tree.node(r).form.name() != "f32" {
            return None;
        }
    }
    let len = site.out();
    let mut reads: Vec<Read> = site.reads.into_iter().map(Read::Node).collect();
    reads.extend([Read::Node(slots), Read::Node(positions), Read::Kept(idx)]);
    let kernel = MetalDeltaRows {
        site,
        keeping: Keeping::of(tree),
        entry,
        dec_f4,
    };
    Some(vec![Draft::new(kernel, reads, len)])
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "conv_metal_rows",
    source = fixed(self.entry),
    order = Order::PerConfig,
    sizes = [self.seqs, self.rows.most(), self.width, self.taps, self.dilation],
    work = self.rows.most() * self.width * self.taps * 2,
    guess = GUESS,
    emits = f32_only,
    accepts = conv_accepts,
    needs = conv_needs(self),
    launches = conv_launches,
    commit = conv_commit,
)]
struct MetalConvRows {
    seqs: usize,
    rows: Extent,
    width: usize,
    taps: usize,
    dilation: usize,
    w_bf16: bool,
    keeping: Keeping,
    entry: &'static str,
}

fn conv_accepts(k: &MetalConvRows, input: usize, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == if k.w_bf16 { "bf16" } else { "f32" },
        2 | 3 => !form.narrow(),
        _ => form.name() == "f32",
    }
}

fn conv_needs(k: &MetalConvRows) -> Needs {
    let threads = if k.rows.most() > 16 {
        CM_THREADS
    } else {
        CM_DEC_THREADS
    };
    Needs {
        lanes: 32,
        group_bytes: 64,
        group_threads: threads,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn conv_launches(k: &MetalConvRows, step: &Step) -> Vec<Launch> {
    let threads = if k.rows.most() > 16 {
        CM_THREADS
    } else {
        CM_DEC_THREADS
    };
    let gx = (k.width / 4).div_ceil(threads);
    let make = |mode: usize, down: usize| Launch {
        args: vec![
            step.read(0),
            step.read(1),
            step.read(2),
            step.read(3),
            step.read(4),
            step.out(),
            k.keeping.keep(step),
            Arg::count(k.seqs),
            Arg::count(k.width),
            Arg::count(k.taps),
            Arg::count(k.dilation),
            Arg::size(k.keeping.rows(k.rows)),
            Arg::count(mode),
            Arg::count(usize::from(k.w_bf16)),
            Arg::count(usize::from(k.keeping != Keeping::Peeks)),
        ],
        shape: Shape::Grid {
            across: gx,
            down,
            threads,
        },
    };
    if k.rows.most() > 16 {
        vec![make(0, k.rows.most().div_ceil(CM_BR)), make(1, k.seqs)]
    } else {
        vec![make(2, k.seqs)]
    }
}

fn conv_commit(k: &MetalConvRows) -> Option<KernelRef> {
    k.keeping.twin(|keeping| MetalConvRows { keeping, ..*k })
}

fn propose_conv(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Op(Op::Conv(c)) = node.term else {
        return None;
    };
    let idx = c.state?;
    let [x, w, slots, positions] = node.args[..] else {
        return None;
    };
    let w_bf16 = match tree.node(w).form.name() {
        "f32" => false,
        "bf16" => true,
        _ => return None,
    };
    if node.form.name() != "f32" || tree.node(x).form.name() != "f32" {
        return None;
    }
    let rows = tree.node(x).extents[0];
    let width = *node.shape.last()?;
    if tree.node(w).shape.len() != 2 {
        return None;
    }
    let taps = tree.node(w).shape[1];
    let dilation = c.dilation as usize;
    let hist = taps.checked_sub(1)? * dilation;
    if width == 0 || !width.is_multiple_of(4) || hist == 0 || hist > MAX_TAPS || taps > MAX_TAPS {
        return None;
    }
    let seqs = *tree.fixed_kept(idx)?.first()?;
    let entry = if taps <= 4 && hist <= 4 {
        "cm_conv_t4"
    } else {
        "cm_conv_rows"
    };
    let conv = MetalConvRows {
        seqs,
        rows,
        width,
        taps,
        dilation,
        w_bf16,
        keeping: Keeping::of(tree),
        entry,
    };
    let reads = [
        Read::Node(x),
        Read::Node(w),
        Read::Node(slots),
        Read::Node(positions),
        Read::Kept(idx),
    ];
    Some(vec![Draft::new(conv, reads, rows.most() * width)])
}

fn why(_: &Tree, _: T) -> Option<String> {
    None
}

tessera_lower::rule!("delta_metal", propose_delta, why = why, lacks = LACKS);
tessera_lower::rule!("conv_metal", propose_conv, why = why, lacks = LACKS);
