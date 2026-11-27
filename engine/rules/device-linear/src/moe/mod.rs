//! CUDA Tensor-Core picked trellis matmul for MoE experts.

use super::{CUDA_LACKS, cuda_module, cuda_rule, io_defs, x_accepts};
use tessera_graph::{Act, Op, Order, Select, Stage};
use tessera_linear::parse_form;
use tessera_lower::form::Load as Stored;
use tessera_lower::sites::Rotated;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Term, Tree};

const KERNEL: &str = include_str!("trellis.cu");
const BLOCK: usize = 128;
const THREADS: usize = 256;
const SMEM: usize = 4096 * 4;

tessera_lower::knob!("picked_trellis", "chunk", 1, 4);
tessera_lower::knob!("picked_trellis", "min_blocks", 1, 16);
tessera_lower::knob!("picked_trellis", "gen_min_blocks", 1, 16);

cuda_module!(cuda_source, moe_source_with, "moe_trellis.kernel", KERNEL);

fn entry_name(k: &PickedTrellis) -> &'static str {
    if k.turned && k.bits == 3 && k.book == 2 {
        "ptr_k3_pmcgx_t"
    } else {
        "ptr_gen"
    }
}

fn trellis_source(k: &PickedTrellis) -> Source {
    let entry = entry_name(k);
    let mut base = Vec::new();
    if let Some(c) = k.chunk {
        base.push(("PTR_CHUNK", c as usize));
    }
    if entry == "ptr_k3_pmcgx_t" {
        if let Some(mb) = k.min_blocks {
            base.push(("PTR_K3_MIN_BLOCKS", mb as usize));
        }
    } else if let Some(mb) = k.gen_min_blocks {
        base.push(("PTR_GEN_MIN_BLOCKS", mb as usize));
    }
    moe_source_with(entry, io_defs(base, k.x_bf16, k.y_bf16, k.o_bf16))
}

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.form.suffixed(if self.turned { "picked_trellis_t" } else { "picked_trellis" }),
    sizes = [self.rows, self.n, self.k],
    work = self.rows * self.n * self.k,
    accepts = accepts,
    emits = emits,
    needs = needs(),
    order = Order::PerConfig,
    source = trellis_source(self),
    launches = launches,
    load = load,
    unproven,
)]
struct PickedTrellis {
    rows: usize,
    n: usize,
    k: usize,
    count: usize,
    repeat: usize,
    first: usize,
    bits: u32,
    book: u32,
    turned: bool,
    post: u32,
    chunk: Option<u32>,
    min_blocks: Option<u32>,
    gen_min_blocks: Option<u32>,
    x_bf16: bool,
    y_bf16: bool,
    o_bf16: bool,
    form: &'static dyn Stored,
}

fn accepts(k: &PickedTrellis, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => x_accepts(k.x_bf16, form),
        1 => form.name() == k.form.name(),
        2 => form.name() == "u32",
        5 => x_accepts(k.o_bf16, form),
        _ => form.name() == "f32",
    }
}

fn emits(k: &PickedTrellis, form: &dyn Stored) -> bool {
    x_accepts(k.y_bf16, form)
}

fn needs() -> Needs {
    Needs {
        lanes: 32,
        group_bytes: SMEM,
        group_threads: THREADS,
        compiler: true,
        bf16_tile: true,
        lacks: CUDA_LACKS,
        ..Needs::default()
    }
}

fn load(k: &PickedTrellis, _: usize) -> Option<Load> {
    let groups = (k.n / BLOCK) * k.rows;
    let x_elem = if k.x_bf16 { 2 } else { 4 };
    let o_elem = if k.o_bf16 { 2 } else { 4 };
    let y_elem = if k.y_bf16 { 2 } else { 4 };
    let mut reads = vec![
        k.form.bytes(k.rows * k.n * k.k) as f64,
        (x_elem * (k.rows / k.repeat) * k.k) as f64,
        (4 * k.rows) as f64,
    ];
    if k.turned {
        reads.extend([(4 * k.rows * k.k) as f64, (4 * k.rows * k.n) as f64]);
    }
    if k.post == 1 {
        reads.push((o_elem * k.n) as f64);
    } else if k.post == 2 {
        reads.push((o_elem * k.rows * k.n) as f64);
    }
    let mut load = Load {
        threads: groups * THREADS,
        group: THREADS,
        trips: (k.k / BLOCK) as f64,
        reads,
        written: (y_elem * k.rows * k.n) as f64,
        flight: 1024.0,
        ..Load::default()
    };
    load.trip.fresh = 1.0;
    Some(load)
}

fn launches(k: &PickedTrellis, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: k.n / BLOCK,
        down: k.rows,
        threads: THREADS,
    };
    let su = if k.turned { step.read(3) } else { step.read(0) };
    let sv = if k.turned { step.read(4) } else { step.read(0) };
    let extra = if k.post != 0 {
        step.read(5)
    } else {
        step.read(0)
    };
    let args = vec![
        step.read(0),
        step.read(1),
        step.read(2),
        su,
        sv,
        extra,
        step.out(),
        Arg::count(k.rows),
        Arg::count(k.n),
        Arg::count(k.k),
        Arg::count(k.count),
        Arg::count(k.repeat),
        Arg::count(k.first),
        Arg::value(&k.bits),
        Arg::value(&k.book),
        Arg::count(usize::from(k.turned)),
        Arg::value(&k.post),
    ];
    vec![Launch { args, shape }]
}

fn gather_of(tree: &Tree, t: T) -> Option<(T, T)> {
    let node = tree.node(t);
    if node.term != Term::Op(Op::Select(Select::Gather)) || tree.root(t) {
        return None;
    }
    Some((node.args[0], node.args[1]))
}

fn match_pick(tree: &Tree, x: T, pick: T, turned: bool) -> Option<(PickedTrellis, T, T)> {
    let (codes, ids) = gather_of(tree, pick)?;
    let x_name = tree.node(x).form.name();
    if !tree.leaf(codes) || !matches!(x_name, "f32" | "bf16") || tree.node(ids).form.name() != "u32"
    {
        return None;
    }
    let x_bf16 = x_name == "bf16";
    let &[count, n, k] = tree.node(codes).shape.as_slice() else {
        return None;
    };
    let form = tree.node(codes).form;
    let (bits, book) = parse_form(form.name())?;
    if !n.is_multiple_of(BLOCK) || !k.is_multiple_of(BLOCK) {
        return None;
    }
    let rows = tree.elements(ids);
    let x_rows = tree.elements(x) / k;
    if rows == 0 || x_rows == 0 || !rows.is_multiple_of(x_rows) {
        return None;
    }
    let first = tree
        .share(codes)
        .filter(|s| s.axis == 0)
        .map_or(0, |s| s.first);
    let kernel = PickedTrellis {
        rows,
        n,
        k,
        count,
        repeat: rows / x_rows,
        first,
        bits,
        book,
        turned,
        post: 0,
        chunk: tree.knobs.get("picked_trellis", "chunk").map(|v| v as u32),
        min_blocks: tree
            .knobs
            .get("picked_trellis", "min_blocks")
            .map(|v| v as u32),
        gen_min_blocks: tree
            .knobs
            .get("picked_trellis", "gen_min_blocks")
            .map(|v| v as u32),
        x_bf16,
        y_bf16: false,
        o_bf16: false,
        form,
    };
    Some((kernel, codes, ids))
}

fn match_turned(tree: &Tree, at: T) -> Option<(PickedTrellis, [T; 5])> {
    if !matches!(tree.node(at).form.name(), "f32" | "bf16") {
        return None;
    }
    let r = tree
        .site::<Rotated>(at)
        .filter(|r| r.block as usize == BLOCK)?;
    let (x, pick, su_emb, sv_emb) = (r.x, r.w, r.suh, r.svh);
    let (su, su_ids) = gather_of(tree, su_emb)?;
    let (sv, sv_ids) = gather_of(tree, sv_emb)?;
    let (mut kernel, codes, ids) = match_pick(tree, x, pick, true)?;
    kernel.y_bf16 = tree.node(at).form.name() == "bf16";
    let ok = su_ids == ids
        && sv_ids == ids
        && tree.leaf(su)
        && tree.leaf(sv)
        && kernel.first == 0
        && tree.node(su).shape == [kernel.count, kernel.k]
        && tree.node(sv).shape == [kernel.count, kernel.n]
        && tree.node(su).form.name() == "f32"
        && tree.node(sv).form.name() == "f32";
    ok.then_some((kernel, [x, codes, ids, su, sv]))
}

/// Matches rotated or plain picked trellis MoE matmuls.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    if !matches!(node.form.name(), "f32" | "bf16") {
        return None;
    }
    let y_bf16 = node.form.name() == "bf16";
    if node.term == Term::Stage(Stage::Mul) {
        if let Some((mut kernel, reads)) = match_turned(tree, at) {
            kernel.y_bf16 = y_bf16;
            return Some(vec![Draft::new(kernel, reads, kernel.rows * kernel.n)]);
        }
        let (a, b) = (node.args[0], *node.args.get(1)?);
        let silu = Term::Stage(Stage::Act(Act::Silu));
        for (cand, other) in [(a, b), (b, a)] {
            if cand == other
                || tree.root(cand)
                || !matches!(tree.node(other).form.name(), "f32" | "bf16")
            {
                continue;
            }
            let o_bf16 = tree.node(other).form.name() == "bf16";
            if let Some(&[inner]) = tree.inner(cand, &silu)
                && !tree.root(inner)
                && let Some((mut k, [x, c, ids, su, sv])) = match_turned(tree, inner)
                && tree.elements(other) == k.rows * k.n
            {
                k.post = 2;
                k.y_bf16 = y_bf16;
                k.o_bf16 = o_bf16;
                return Some(vec![Draft::new(
                    k,
                    [x, c, ids, su, sv, other],
                    k.rows * k.n,
                )]);
            }
            if let Some((mut k, [x, c, ids, su, sv])) = match_turned(tree, cand)
                && tree.elements(other) == k.n
            {
                k.post = 1;
                k.y_bf16 = y_bf16;
                k.o_bf16 = o_bf16;
                return Some(vec![Draft::new(
                    k,
                    [x, c, ids, su, sv, other],
                    k.rows * k.n,
                )]);
            }
        }
        return None;
    }
    if node.term == Term::Op(Op::Matmul) {
        let (x, pick) = (node.args[0], node.args[1]);
        let (mut kernel, codes, ids) = match_pick(tree, x, pick, false)?;
        kernel.y_bf16 = y_bf16;
        let len = kernel.rows * kernel.n;
        return Some(vec![Draft::new(kernel, [x, codes, ids], len)]);
    }
    None
}

cuda_rule!("picked_trellis", propose);
