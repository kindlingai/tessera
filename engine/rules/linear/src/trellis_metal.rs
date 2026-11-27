//! Metal tile-streaming trellis matmul for decode and prefill.

use crate::{Dense, launch_args};
use tessera_graph::Order;
use tessera_lower::Backend;
use tessera_lower::form::Load as Stored;
use tessera_lower::sites::Coded;
use tessera_lower::{
    Arg, ByBackend, Draft, Kernel, Knobs, Launch, Load, Needs, Read, Shape, Source, Step, T, Tree,
    knob,
};
use tessera_mem::trellis::{parse_form, parse_mix_form};

const BLOCK: usize = 128;
const THREADS: usize = 256;
const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];
const SOURCE: &str = include_str!("trellis_metal.metal");
const DEC_SMEM: usize = 4 * 128 * 4;
const PF16_SMEM: usize = (16 * 132 + 8 * 16 * 20) * 4;
const PF32_SMEM: usize = (32 * 132 + 8 * 16 * 20) * 4;
const MAX_GROUP_EXPERTS: usize = 1024;
const GROUP_SMEM: usize = 2 * (MAX_GROUP_EXPERTS + 1) * 4;
const PF16_PICK_SMEM: usize = PF16_SMEM + 16 * 4;
const PF32_PICK_SMEM: usize = PF32_SMEM + 32 * 4;
const GUESS: f64 = 2.0e-11;

knob!("trellis_metal", "splits", 1, 16);
knob!("trellis_metal", "split_max_rows", 1, 4);
knob!("trellis_metal", "split_goal", 1, 128);
knob!("trellis_metal", "mid_rows", 4, 16);

fn split_k(knobs: &Knobs, m: usize, n: usize, k: usize) -> usize {
    let ibs = k / BLOCK;
    let max_rows = knobs.get("trellis_metal", "split_max_rows").unwrap_or(4);
    if m > max_rows || ibs <= 1 {
        return 1;
    }
    if let Some(s) = knobs.get("trellis_metal", "splits") {
        let s = s.min(ibs);
        if ibs.is_multiple_of(s) {
            return s;
        }
    }
    let default_goal = if m <= 2 { 40 } else { 20 };
    let goal = knobs
        .get("trellis_metal", "split_goal")
        .unwrap_or(default_goal);
    let want = goal.div_ceil(n / BLOCK).clamp(1, 8).min(ibs);
    (1..=ibs.min(8))
        .filter(|d| ibs.is_multiple_of(*d))
        .min_by_key(|d| (d.abs_diff(want), *d < want))
        .unwrap_or(1)
}

fn fixed(entry: &'static str) -> Source {
    Source::Fixed {
        text: ByBackend([SOURCE, "", ""]),
        entry,
    }
}

fn tile_rows(m: usize, mid_rows: usize) -> usize {
    if m <= 4 {
        m
    } else if m <= mid_rows {
        16
    } else {
        32
    }
}

fn entry(t: &TrellisMetal) -> &'static str {
    let m = t.shape[0];
    if m == 1 {
        return match (t.book, t.bits) {
            (0, 2) => "tm_k2_d1",
            (0, 3) => "tm_k3_d1",
            (0, 4) => "tm_k4_d1",
            _ => "tm_gen_d1",
        };
    }
    if m <= 4 {
        return "tm_dec4";
    }
    if m <= t.mid_rows {
        return match (t.book, t.bits) {
            (0, 2) => "tm_k2_pf16",
            (0, 3) => "tm_k3_pf16",
            (0, 4) => "tm_k4_pf16",
            _ => "tm_gen_pf16",
        };
    }
    match (t.book, t.bits) {
        (0, 2) => "tm_k2_pf32",
        (0, 3) => "tm_k3_pf32",
        (0, 4) => "tm_k4_pf32",
        _ => "tm_gen_pf32",
    }
}

fn smem_bytes(m: usize, mid_rows: usize) -> usize {
    if m <= 4 {
        DEC_SMEM
    } else if m <= mid_rows {
        PF16_SMEM
    } else {
        PF32_SMEM
    }
}

fn metal_needs(t: &TrellisMetal) -> Needs {
    let m = t.shape[0];
    Needs {
        lanes: 32,
        group_bytes: smem_bytes(m, t.mid_rows),
        group_threads: THREADS,
        tile: m > 4,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

/// Dense Metal trellis matmul with tile streaming and MMA prefill.
#[derive(Clone, Debug, Kernel)]
#[launch(
    name = self.form.suffixed(if self.turned { "trellis_turned" } else { "trellis_metal" }),
    sizes = [self.shape[0], self.shape[1], self.shape[2]],
    work = self.shape[0] * self.shape[1] * self.shape[2],
    guess = GUESS,
    accepts = accepts,
    emits = f32_only,
    needs = metal_needs(self),
    order = Order::PerConfig,
    source = fixed(entry(self)),
    launches = launches,
    load = load,
    load_key = tessera_lower::load_key((
        self.shape, self.splits, self.mid_rows, self.bits, self.book, self.turned, self.form.name(),
    )),
)]
struct TrellisMetal {
    shape: [usize; 3],
    splits: usize,
    mid_rows: usize,
    bits: u32,
    book: u32,
    turned: bool,
    form: &'static dyn Stored,
}

fn f32_only<K>(_: &K, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn accepts(k: &TrellisMetal, input: usize, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == k.form.name(),
        _ => form.name() == "f32",
    }
}

fn load(t: &TrellisMetal, _: usize) -> Option<Load> {
    let [m, n, k] = t.shape;
    let down = m.div_ceil(tile_rows(m, t.mid_rows));
    let groups = (n / BLOCK) * t.splits * down;
    let mut reads = vec![t.form.bytes(n * k) as f64, (4 * m * k) as f64];
    if t.turned {
        reads.push((4 * k) as f64);
        if t.splits == 1 {
            reads.push((4 * n) as f64);
        }
    }
    Some(Load {
        threads: groups * THREADS,
        group: THREADS,
        trips: (k / BLOCK / t.splits) as f64,
        reads,
        written: (4 * t.splits * m * n) as f64,
        flight: 1024.0,
        group_bytes: smem_bytes(m, t.mid_rows) as f64,
        ..Load::default()
    })
}

fn launches(t: &TrellisMetal, step: &Step) -> Vec<Launch> {
    let [m, n, k] = t.shape;
    let shape = Shape::Grid {
        across: (n / BLOCK) * t.splits,
        down: m.div_ceil(tile_rows(m, t.mid_rows)),
        threads: THREADS,
    };
    let args = launch_args(step, [m, n, k, t.splits], t.bits, t.book, t.turned);
    vec![Launch { args, shape }]
}

/// Deterministic split-K reduction and optional output rotation on Metal.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "trellis_reduce",
    sizes = [self.shape[0], self.shape[1], self.splits],
    work = self.shape[0] * self.shape[1] * self.splits,
    guess = GUESS,
    accepts = reduce_accepts,
    emits = f32_only,
    needs = reduce_needs(),
    order = Order::PerConfig,
    source = fixed("tm_reduce"),
    launches = reduce_launches,
    load = reduce_load,
    load_key = tessera_lower::load_key((self.shape, self.splits, self.turned)),
)]
struct MetalReduce {
    shape: [usize; 2],
    splits: usize,
    turned: bool,
}

fn reduce_accepts(_: &MetalReduce, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
}

fn reduce_needs() -> Needs {
    Needs {
        lanes: 32,
        group_threads: THREADS,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn reduce_load(r: &MetalReduce, _: usize) -> Option<Load> {
    let [m, n] = r.shape;
    let mut reads = vec![(4 * r.splits * m * n) as f64];
    if r.turned {
        reads.push((4 * n) as f64);
    }
    Some(Load {
        threads: (n / BLOCK) * THREADS,
        group: THREADS,
        trips: r.splits as f64,
        reads,
        written: (4 * m * n) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn reduce_launches(r: &MetalReduce, step: &Step) -> Vec<Launch> {
    let [m, n] = r.shape;
    let shape = Shape::Grid {
        across: n / BLOCK,
        down: 1,
        threads: THREADS,
    };
    let svh = if r.turned { step.read(1) } else { step.read(0) };
    let args = vec![
        step.read(0),
        svh,
        step.out(),
        Arg::count(m),
        Arg::count(n),
        Arg::count(r.splits),
        Arg::count(usize::from(r.turned)),
    ];
    vec![Launch { args, shape }]
}

fn drafts(d: &Dense, knobs: &Knobs) -> Vec<Draft> {
    let [m, n, k] = [d.m, d.n, d.k];
    let (splits, turned) = (split_k(knobs, m, n, k), d.turned());
    let mid_rows = knobs.get("trellis_metal", "mid_rows").unwrap_or(16);
    let kernel = TrellisMetal {
        shape: [m, n, k],
        splits,
        mid_rows,
        bits: d.bits,
        book: d.book,
        turned,
        form: d.form,
    };
    let reduce = MetalReduce {
        shape: [m, n],
        splits,
        turned,
    };
    d.drafts(kernel, splits, reduce)
}

fn eval_metal(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    Some(Dense::find(tree, at, BLOCK)?.map(|d| drafts(&d, &tree.knobs)))
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    eval_metal(tree, at)?.ok()
}

fn why(tree: &Tree, at: T) -> Option<String> {
    eval_metal(tree, at)?.err()
}

tessera_lower::rule!("trellis_metal", propose, why = why, lacks = LACKS);

/// Picked expert matmuls on Metal. Zero `bits` means mixed.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.site.form.suffixed(if self.site.turned { "picked_metal_t" } else { "picked_metal" }),
    sizes = [self.site.rows, self.site.n, self.site.k],
    work = self.site.rows * self.site.n * self.site.k,
    guess = GUESS,
    accepts = self.site.accepts(input, form),
    emits = f32_only,
    needs = picked_needs(),
    order = Order::PerConfig,
    source = fixed(picked_entry(self)),
    launches = picked_launches,
    load = Some(self.site.load(BLOCK, THREADS)),
)]
struct PickedTrellisMetal {
    site: Coded,
    bits: u32,
    book: u32,
}

fn picked_entry(p: &PickedTrellisMetal) -> &'static str {
    if p.bits == 0 && p.book == 2 && p.site.turned {
        "tm_pickmix_pmcgx_t"
    } else {
        "tm_picked_gen"
    }
}

fn picked_needs() -> Needs {
    Needs {
        lanes: 32,
        group_bytes: DEC_SMEM,
        group_threads: THREADS,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn picked_launches(p: &PickedTrellisMetal, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: p.site.n / BLOCK,
        down: p.site.rows,
        threads: THREADS,
    };
    let args = p.site.args(step, p.bits, p.book);
    vec![Launch { args, shape }]
}

/// Groups routed row indices by expert id on Metal.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "picked_metal_group",
    sizes = [self.rows, self.count],
    work = self.rows + self.count,
    guess = GUESS,
    accepts = pick_group_accepts,
    emits = f32_only,
    needs = pick_group_needs(),
    order = Order::PerConfig,
    source = fixed("tm_pick_group"),
    launches = pick_group_launches,
    load = pick_group_load,
)]
struct MetalPickGroup {
    rows: usize,
    count: usize,
    first: usize,
}

fn pick_group_accepts(_: &MetalPickGroup, _: usize, form: &dyn Stored) -> bool {
    form.name() == "u32"
}

fn pick_group_needs() -> Needs {
    Needs {
        lanes: 32,
        group_bytes: GROUP_SMEM,
        group_threads: THREADS,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn pick_group_load(g: &MetalPickGroup, _: usize) -> Option<Load> {
    let meta_bytes = (4 * (2 * (g.count + 1) + g.rows)) as f64;
    Some(Load {
        threads: THREADS,
        group: THREADS,
        trips: g.rows.div_ceil(THREADS) as f64,
        reads: vec![(4 * g.rows) as f64],
        written: meta_bytes,
        flight: 64.0,
        group_bytes: GROUP_SMEM as f64,
        ..Load::default()
    })
}

fn pick_group_launches(g: &MetalPickGroup, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: 1,
        down: 1,
        threads: THREADS,
    };
    let args = vec![
        step.read(0),
        step.out(),
        Arg::count(g.rows),
        Arg::count(g.count),
        Arg::count(g.first),
    ];
    vec![Launch { args, shape }]
}

/// Grouped-expert MMA prefill for picked trellis matmuls on Metal.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = self.site.form.suffixed(if self.site.turned { "picked_metal_pf_t" } else { "picked_metal_pf" }),
    sizes = [self.site.rows, self.site.n, self.site.k],
    work = self.site.rows * self.site.n * self.site.k,
    guess = GUESS,
    accepts = picked_pf_accepts,
    emits = f32_only,
    needs = picked_pf_needs(self),
    order = Order::PerConfig,
    source = fixed(picked_pf_entry(self)),
    launches = picked_pf_launches,
    load = picked_pf_load,
)]
struct PickedTrellisPfMetal {
    site: Coded,
    bits: u32,
    book: u32,
    wide: bool,
}

fn picked_pf_accepts(p: &PickedTrellisPfMetal, input: usize, form: &dyn Stored) -> bool {
    match input {
        2 => form.name() == "f32",
        _ => p.site.accepts(input, form),
    }
}

fn picked_pf_entry(p: &PickedTrellisPfMetal) -> &'static str {
    let fast = p.bits == 0 && p.book == 2 && p.site.turned;
    match (fast, p.wide) {
        (true, false) => "tm_pickmix_pf16_pmcgx_t",
        (true, true) => "tm_pickmix_pf32_pmcgx_t",
        (false, false) => "tm_pick_pf16_gen",
        (false, true) => "tm_pick_pf32_gen",
    }
}

fn picked_pf_smem(wide: bool) -> usize {
    if wide { PF32_PICK_SMEM } else { PF16_PICK_SMEM }
}

fn picked_pf_needs(p: &PickedTrellisPfMetal) -> Needs {
    Needs {
        lanes: 32,
        group_bytes: picked_pf_smem(p.wide),
        group_threads: THREADS,
        tile: true,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn picked_pf_load(p: &PickedTrellisPfMetal, _: usize) -> Option<Load> {
    let s = p.site;
    let bm = if p.wide { 32 } else { 16 };
    let passes = s.rows.div_ceil(s.count.max(1)).div_ceil(bm);
    let groups = (s.n / BLOCK) * (s.count + 1);
    let out = s.rows * s.n;
    let mut reads = vec![
        s.form.bytes(s.count * passes * s.n * s.k) as f64,
        (4 * (s.rows / s.repeat) * s.k) as f64,
        (4 * (2 * (s.count + 1) + s.rows)) as f64,
    ];
    if s.turned {
        reads.extend([(4 * s.count * s.k) as f64, (4 * s.count * s.n) as f64]);
    }
    match s.post {
        1 => reads.push((4 * s.n) as f64),
        2 => reads.push((4 * out) as f64),
        _ => {}
    }
    Some(Load {
        threads: groups * THREADS,
        group: THREADS,
        trips: (passes * (s.k / BLOCK)) as f64,
        reads,
        written: (4 * out) as f64,
        flight: 1024.0,
        group_bytes: picked_pf_smem(p.wide) as f64,
        ..Load::default()
    })
}

fn picked_pf_launches(p: &PickedTrellisPfMetal, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: p.site.n / BLOCK,
        down: p.site.count + 1,
        threads: THREADS,
    };
    let args = p.site.args(step, p.bits, p.book);
    vec![Launch { args, shape }]
}

/// Matches rotated or plain picked trellis matmuls of either kind.
fn propose_picked(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (site, reads) = Coded::find(tree, at, BLOCK)?;
    let name = site.form.name();
    let mixed = parse_mix_form(name).map(|book| (0, book));
    let (bits, book) = parse_form(name).or(mixed)?;
    if site.count <= MAX_GROUP_EXPERTS && site.rows >= 64 && site.rows >= site.count * 2 {
        let meta_len = 2 * (site.count + 1) + site.rows;
        let group = MetalPickGroup {
            rows: site.rows,
            count: site.count,
            first: site.first,
        };
        let pf = PickedTrellisPfMetal {
            site,
            bits,
            book,
            wide: site.rows > site.count * 16,
        };
        let mut pf_reads: Vec<Read> = reads.iter().copied().map(Read::Node).collect();
        pf_reads[2] = Read::Draft(0);
        return Some(vec![
            Draft::new(group, [Read::Node(reads[2])], meta_len),
            Draft::new(pf, pf_reads, site.rows * site.n),
        ]);
    }
    let kernel = PickedTrellisMetal { site, bits, book };
    Some(vec![Draft::new(kernel, reads, site.rows * site.n)])
}

fn picked_why(_: &Tree, _: T) -> Option<String> {
    None
}

tessera_lower::rule!(
    "picked_trellis_metal",
    propose_picked,
    why = picked_why,
    lacks = LACKS
);
