//! Native Metal simdgroup_matrix prefill matmul across dense and block forms.

use tessera_graph::Order;
use tessera_lower::Backend;
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Needs, Read, Shape, Step, T, Tree, sites};

const THREADS_32: usize = 128;
const THREADS_64: usize = 256;
const BN: usize = 64;
const BK: usize = 32;
const STRIDE: usize = 36;
const LACKS: &[Backend] = &[Backend::Cuda, Backend::Rocm];
const SOURCE: &str = include_str!("mm_metal.metal");
/// Low default guess so Metal prefers native prefill matmul.
const GUESS: f64 = 1.0e-12;

struct EntrySet {
    form: &'static str,
    block: bool,
    min_m: usize,
    e32: &'static str,
    e64: &'static str,
}

const ENTRIES: &[EntrySet] = &[
    EntrySet {
        form: "f16",
        block: false,
        min_m: 64,
        e32: "mm32_f16",
        e64: "mm64_f16",
    },
    EntrySet {
        form: "fp8",
        block: false,
        min_m: 32,
        e32: "mm32_fp8",
        e64: "mm64_fp8",
    },
    EntrySet {
        form: "mxfp8",
        block: true,
        min_m: 9,
        e32: "mm32_mxfp8",
        e64: "mm64_mxfp8",
    },
    EntrySet {
        form: "q8_0",
        block: true,
        min_m: 9,
        e32: "mm32_q8_0",
        e64: "mm64_q8_0",
    },
    EntrySet {
        form: "int4_g32",
        block: true,
        min_m: 9,
        e32: "mm32_int4_g32",
        e64: "mm64_int4_g32",
    },
    EntrySet {
        form: "q4_0",
        block: true,
        min_m: 9,
        e32: "mm32_q4_0",
        e64: "mm64_q4_0",
    },
];

#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "mm_metal",
    source = tessera_lower::Source::metal(SOURCE, self.entry),
    order = Order::PerConfig,
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    guess = GUESS,
    emits = crate::f32_only,
    accepts = mm_accepts,
    needs = mm_needs(self),
    launches = mm_launches,
    load = mm_load,
)]
struct MetalMm {
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    form: &'static dyn Stored,
    entry: &'static str,
}

fn mm_accepts(k: &MetalMm, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == "f32",
        1 => form.name() == k.form.name(),
        _ => false,
    }
}

fn mm_threads(bm: usize) -> usize {
    if bm == 32 { THREADS_32 } else { THREADS_64 }
}

fn mm_needs(k: &MetalMm) -> Needs {
    Needs {
        lanes: 32,
        group_bytes: (k.bm + BN) * STRIDE * 4,
        group_threads: mm_threads(k.bm),
        tile: true,
        compiler: true,
        lacks: LACKS,
        ..Needs::default()
    }
}

fn mm_load(k: &MetalMm, _: usize) -> Option<Load> {
    let gm = k.m.div_ceil(k.bm);
    let gn = k.n.div_ceil(BN);
    let bands = gm.div_ceil(8);
    let threads = mm_threads(k.bm);
    Some(Load {
        threads: gm * gn * threads,
        group: threads,
        trips: k.k.div_ceil(BK) as f64,
        reads: vec![
            (4 * k.m * k.k) as f64,
            (bands * k.form.bytes(k.n * k.k)) as f64,
        ],
        written: (4 * k.m * k.n) as f64,
        flight: 64.0,
        group_bytes: ((k.bm + BN) * STRIDE * 4) as f64,
        ..Load::default()
    })
}

fn mm_launches(k: &MetalMm, step: &Step) -> Vec<Launch> {
    let across = k.m.div_ceil(k.bm) * k.n.div_ceil(BN);
    vec![Launch {
        args: vec![
            step.read(0),
            step.read(1),
            step.out(),
            Arg::count(k.m),
            Arg::count(k.n),
            Arg::count(k.k),
        ],
        shape: Shape::Grid {
            across,
            down: 1,
            threads: mm_threads(k.bm),
        },
    }]
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Matmul = tree.site(at)?;
    if site.picked.is_some() || site.m < 9 || site.k == 0 || site.n == 0 {
        return None;
    }
    if tree.node(site.x).form.name() != "f32" || tree.node(at).form.name() != "f32" {
        return None;
    }
    let set = ENTRIES.iter().find(|e| e.form == site.form.name())?;
    if site.m < set.min_m || (set.block && !site.k.is_multiple_of(BK)) {
        return None;
    }
    let (bm, entry) = if site.m <= 32 {
        (32, set.e32)
    } else {
        (64, set.e64)
    };
    let kernel = MetalMm {
        m: site.m,
        n: site.n,
        k: site.k,
        bm,
        form: site.form,
        entry,
    };
    Some(vec![Draft::new(
        kernel,
        [Read::Node(site.x), Read::Node(site.w)],
        site.m * site.n,
    )])
}

fn why(_: &Tree, _: T) -> Option<String> {
    None
}

tessera_lower::rule!("mm_metal", propose, why = why, lacks = LACKS);
