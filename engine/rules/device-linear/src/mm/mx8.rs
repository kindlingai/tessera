//! Blackwell native block-scaled MXFP8 prep and matmul kernels.

use super::*;

const MX8_SMEM: usize = 2 * (128 * 36 + 128 * 32 + 128) * 4;

fn mxfp8_pre_len(m: usize, k: usize) -> usize {
    (m * k / 128) * 33
}

fn mx8_defs(
    blk: Option<(&'static str, u32)>,
    x_bf16: bool,
    y_bf16: bool,
    o_bf16: bool,
) -> Vec<(&'static str, usize)> {
    let mut defs = vec![("PRE_ONLY_MX8", 1)];
    if let Some((name, val)) = blk {
        defs.push((name, val as usize));
    }
    io_defs(defs, x_bf16, y_bf16, o_bf16)
}

/// Quantizes activations once into packed E4M3 and UE8M0 scales.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "fast_mm_mxfp8_prep",
    sizes = [self.m, self.k],
    work = self.m * self.k,
    accepts = mx8_prep_accepts,
    emits = f32_only,
    needs = mx8_prep_needs(),
    order = Order::PerConfig,
    source = prefill_source_with(
        "tessera_mm_mxfp8_prep",
        mx8_defs(
            self.min_blocks.map(|v| ("MM_PREP_BLK", v)),
            self.x_bf16,
            false,
            false,
        ),
    ),
    launches = mx8_prep_launches,
    shape = Shape::grid(((self.k / 32).div_ceil(32), self.m), 256),
    load = mx8_prep_load,
    load_key = tessera_lower::load_key((self.m, self.k, self.min_blocks, self.x_bf16)),
    shared,
    unproven,
)]
struct Mxfp8Prep {
    m: usize,
    k: usize,
    min_blocks: Option<u32>,
    x_bf16: bool,
}

fn mx8_prep_accepts(p: &Mxfp8Prep, _: usize, form: &dyn Stored) -> bool {
    x_accepts(p.x_bf16, form)
}

fn mx8_prep_needs() -> Needs {
    Needs {
        lanes: 32,
        group_bytes: 0,
        group_threads: 256,
        compiler: true,
        lacks: CUDA_LACKS,
        mxfp8_tile: true,
        ..Needs::default()
    }
}

fn mx8_prep_load(p: &Mxfp8Prep, _: usize) -> Option<Load> {
    let (across, down) = ((p.k / 32).div_ceil(32), p.m);
    let x_elem = if p.x_bf16 { 2 } else { 4 };
    let mut load = Load {
        threads: across * down * 256,
        group: 256,
        trips: 1.0,
        reads: vec![(x_elem * p.m * p.k) as f64],
        written: (4 * mxfp8_pre_len(p.m, p.k)) as f64,
        flight: 512.0,
        ..Load::default()
    };
    load.trip.fresh = 1.0;
    Some(load)
}

fn mx8_prep_launches(p: &Mxfp8Prep, step: &Step) -> Vec<Launch> {
    let shape = p.shape().expect("a grid");
    let args = vec![step.read(0), step.out(), Arg::count(p.m), Arg::count(p.k)];
    vec![Launch { args, shape }]
}

/// Blackwell native 128x128 block-scaled MXFP8 Tensor-Core matmul.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = if self.ks > 1 { "fast_mm_mxfp8_mx8_sk" } else { "fast_mm_mxfp8_mx8" },
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    accepts = mx8_mm_accepts,
    emits = mx8_mm_emits,
    needs = mx8_mm_needs(),
    order = Order::PerConfig,
    source = prefill_source_with(
        if self.ks > 1 {
            "tessera_mm_mxfp8_mx8_sk"
        } else {
            "tessera_mm_mxfp8_mx8"
        },
        mx8_defs(
            self.min_blocks.map(|v| {
                let def = if self.ks > 1 {
                    "MM_MX8_SK_BLK"
                } else {
                    "MM_MX8_BLK"
                };
                (def, v)
            }),
            false,
            self.y_bf16,
            self.o_bf16,
        ),
    ),
    launches = mx8_mm_launches,
    shape = Shape::grid((self.n.div_ceil(128) * self.ks, self.m.div_ceil(128)), 256),
    load = mx8_mm_load,
    load_key = tessera_lower::load_key((
        self.m, self.n, self.k, self.ks, self.mode, self.min_blocks, self.form.name(),
        self.y_bf16, self.o_bf16,
    )),
    unproven,
)]
struct Mxfp8Mx8 {
    m: usize,
    n: usize,
    k: usize,
    ks: usize,
    mode: u32,
    min_blocks: Option<u32>,
    y_bf16: bool,
    o_bf16: bool,
    form: &'static dyn Stored,
}

fn mx8_mm_accepts(k: &Mxfp8Mx8, input: usize, form: &dyn Stored) -> bool {
    weight_io_accepts(input, false, k.o_bf16, k.form, form)
}

fn mx8_mm_emits(k: &Mxfp8Mx8, form: &dyn Stored) -> bool {
    x_accepts(k.y_bf16, form)
}

fn mx8_mm_needs() -> Needs {
    Needs {
        lanes: 32,
        group_bytes: 0,
        dynamic_shared: MX8_SMEM,
        group_threads: 256,
        compiler: true,
        lacks: CUDA_LACKS,
        mxfp8_tile: true,
        ..Needs::default()
    }
}

fn mx8_mm_load(k: &Mxfp8Mx8, _: usize) -> Option<Load> {
    let groups = k.n.div_ceil(128) * k.ks * k.m.div_ceil(128);
    let y_elem = if k.y_bf16 { 2 } else { 4 };
    let mut reads = vec![
        k.form.bytes(k.n * k.k) as f64,
        (4 * mxfp8_pre_len(k.m, k.k)) as f64,
    ];
    push_epi_reads(&mut reads, k.mode, k.o_bf16, k.m, k.n);
    let mut load = Load {
        threads: groups * 256,
        group: 256,
        trips: k.k as f64 / (128.0 * k.ks as f64),
        reads,
        written: (y_elem * k.ks * k.m * k.n) as f64,
        flight: 1024.0,
        ..Load::default()
    };
    load.trip.fresh = 1.0;
    load.trip.loads = 8.5;
    load.trip.width = 16.0;
    load.trip.fp8_flops = (2 * 128 * 128 * 128 / 256) as f64;
    Some(load)
}

fn mx8_mm_launches(k: &Mxfp8Mx8, step: &Step) -> Vec<Launch> {
    let shape = k.shape().expect("a grid");
    let other = if has_other(k.mode) {
        step.read(2)
    } else {
        step.read(0)
    };
    let mut args = vec![
        step.read(0),
        step.read(1),
        other,
        step.out(),
        Arg::count(k.m),
        Arg::count(k.n),
        Arg::count(k.k),
        Arg::value(&k.mode),
    ];
    if k.ks > 1 {
        args.push(Arg::count(k.ks));
    }
    vec![Launch { args, shape }]
}

fn draft_mx8(site: MmSite, epi: Epi) -> Vec<Draft> {
    let prep = Mxfp8Prep {
        m: site.m,
        k: site.k,
        min_blocks: site.prep_min_blocks,
        x_bf16: site.x_bf16,
    };
    let pre_len = mxfp8_pre_len(site.m, site.k);
    if site.ks > 1 {
        let mm = Mxfp8Mx8 {
            m: site.m,
            n: site.n,
            k: site.k,
            ks: site.ks,
            mode: 0,
            min_blocks: site.min_blocks,
            y_bf16: false,
            o_bf16: false,
            form: site.form,
        };
        return vec![
            Draft::new(prep, [Read::Node(site.x)], pre_len),
            Draft::new(
                mm,
                [Read::Draft(0), Read::Node(site.w)],
                site.ks * site.m * site.n,
            ),
            site.reduce_draft(1, epi),
        ];
    }
    let mm = Mxfp8Mx8 {
        m: site.m,
        n: site.n,
        k: site.k,
        ks: 1,
        mode: epi.mode,
        min_blocks: site.min_blocks,
        y_bf16: epi.y_bf16,
        o_bf16: epi.o_bf16,
        form: site.form,
    };
    let reads = [Read::Draft(0), Read::Node(site.w)]
        .into_iter()
        .chain(epi.other.map(Read::Node));
    vec![
        Draft::new(prep, [Read::Node(site.x)], pre_len),
        Draft::new(mm, reads, site.m * site.n),
    ]
}

fn mx8_site(tree: &Tree, mat: T) -> Option<MmSite> {
    let mut site = mm_site(tree, mat)?;
    if site.form.name() != "mxfp8" || site.m < knob(tree, "fast_mm_mxfp8_mx8", "min_rows", 64) {
        return None;
    }
    let chunks = site.k / 512;
    let set = set_ks(tree, "fast_mm_mxfp8_mx8_sk", chunks)
        .or_else(|| set_ks(tree, "fast_mm_mxfp8_sk", chunks));
    let min_g = knob(tree, "fast_mm_mxfp8_mx8", "min_groups", 40);
    let target = knob(tree, "fast_mm_mxfp8_mx8", "target_groups", 64);
    site.ks = set.unwrap_or_else(|| mxfp8_tile_ks(site.m, site.n, site.k, 128, min_g, 24, target));
    let mx_rule = if site.ks > 1 {
        "fast_mm_mxfp8_mx8_sk"
    } else {
        "fast_mm_mxfp8_mx8"
    };
    site.min_blocks = tree.knobs.get(mx_rule, "min_blocks").map(|v| v as u32);
    site.prep_min_blocks = tree
        .knobs
        .get("fast_mm_mxfp8_mx8", "prep_min_blocks")
        .map(|v| v as u32);
    site.reduce_min_blocks = tree
        .knobs
        .get("fast_mm_mxfp8_mx8_sk", "reduce_min_blocks")
        .map(|v| v as u32);
    Some(site)
}

pub(crate) fn propose_mm_mx8(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (site, epi) = match_mm_site(tree, at, mx8_site)?;
    Some(draft_mx8(site, epi))
}
