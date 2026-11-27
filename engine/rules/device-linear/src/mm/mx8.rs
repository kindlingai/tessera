//! Blackwell native block-scaled MXFP8 prep and matmul kernels.

use super::*;

const MX8_SMEM: usize = 2 * (128 * 36 + 128 * 32 + 128) * 4;
const MX8_SMEM_IX: usize = 2 * (128 * 36 + 128 * 36) * 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MxIo {
    F32,
    Bf16,
    Mxfp8,
}

impl MxIo {
    pub(super) fn from_name(name: &str) -> Option<Self> {
        match name {
            "f32" => Some(Self::F32),
            "bf16" => Some(Self::Bf16),
            "mxfp8" => Some(Self::Mxfp8),
            _ => None,
        }
    }

    pub(super) fn matches(self, form: &dyn Stored) -> bool {
        match self {
            Self::F32 => form.name() == "f32",
            Self::Bf16 => form.name() == "bf16",
            Self::Mxfp8 => form.name() == "mxfp8",
        }
    }

    pub(super) fn elem_bytes(self) -> f64 {
        match self {
            Self::F32 => 4.0,
            Self::Bf16 => 2.0,
            Self::Mxfp8 => 33.0 / 32.0,
        }
    }
}

pub(super) fn mxfp8_pre_len(m: usize, k: usize) -> usize {
    (m * k / 128) * 33
}

fn mx8_defs<'a>(
    base: impl IntoIterator<Item = (&'a str, usize)>,
    x_mx8: bool,
    y_io: MxIo,
    o_io: MxIo,
) -> Vec<(&'a str, usize)> {
    let mut defs = io_defs(base, false, y_io == MxIo::Bf16, o_io == MxIo::Bf16);
    if x_mx8 {
        defs.push(("X_MXFP8", 1));
    }
    if y_io == MxIo::Mxfp8 {
        defs.push(("Y_MXFP8", 1));
    }
    if o_io == MxIo::Mxfp8 {
        defs.push(("O_MXFP8", 1));
    }
    defs
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
        x_defs(self.min_blocks.map(|v| ("MM_PREP_BLK", v as usize)), self.x_bf16),
    ),
    launches = mx8_prep_launches,
    shape = Shape::grid(((self.k / 32).div_ceil(32), self.m), 256),
    load = mx8_prep_load,
    shared,
    unproven,
)]
pub(super) struct Mxfp8Prep {
    pub(super) m: usize,
    pub(super) k: usize,
    pub(super) min_blocks: Option<u32>,
    pub(super) x_bf16: bool,
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
    needs = mx8_mm_needs(self),
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
                (def, v as usize)
            }),
            self.x_mx8,
            self.y_io,
            self.o_io,
        ),
    ),
    launches = mx8_mm_launches,
    shape = Shape::grid((self.n.div_ceil(128) * self.ks, self.m.div_ceil(128)), 256),
    load = mx8_mm_load,
    unproven,
)]
pub(super) struct Mxfp8Mx8 {
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) ks: usize,
    pub(super) mode: u32,
    pub(super) min_blocks: Option<u32>,
    pub(super) x_mx8: bool,
    pub(super) y_io: MxIo,
    pub(super) o_io: MxIo,
    pub(super) form: &'static dyn Stored,
}

fn mx8_mm_accepts(k: &Mxfp8Mx8, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => {
            if k.x_mx8 {
                form.name() == "mxfp8"
            } else {
                form.name() == "f32"
            }
        }
        1 => form.name() == k.form.name(),
        2 => k.o_io.matches(form),
        _ => form.name() == "f32",
    }
}

fn mx8_mm_emits(k: &Mxfp8Mx8, form: &dyn Stored) -> bool {
    k.y_io.matches(form)
}

fn mx8_mm_needs(k: &Mxfp8Mx8) -> Needs {
    Needs {
        lanes: 32,
        group_bytes: 0,
        dynamic_shared: if k.x_mx8 { MX8_SMEM_IX } else { MX8_SMEM },
        group_threads: 256,
        compiler: true,
        lacks: CUDA_LACKS,
        mxfp8_tile: true,
        ..Needs::default()
    }
}

fn mx8_mm_load(k: &Mxfp8Mx8, _: usize) -> Option<Load> {
    let groups = k.n.div_ceil(128) * k.ks * k.m.div_ceil(128);
    let o_elem = k.o_io.elem_bytes();
    let y_elem = k.y_io.elem_bytes();
    let mut reads = vec![
        k.form.bytes(k.n * k.k) as f64,
        (4 * mxfp8_pre_len(k.m, k.k)) as f64,
    ];
    if matches!(k.mode, 1 | 3 | 4) {
        reads.push(o_elem * (k.m * k.n) as f64);
    } else if k.mode == 2 {
        reads.push(o_elem * k.n as f64);
    }
    let mut load = Load {
        threads: groups * 256,
        group: 256,
        trips: k.k as f64 / (128.0 * k.ks as f64),
        reads,
        written: y_elem * (k.ks * k.m * k.n) as f64,
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
    let ks = (k.ks > 1).then_some(k.ks);
    matmul_launches(step, shape, [k.m, k.n, k.k], k.mode, ks)
}

/// Split-K reduction with optional MXFP8 epilogue and output quantization.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "fast_mm_reduce_mx8",
    sizes = [self.m, self.n, self.ks],
    work = self.m * self.n * self.ks,
    accepts = mx8_red_accepts,
    emits = mx8_red_emits,
    needs = mx8_prep_needs(),
    order = Order::PerConfig,
    source = prefill_source_with(
        "tessera_mm_reduce_mx8",
        mx8_defs(
            self.min_blocks.map(|v| ("MM_PREP_BLK", v as usize)),
            false,
            self.y_io,
            self.o_io,
        ),
    ),
    launches = mx8_red_launches,
    shape = Shape::grid(((self.n / 4).div_ceil(256), self.m), 256),
    load = mx8_red_load,
    unproven,
)]
pub(super) struct Mxfp8Reduce {
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) ks: usize,
    pub(super) mode: u32,
    pub(super) min_blocks: Option<u32>,
    pub(super) y_io: MxIo,
    pub(super) o_io: MxIo,
}

fn mx8_red_accepts(r: &Mxfp8Reduce, input: usize, form: &dyn Stored) -> bool {
    match input {
        1 => r.o_io.matches(form),
        _ => form.name() == "f32",
    }
}

fn mx8_red_emits(r: &Mxfp8Reduce, form: &dyn Stored) -> bool {
    r.y_io.matches(form)
}

fn mx8_red_load(r: &Mxfp8Reduce, _: usize) -> Option<Load> {
    let o_elem = r.o_io.elem_bytes();
    let y_elem = r.y_io.elem_bytes();
    let mut reads = vec![(4 * r.ks * r.m * r.n) as f64];
    if matches!(r.mode, 1 | 3 | 4) {
        reads.push(o_elem * (r.m * r.n) as f64);
    } else if r.mode == 2 {
        reads.push(o_elem * r.n as f64);
    }
    Some(Load {
        threads: (r.n / 4).div_ceil(256) * r.m * 256,
        group: 256,
        trips: r.ks as f64,
        reads,
        written: y_elem * (r.m * r.n) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn mx8_red_launches(r: &Mxfp8Reduce, step: &Step) -> Vec<Launch> {
    let shape = r.shape().expect("a grid");
    reduce_launches(step, shape, [r.m, r.n, r.ks], r.mode)
}

pub(super) fn mx8_site(tree: &Tree, mat: T) -> Option<MmSite> {
    let mut site = check_mm(tree, mat, true)?.ok()?;
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

impl MmSite {
    pub(super) fn draft_mx8(self, mode: u32, other: Option<T>) -> Vec<Draft> {
        let x_mx8 = self.x_io == MxIo::Mxfp8;
        let mut drafts = Vec::with_capacity(3);
        let x_read = if x_mx8 {
            Read::Node(self.x)
        } else {
            let prep = Mxfp8Prep {
                m: self.m,
                k: self.k,
                min_blocks: self.prep_min_blocks,
                x_bf16: self.x_io == MxIo::Bf16,
            };
            drafts.push(Draft::new(
                prep,
                [Read::Node(self.x)],
                mxfp8_pre_len(self.m, self.k),
            ));
            Read::Draft(0)
        };
        let mm_idx = drafts.len();
        if self.ks > 1 {
            let mm = Mxfp8Mx8 {
                m: self.m,
                n: self.n,
                k: self.k,
                ks: self.ks,
                mode: 0,
                min_blocks: self.min_blocks,
                x_mx8,
                y_io: MxIo::F32,
                o_io: MxIo::F32,
                form: self.form,
            };
            drafts.push(Draft::new(
                mm,
                [x_read, Read::Node(self.w)],
                self.ks * self.m * self.n,
            ));
            drafts.push(self.reduce_draft(mm_idx, mode, other));
            return drafts;
        }
        let mm = Mxfp8Mx8 {
            m: self.m,
            n: self.n,
            k: self.k,
            ks: 1,
            mode,
            min_blocks: self.min_blocks,
            x_mx8,
            y_io: self.y_io,
            o_io: self.o_io,
            form: self.form,
        };
        let mut reads = vec![x_read, Read::Node(self.w)];
        if let Some(extra) = other {
            reads.push(Read::Node(extra));
        }
        drafts.push(Draft::new(mm, reads, self.m * self.n));
        drafts
    }
}
