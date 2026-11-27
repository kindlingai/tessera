//! Blackwell native block-scaled MXFP8 prep and matmul kernels.

use super::*;

const MX8_SMEM: usize = 2 * (128 * 36 + 128 * 32 + 128) * 4;

pub(super) fn mxfp8_pre_len(m: usize, k: usize) -> usize {
    (m * k / 128) * 33
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
        self.min_blocks.map(|v| ("MM_PREP_BLK", v as usize)),
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
}

fn mx8_prep_accepts(_: &Mxfp8Prep, _: usize, form: &dyn Stored) -> bool {
    form.name() == "f32"
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
    let mut load = Load {
        threads: across * down * 256,
        group: 256,
        trips: 1.0,
        reads: vec![(4 * p.m * p.k) as f64],
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
    emits = f32_only,
    needs = mx8_mm_needs(),
    order = Order::PerConfig,
    source = prefill_source_with(
        if self.ks > 1 {
            "tessera_mm_mxfp8_mx8_sk"
        } else {
            "tessera_mm_mxfp8_mx8"
        },
        self.min_blocks.map(|v| {
            let def = if self.ks > 1 {
                "MM_MX8_SK_BLK"
            } else {
                "MM_MX8_BLK"
            };
            (def, v as usize)
        }),
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
    pub(super) form: &'static dyn Stored,
}

fn mx8_mm_accepts(k: &Mxfp8Mx8, input: usize, form: &dyn Stored) -> bool {
    weight_accepts(input, k.form, form)
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
    let mut reads = vec![
        k.form.bytes(k.n * k.k) as f64,
        (4 * mxfp8_pre_len(k.m, k.k)) as f64,
    ];
    if matches!(k.mode, 1 | 3 | 4) {
        reads.push((4 * k.m * k.n) as f64);
    } else if k.mode == 2 {
        reads.push((4 * k.n) as f64);
    }
    let mut load = Load {
        threads: groups * 256,
        group: 256,
        trips: k.k as f64 / (128.0 * k.ks as f64),
        reads,
        written: (4 * k.ks * k.m * k.n) as f64,
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
    let other = if matches!(k.mode, 1..=4) {
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
