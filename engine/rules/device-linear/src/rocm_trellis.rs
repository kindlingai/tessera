//! RDNA 3/3.5 ROCm trellis WMMA rule dispatch.

use super::epi::{epi_accepts, has_bias2, has_other, match_epi_site, pick_divisor, push_epi_reads};
use super::{
    BLOCK, THREADS, TrellisMma, TrellisPreMma, TrellisPrep, TrellisReduce, f32_only,
    reduce_load_for, rocm_needs, rocm_rule, rocm_source,
};
use std::sync::OnceLock;
use tessera_device::{Backend, flatten_for};
use tessera_graph::Order;
use tessera_linear::Dense;
use tessera_lower::form::Load as Stored;
use tessera_lower::{Arg, Draft, Kernel, Launch, Load, Needs, Read, Shape, Step, T, Tree};

const ROCM_TRELLIS: &str = include_str!("rocm_trellis.cu");
const SMEM_T1: usize = 24 * 1024;
const SMEM_T2: usize = 32 * 1024;
const SMEM_PRE: usize = 21_504;

pub(crate) fn rocm_trellis_source() -> &'static str {
    static ROCM: OnceLock<&'static str> = OnceLock::new();
    ROCM.get_or_init(|| flatten_for(Backend::Rocm, "rocm_trellis.kernel", ROCM_TRELLIS))
}

fn dtr_ks_for(m: usize, n: usize, k: usize, narrow_m: usize, targets: [usize; 2]) -> usize {
    let ibs = k / BLOCK;
    if m > 16 || ibs <= 1 {
        return 1;
    }
    let target = if m <= narrow_m {
        targets[0]
    } else {
        targets[1]
    };
    let cap = ibs.min(16);
    let want = target.div_ceil(n / BLOCK).clamp(1, cap);
    pick_divisor(ibs, want, cap)
}

pub(crate) fn small_tile(m: usize) -> usize {
    if m <= 16 { 16 } else { 32 }
}

pub(crate) fn small_needs(tile: usize) -> Needs {
    rocm_needs(if tile == 16 { SMEM_T1 } else { SMEM_T2 }, THREADS, true)
}

pub(crate) fn pre_tile(m: usize, n: usize) -> usize {
    let tail_ok = m >= 512 || m.is_multiple_of(256) || (m % 256) >= 192;
    if m >= 256 && tail_ok && m.div_ceil(256) * (n / BLOCK) >= 40 {
        256
    } else if m > 64 {
        128
    } else {
        64
    }
}

pub(crate) fn pre_needs() -> Needs {
    rocm_needs(SMEM_PRE, THREADS, true)
}

fn rocm_pre_ks(m: usize, n: usize, k: usize) -> usize {
    let ibs = k / BLOCK;
    let tile = pre_tile(m, n);
    let base = (n / BLOCK) * m.div_ceil(tile);
    if tile == 256 || ibs < 8 || base >= 64 {
        return 1;
    }
    let cap = if base <= 8 { 8 } else { 4 }.min(ibs / 2);
    let want = 64usize.div_ceil(base).clamp(1, cap);
    pick_divisor(ibs, want, cap)
}

pub(crate) fn small_entry(t: &TrellisMma) -> &'static str {
    match (t.tile, t.bits, t.book, t.turned, t.ks > 1) {
        (16, 4, 0, true, true) => "rocm_dtr_k4_t1s",
        (16, 4, 0, true, false) => "rocm_dtr_k4_t1",
        (16, 4, 0, false, true) => "rocm_dtr_k4_p1s",
        (16, 4, 0, false, false) => "rocm_dtr_k4_p1",
        (32, 4, 0, true, true) => "rocm_dtr_k4_t2s",
        (32, 4, 0, true, false) => "rocm_dtr_k4_t2",
        (16, _, _, _, _) => "rocm_dtr_gen1",
        (_, _, _, _, _) => "rocm_dtr_gen2",
    }
}

pub(crate) fn pre_entry(t: &TrellisPreMma) -> &'static str {
    match (t.tile, t.bits, t.book, t.turned, t.ks > 1) {
        (256, 4, 0, true, false) => "rocm_dtr_pre_k4_t16",
        (128, 4, 0, true, true) => "rocm_dtr_pre_k4_t8s",
        (128, 4, 0, true, false) => "rocm_dtr_pre_k4_t8",
        (64, 4, 0, true, true) => "rocm_dtr_pre_k4_t4s",
        (64, 4, 0, true, false) => "rocm_dtr_pre_k4_t4",
        (256, _, _, _, _) => "rocm_dtr_pre_gen16",
        (128, _, _, _, _) => "rocm_dtr_pre_gen8",
        (_, _, _, _, _) => "rocm_dtr_pre_gen4",
    }
}

/// Split-K ROCm trellis reduction with output rotation and epilogue.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "rocm_trellis_reduce_epi",
    sizes = [self.m, self.n, self.ks],
    work = self.m * self.n * self.ks,
    accepts = red_accepts,
    emits = f32_only,
    needs = rocm_needs(0, THREADS, false),
    order = Order::PerConfig,
    source = rocm_source(rocm_trellis_source(), "rocm_dtr_reduce_epi"),
    launches = red_launches,
    load = red_load,
    unproven,
)]
struct RocmTrellisRed {
    m: usize,
    n: usize,
    ks: usize,
    turned: bool,
    mode: u32,
}

fn red_accepts(r: &RocmTrellisRed, input: usize, form: &dyn Stored) -> bool {
    let other_at = 1 + usize::from(r.turned);
    if input >= other_at {
        return epi_accepts(r.mode, input, other_at, form);
    }
    form.name() == "f32"
}

fn red_load(r: &RocmTrellisRed, _: usize) -> Option<Load> {
    let mut load = reduce_load_for(r.m, r.n, r.ks, r.turned, false)?;
    push_epi_reads(&mut load.reads, r.mode, r.m, r.n);
    Some(load)
}

fn red_launches(r: &RocmTrellisRed, step: &Step) -> Vec<Launch> {
    let shape = Shape::Grid {
        across: r.n / BLOCK,
        down: r.m.div_ceil(8),
        threads: THREADS,
    };
    let svh = if r.turned { step.read(1) } else { step.read(0) };
    let other_at = 1 + usize::from(r.turned);
    let has_o = has_other(r.mode);
    let other = if has_o {
        step.read(other_at)
    } else {
        step.read(0)
    };
    let bias_at = other_at + usize::from(has_o);
    let bias = if has_bias2(r.mode) {
        step.read(bias_at)
    } else {
        step.read(0)
    };
    let args = vec![
        step.read(0),
        svh,
        other,
        bias,
        step.out(),
        Arg::count(r.m),
        Arg::count(r.n),
        Arg::count(r.ks),
        Arg::count(usize::from(r.turned)),
        Arg::value(&r.mode),
    ];
    vec![Launch { args, shape }]
}

fn dense_pick(tree: &Tree, at: T) -> Option<Dense> {
    Dense::find(tree, at, BLOCK)?.ok()
}

fn eval_rocm_trellis(tree: &Tree, at: T) -> Option<Result<Vec<Draft>, String>> {
    let Some((d, mode, other, bias2)) = match_epi_site(tree, at, dense_pick) else {
        return Dense::find(tree, at, BLOCK).and_then(|r| r.err().map(Err));
    };
    let (m, n, k, turned) = (d.m, d.n, d.k, d.turned());
    if m < 33 && mode != 0 {
        return None;
    }
    if m >= 33 {
        let [suh, svh] = d.turn.map_or([None, None], |[u, v]| [Some(u), Some(v)]);
        let p_reads = [d.x].into_iter().chain(suh).map(Read::Node);
        let prep = TrellisPrep {
            m,
            k,
            min_blocks: None,
            turned,
            x_bf16: false,
            rocm: true,
        };
        let pre_len = m.div_ceil(64) * (k / BLOCK) * 8 * 512;
        let ks = if d.bits == 4 && d.book == 0 && turned {
            rocm_pre_ks(m, n, k)
        } else {
            1
        };
        let tile = pre_tile(m, n);
        if ks > 1 {
            let mma = TrellisPreMma {
                m,
                n,
                k,
                ks,
                tile,
                min_blocks: None,
                bits: d.bits,
                book: d.book,
                turned,
                x_bf16: false,
                y_bf16: false,
                rocm: true,
                mode: 0,
                form: d.form,
            };
            let red = RocmTrellisRed {
                m,
                n,
                ks,
                turned,
                mode,
            };
            let r_reads = [Read::Draft(1)]
                .into_iter()
                .chain(svh.map(Read::Node))
                .chain(other.map(Read::Node))
                .chain(bias2.map(Read::Node));
            return Some(Ok(vec![
                Draft::new(prep, p_reads, pre_len),
                Draft::new(mma, [Read::Draft(0), Read::Node(d.w)], ks * m * n),
                Draft::new(red, r_reads, m * n),
            ]));
        }
        let m_reads = [Read::Draft(0), Read::Node(d.w)]
            .into_iter()
            .chain(svh.map(Read::Node))
            .chain(other.map(Read::Node))
            .chain(bias2.map(Read::Node));
        let mma = TrellisPreMma {
            m,
            n,
            k,
            ks: 1,
            tile,
            min_blocks: None,
            bits: d.bits,
            book: d.book,
            turned,
            x_bf16: false,
            y_bf16: false,
            rocm: true,
            mode,
            form: d.form,
        };
        return Some(Ok(vec![
            Draft::new(prep, p_reads, pre_len),
            Draft::new(mma, m_reads, m * n),
        ]));
    }
    let ks = dtr_ks_for(m, n, k, 4, [80, 40]);
    let mma = TrellisMma {
        m,
        n,
        k,
        ks,
        tile: small_tile(m),
        min_blocks: None,
        bits: d.bits,
        book: d.book,
        turned,
        x_bf16: false,
        y_bf16: false,
        rocm: true,
        form: d.form,
    };
    let red = TrellisReduce {
        m,
        n,
        ks,
        min_blocks: None,
        turned,
        y_bf16: false,
        rocm: true,
    };
    Some(Ok(d.drafts(mma, ks, red)))
}

rocm_rule!("rocm_trellis_mma", check = eval_rocm_trellis);
