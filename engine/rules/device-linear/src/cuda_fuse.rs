//! Fused CUDA/ROCm trellis prep (norm + multi-output) and direct xpre chains.

use super::gemv::{CudaTrellisRed, cuda_pre_ks, dense_pick, match_cuda_site};
use super::rocm_trellis::rocm_pre_ks;
use super::{
    BLOCK, THREADS, TrellisPreMma, TrellisPrep, dtr_needs, f32_only, rocm_needs, rocm_source,
};
use std::sync::OnceLock;
use tessera_device::{Backend, flatten_for};
use tessera_graph::{Order, Stage};
use tessera_linear::Dense;
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, ByBackend, Draft, Kernel, Launch, Load, Read, Shape, Source, Step, T, Term, Tree,
};

const PREP_FUSE: &str = include_str!("prep_fuse.cu");

fn prep_fuse_raw() -> &'static str {
    static CUDA: OnceLock<&'static str> = OnceLock::new();
    CUDA.get_or_init(|| flatten_for(Backend::Cuda, "prep_fuse.kernel", PREP_FUSE))
}

fn prep_fuse_rocm() -> &'static str {
    static ROCM: OnceLock<&'static str> = OnceLock::new();
    ROCM.get_or_init(|| flatten_for(Backend::Rocm, "prep_fuse.kernel", PREP_FUSE))
}

fn prep_fuse_source(mx8: bool, rocm: bool) -> Source {
    static CUDA_F16: OnceLock<&'static str> = OnceLock::new();
    if rocm {
        return rocm_source(prep_fuse_rocm(), "dtr_prep_fused");
    }
    let text = if mx8 {
        prep_fuse_raw()
    } else {
        *CUDA_F16.get_or_init(|| {
            Box::leak(format!("#define DTR_FORCE_FP16 1\n{}", prep_fuse_raw()).into_boxed_str())
        })
    };
    Source::Fixed {
        text: ByBackend(["", text, ""]),
        entry: "dtr_prep_fused",
    }
}

fn pre_len(m: usize, k: usize) -> usize {
    m.div_ceil(64) * (k / BLOCK) * 8 * 512
}

fn prep_fused_name(p: &TrellisPrepFused) -> &'static str {
    match (p.rocm, p.flags & 32 != 0, p.flags & 16 != 0, p.count) {
        (true, true, true, _) => "rocm_trellis_prep_ln",
        (true, true, false, 1) => "rocm_trellis_prep_rms",
        (true, true, false, 2) => "rocm_trellis_prep_rms2",
        (true, true, false, _) => "rocm_trellis_prep_rms3",
        (true, false, _, 2) => "rocm_trellis_prep_pair",
        (true, false, _, _) => "rocm_trellis_prep_multi",
        (false, true, true, _) => "trellis_prep_ln",
        (false, true, false, 1) => "trellis_prep_rms",
        (false, true, false, 2) => "trellis_prep_rms2",
        (false, true, false, _) => "trellis_prep_rms3",
        (false, false, _, 2) => "trellis_prep_pair",
        (false, false, _, _) => "trellis_prep_multi",
    }
}

/// Fused norm and multi-output trellis input rotation.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = prep_fused_name(self),
    sizes = [self.m, self.k, self.count],
    work = self.m * self.k * self.count,
    accepts = prep_fused_accepts,
    emits = f32_only,
    needs = if self.rocm { rocm_needs(4096, THREADS, true) } else { dtr_needs(4096) },
    order = Order::PerConfig,
    source = prep_fuse_source(self.mx8, self.rocm),
    launches = prep_fused_launches,
    load = prep_fused_load,
    shared,
    unproven,
)]
struct TrellisPrepFused {
    m: usize,
    k: usize,
    count: usize,
    eps: f32,
    flags: u32,
    mx8: bool,
    rocm: bool,
}

fn prep_fused_accepts(p: &TrellisPrepFused, input: usize, form: &dyn Stored) -> bool {
    if input == 0 {
        return form.name() == "f32";
    }
    let mut idx = 1;
    if (p.flags & 1) != 0 {
        if input == idx {
            return form.name() == if (p.flags & 2) != 0 { "bf16" } else { "f32" };
        }
        idx += 1;
    }
    if (p.flags & 4) != 0 && input == idx {
        return form.name() == if (p.flags & 8) != 0 { "bf16" } else { "f32" };
    }
    form.name() == "f32"
}

fn prep_fused_load(p: &TrellisPrepFused, _: usize) -> Option<Load> {
    let share = p.count.max(1) as f64;
    let across = if (p.flags & 32) != 0 { 1 } else { p.k / BLOCK };
    let mut reads = vec![(4 * p.m * p.k) as f64 / share];
    if (p.flags & 1) != 0 {
        reads.push((if (p.flags & 2) != 0 { 2 } else { 4 } * p.k) as f64 / share);
    }
    if (p.flags & 4) != 0 {
        reads.push((if (p.flags & 8) != 0 { 2 } else { 4 } * p.k) as f64 / share);
    }
    if (p.flags & 64) != 0 {
        for _ in 0..p.count {
            reads.push((4 * p.k) as f64 / share);
        }
    }
    Some(Load {
        threads: across * p.m.div_ceil(8) * THREADS,
        group: THREADS,
        trips: 1.0,
        reads,
        written: (p.m * p.k) as f64,
        flight: 512.0,
        ..Load::default()
    })
}

fn prep_fused_launches(p: &TrellisPrepFused, step: &Step) -> Vec<Launch> {
    let across = if (p.flags & 32) != 0 { 1 } else { p.k / BLOCK };
    let shape = Shape::Grid {
        across,
        down: p.m.div_ceil(8),
        threads: THREADS,
    };
    let mut idx = 1;
    let w_norm = if (p.flags & 1) != 0 {
        let a = step.read(idx);
        idx += 1;
        a
    } else {
        step.read(0)
    };
    let b_norm = if (p.flags & 4) != 0 {
        let a = step.read(idx);
        idx += 1;
        a
    } else {
        step.read(0)
    };
    let turned = (p.flags & 64) != 0;
    let suh0 = if turned { step.read(idx) } else { step.read(0) };
    let suh1 = if turned && p.count >= 2 {
        step.read(idx + 1)
    } else if turned {
        step.read(idx)
    } else {
        step.read(0)
    };
    let suh2 = if turned && p.count >= 3 {
        step.read(idx + 2)
    } else if turned {
        step.read(idx)
    } else {
        step.read(0)
    };
    let args = vec![
        step.read(0),
        w_norm,
        b_norm,
        suh0,
        suh1,
        suh2,
        step.out(),
        Arg::count(p.m),
        Arg::count(p.k),
        Arg::value(&p.eps),
        Arg::count(p.count),
        Arg::value(&p.flags),
    ];
    vec![Launch { args, shape }]
}

#[derive(Clone, Copy)]
struct NormSpec {
    x: T,
    w: Option<T>,
    b: Option<T>,
    eps: f32,
    center: bool,
}

fn plain_1d(tree: &Tree, t: T, width: usize) -> Option<bool> {
    if tree.elements(t) != width || tree.node(t).shape.last() != Some(&width) {
        return None;
    }
    match tree.node(t).form.name() {
        "f32" => Some(false),
        "bf16" => Some(true),
        _ => None,
    }
}

fn match_norm_input(tree: &Tree, h: T, m: usize, k: usize) -> Option<NormSpec> {
    let len = m * k;
    let node = tree.node(h);
    if node.form.name() != "f32" || tree.elements(h) != len || node.shape.last() != Some(&k) {
        return None;
    }
    let (norm_t, b) = if node.term == Term::Stage(Stage::Add) && node.args.len() == 2 {
        let (a, c) = (node.args[0], node.args[1]);
        let pair = if tree.elements(a) == len && plain_1d(tree, c, k).is_some() {
            Some((a, c))
        } else if tree.elements(c) == len && plain_1d(tree, a, k).is_some() {
            Some((c, a))
        } else {
            None
        };
        let (nt, bt) = pair?;
        if tree.root(nt) {
            return None;
        }
        (nt, Some(bt))
    } else {
        (h, None)
    };
    let n_node = tree.node(norm_t);
    if n_node.shape.last() != Some(&k) {
        return None;
    }
    let (src, w, eps) = match n_node.term {
        Term::Stage(Stage::RmsNorm { eps }) => {
            let wt = n_node.args[1];
            plain_1d(tree, wt, k)?;
            (n_node.args[0], Some(wt), eps)
        }
        Term::Stage(Stage::Rms { eps }) => (n_node.args[0], None, eps),
        _ => return None,
    };
    let (x, center) = if !tree.root(src) && tree.node(src).term == Term::Stage(Stage::Center) {
        let c_node = tree.node(src);
        if c_node.shape.last() != Some(&k) {
            return None;
        }
        (c_node.args[0], true)
    } else {
        (src, false)
    };
    if tree.elements(x) != len
        || tree.node(x).shape.last() != Some(&k)
        || tree.node(x).form.name() != "f32"
    {
        return None;
    }
    Some(NormSpec {
        x,
        w,
        b,
        eps,
        center,
    })
}

#[allow(
    clippy::too_many_arguments,
    reason = "fused prep builder takes norm and backend flags"
)]
fn build_fused_prep(
    tree: &Tree,
    h: T,
    norm: Option<NormSpec>,
    suhs: &[T],
    m: usize,
    k: usize,
    mx8: bool,
    rocm: bool,
) -> Draft {
    let mut flags = 64u32;
    let mut reads = Vec::with_capacity(3 + suhs.len());
    let eps = if let Some(ns) = norm {
        flags |= 32;
        if ns.center {
            flags |= 16;
        }
        reads.push(Read::Node(ns.x));
        if let Some(w) = ns.w {
            flags |= 1;
            if tree.node(w).form.name() == "bf16" {
                flags |= 2;
            }
            reads.push(Read::Node(w));
        }
        if let Some(b) = ns.b {
            flags |= 4;
            if tree.node(b).form.name() == "bf16" {
                flags |= 8;
            }
            reads.push(Read::Node(b));
        }
        ns.eps
    } else {
        reads.push(Read::Node(h));
        0.0
    };
    for &s in suhs {
        reads.push(Read::Node(s));
    }
    let kernel = TrellisPrepFused {
        m,
        k,
        count: suhs.len(),
        eps,
        flags,
        mx8,
        rocm,
    };
    Draft::new(kernel, reads, suhs.len() * pre_len(m, k))
}

/// Finds shared norm and multi-suh prep for rotated trellis readers.
#[allow(
    clippy::too_many_arguments,
    reason = "shared prep planner takes graph site and backend flags"
)]
fn shared_prep_plan(
    tree: &Tree,
    at: T,
    h: T,
    m: usize,
    k: usize,
    mx8: bool,
    rocm: bool,
    absorb_norm: bool,
) -> Option<(Draft, Vec<T>)> {
    if tree.outputs.iter().any(|(o, _)| *o == h) || tree.appends(h) {
        return None;
    }
    let mut direct = Vec::new();
    for i in (h.0 + 1)..(tree.len() as u32) {
        let t = T(i);
        if tree.node(t).args.contains(&h) {
            direct.push(t);
            if direct.len() > 3 {
                return None;
            }
        }
    }
    if direct.is_empty() {
        return None;
    }
    for &dr in &direct {
        let n = tree.node(dr);
        if tree.root(dr) || n.term != Term::Stage(Stage::Mul) || n.args.first() != Some(&h) {
            return None;
        }
    }
    let want_c = tree.compute_at(at);
    let mut found = 0usize;
    let mut suhs = Vec::with_capacity(direct.len());
    for i in (h.0 + 1)..(tree.len() as u32) {
        let t = T(i);
        let n = tree.node(t);
        if n.term != Term::Stage(Stage::Mul) {
            continue;
        }
        let Some(&prev) = n.args.first() else {
            continue;
        };
        if !matches!(
            tree.node(prev).term,
            Term::Stage(Stage::Hadamard { block }) if block as usize == BLOCK
        ) {
            continue;
        }
        let Some(Ok(c)) = Dense::find(tree, t, BLOCK) else {
            continue;
        };
        if c.x == h
            && c.m == m
            && c.k == k
            && tree.compute_at(t) == want_c
            && let Some([suh, _]) = c.turn
        {
            found += 1;
            suhs.push(suh);
        }
    }
    if found != direct.len() {
        return None;
    }
    suhs.sort_by_key(|t| t.0);
    suhs.dedup();
    // Fuse norm only when 8-row CTAs saturate SMs.
    let norm_fits_sm = m >= 384 && m * k >= 1_500_000;
    let can_absorb = absorb_norm && norm_fits_sm && direct.len() <= 2;
    let norm = if can_absorb {
        match_norm_input(tree, h, m, k)
    } else {
        None
    };
    if norm.is_none() && suhs.len() < 2 {
        return None;
    }
    let draft = build_fused_prep(tree, h, norm, &suhs, m, k, mx8, rocm);
    Some((draft, suhs))
}

fn plain_prep_draft(d: Dense, mx8: bool, rocm: bool) -> Draft {
    let suh = d.turn.map(|[u, _]| u);
    let reads = [d.x].into_iter().chain(suh).map(Read::Node);
    let prep = TrellisPrep {
        m: d.m,
        k: d.k,
        turned: d.turned(),
        rocm,
        mx8,
    };
    Draft::new(prep, reads, pre_len(d.m, d.k))
}

#[allow(
    clippy::too_many_arguments,
    reason = "trellis draft builder takes site fields"
)]
fn append_trellis_drafts(
    drafts: &mut Vec<Draft>,
    tree: &Tree,
    d: Dense,
    xpre: Read,
    xslot: u32,
    epi: (u32, Option<Read>, Option<Read>),
    rot_out: Option<T>,
    mx8: bool,
    rocm: bool,
) -> usize {
    let (m, n, k, turned) = (d.m, d.n, d.k, d.turned());
    let (mode, other, bias2) = epi;
    let svh = d.turn.map(|[_, v]| Read::Node(v));
    let ks = if d.bits == 4 && d.book == 0 && turned {
        if rocm {
            rocm_pre_ks(m, n, k)
        } else {
            cuda_pre_ks(tree, m, n, k)
        }
    } else {
        1
    };
    let out_len = if rot_out.is_some() {
        pre_len(m, n)
    } else {
        m * n
    };
    let slot_bits = xslot << 6;
    if ks > 1 {
        let mma_idx = drafts.len();
        let mma = TrellisPreMma {
            m,
            n,
            k,
            ks,
            bits: d.bits,
            book: d.book,
            turned,
            rocm,
            mx8: !rocm && mx8,
            mode: slot_bits,
            rot_out: false,
            form: d.form,
        };
        drafts.push(Draft::new(mma, [xpre, Read::Node(d.w)], ks * m * n));
        let r_mode = mode
            | if rot_out.is_some() {
                32 | if !rocm && mx8 { 256 } else { 0 }
            } else {
                0
            };
        let red = CudaTrellisRed {
            m,
            n,
            ks,
            turned,
            mode: r_mode,
            rocm,
        };
        let r_reads = [Read::Draft(mma_idx)]
            .into_iter()
            .chain(svh)
            .chain(other)
            .chain(bias2)
            .chain(rot_out.map(Read::Node));
        let red_idx = drafts.len();
        drafts.push(Draft::new(red, r_reads, out_len));
        return red_idx;
    }
    let m_reads = [xpre, Read::Node(d.w)]
        .into_iter()
        .chain(svh)
        .chain(other)
        .chain(bias2)
        .chain(rot_out.map(Read::Node));
    let mma_idx = drafts.len();
    let mma = TrellisPreMma {
        m,
        n,
        k,
        ks: 1,
        bits: d.bits,
        book: d.book,
        turned,
        rocm,
        mx8: !rocm && mx8,
        mode: mode | slot_bits,
        rot_out: rot_out.is_some(),
        form: d.form,
    };
    drafts.push(Draft::new(mma, m_reads, out_len));
    mma_idx
}

fn try_fuse_chain(
    tree: &Tree,
    at: T,
    d: Dense,
    epi: (u32, Option<T>, Option<T>),
    mx8: bool,
    rocm: bool,
) -> Option<Vec<Draft>> {
    let [d_suh, _] = d.turn?;
    if tree.root(d.x) || tree.compute_at(d.x) != tree.compute_at(at) {
        return None;
    }
    let (p, p_mode, p_other, p_bias2) = match_cuda_site(tree, d.x, dense_pick)?;
    if p_bias2.is_some() || !p.turned() || p.bits != 4 || p.book != 0 || p.m != d.m || p.n != d.k {
        return None;
    }
    let [p_suh, _] = p.turn?;
    let (d_mode, d_other, d_bias2) = epi;
    let mut drafts = Vec::with_capacity(6);
    if (p_mode & 15) == 3
        && let Some(up_t) = p_other
        && !tree.root(up_t)
        && tree.compute_at(up_t) == tree.compute_at(at)
        && let Some(up) = dense_pick(tree, up_t)
        && up.x == p.x
        && up.m == p.m
        && up.n == p.n
        && up.k == p.k
        && let Some([up_suh, _]) = up.turn
    {
        let (prep_d, up_slot, gate_slot) =
            if let Some((pd, suhs)) = shared_prep_plan(tree, d.x, p.x, p.m, p.k, mx8, rocm, true) {
                let us = suhs.iter().position(|&s| s == up_suh)? as u32;
                let gs = suhs.iter().position(|&s| s == p_suh)? as u32;
                (pd, us, gs)
            } else if up_suh == p_suh {
                (plain_prep_draft(up, mx8, rocm), 0, 0)
            } else {
                let mut suhs = vec![up_suh, p_suh];
                suhs.sort_by_key(|t| t.0);
                let us = suhs.iter().position(|&s| s == up_suh)? as u32;
                let gs = suhs.iter().position(|&s| s == p_suh)? as u32;
                (
                    build_fused_prep(tree, p.x, None, &suhs, p.m, p.k, mx8, rocm),
                    us,
                    gs,
                )
            };
        drafts.push(prep_d);
        let up_idx = append_trellis_drafts(
            &mut drafts,
            tree,
            up,
            Read::Draft(0),
            up_slot,
            (0, None, None),
            None,
            mx8,
            rocm,
        );
        let xpre_down = append_trellis_drafts(
            &mut drafts,
            tree,
            p,
            Read::Draft(0),
            gate_slot,
            (p_mode, Some(Read::Draft(up_idx)), None),
            Some(d_suh),
            mx8,
            rocm,
        );
        append_trellis_drafts(
            &mut drafts,
            tree,
            d,
            Read::Draft(xpre_down),
            0,
            (d_mode, d_other.map(Read::Node), d_bias2.map(Read::Node)),
            None,
            mx8,
            rocm,
        );
        return Some(drafts);
    }
    let (prep_d, p_slot) = if let Some((pd, suhs)) =
        shared_prep_plan(tree, d.x, p.x, p.m, p.k, mx8, rocm, !tree.root(p.x))
    {
        let ps = suhs.iter().position(|&s| s == p_suh)? as u32;
        (pd, ps)
    } else {
        (plain_prep_draft(p, mx8, rocm), 0)
    };
    drafts.push(prep_d);
    let xpre_next = append_trellis_drafts(
        &mut drafts,
        tree,
        p,
        Read::Draft(0),
        p_slot,
        (p_mode, p_other.map(Read::Node), None),
        Some(d_suh),
        mx8,
        rocm,
    );
    append_trellis_drafts(
        &mut drafts,
        tree,
        d,
        Read::Draft(xpre_next),
        0,
        (d_mode, d_other.map(Read::Node), d_bias2.map(Read::Node)),
        None,
        mx8,
        rocm,
    );
    Some(drafts)
}

#[allow(
    clippy::too_many_arguments,
    reason = "trellis prefill planner takes site and backend flags"
)]
pub(super) fn plan_trellis_pre(
    tree: &Tree,
    at: T,
    d: Dense,
    mode: u32,
    other: Option<T>,
    bias2: Option<T>,
    mx8: bool,
    rocm: bool,
) -> Vec<Draft> {
    if let Some(fused) = try_fuse_chain(tree, at, d, (mode, other, bias2), mx8, rocm) {
        return fused;
    }
    let (prep_d, slot) = if let Some([my_suh, _]) = d.turn
        && let Some((pd, suhs)) =
            shared_prep_plan(tree, at, d.x, d.m, d.k, mx8, rocm, !tree.root(d.x))
        && let Some(pos) = suhs.iter().position(|&s| s == my_suh)
    {
        (pd, pos as u32)
    } else {
        (plain_prep_draft(d, mx8, rocm), 0)
    };
    let mut drafts = vec![prep_d];
    append_trellis_drafts(
        &mut drafts,
        tree,
        d,
        Read::Draft(0),
        slot,
        (mode, other.map(Read::Node), bias2.map(Read::Node)),
        None,
        mx8,
        rocm,
    );
    drafts
}
