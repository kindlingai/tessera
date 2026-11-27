//! Fused CUDA RMS norm and add emitting OCP mxfp8.

use super::{CUDA_LACKS, cuda_rule, prefill_source_with, x_accepts};
use tessera_graph::{Order, Stage};
use tessera_lower::form::Load as Stored;
use tessera_lower::{
    Arg, Draft, Kernel, Launch, Load, Needs, Shape, Source, Step, T, Term, Tree, sites,
};

const ADD2: u32 = 1;
const ADD3: u32 = 2;
const WEIGHTED: u32 = 4;
const SAVE_SUM: u32 = 8;
const RELOAD_SUM: u32 = 16;

/// Fused RMS norm and optional residual add emitting OCP mxfp8.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "fast_norm_mxfp8",
    sizes = [self.rows, self.width],
    work = self.rows * self.width,
    accepts = norm_mx8_accepts,
    emits = norm_mx8_emits,
    emits_at = norm_mx8_emits_at,
    needs = norm_mx8_needs(),
    order = Order::PerConfig,
    source = norm_mx8_source(self),
    launches = norm_mx8_launches,
    shape = Shape::grid((self.rows, 1), 256),
    load = norm_mx8_load,
    unproven,
)]
struct NormMxfp8 {
    rows: usize,
    width: usize,
    eps: f32,
    flags: u32,
    bf16_mask: u8,
}

impl NormMxfp8 {
    fn has(&self, flag: u32) -> bool {
        (self.flags & flag) != 0
    }

    fn is_bf16(&self, bit: u8) -> bool {
        (self.bf16_mask & (1 << bit)) != 0
    }
}

fn norm_mx8_needs() -> Needs {
    Needs {
        lanes: 32,
        group_bytes: 32,
        group_threads: 256,
        compiler: true,
        lacks: CUDA_LACKS,
        mxfp8_tile: true,
        ..Needs::default()
    }
}

fn norm_mx8_source(k: &NormMxfp8) -> Source {
    const NAMES: [&str; 5] = ["A_BF16", "B_BF16", "C_BF16", "W_BF16", "S_BF16"];
    let defs: Vec<(&str, usize)> = NAMES
        .iter()
        .enumerate()
        .filter(|(i, _)| k.is_bf16(*i as u8))
        .map(|(_, &name)| (name, 1))
        .collect();
    prefill_source_with("tessera_norm_mxfp8", defs)
}

fn norm_mx8_accepts(k: &NormMxfp8, input: usize, form: &dyn Stored) -> bool {
    let bit = match input {
        0 => 0,
        1 if k.has(ADD2) => 1,
        1 if k.has(WEIGHTED) => 3,
        2 if k.has(ADD3) => 2,
        2 if k.has(ADD2) && k.has(WEIGHTED) => 3,
        3 if k.has(ADD3) && k.has(WEIGHTED) => 3,
        _ => return form.name() == "f32",
    };
    x_accepts(k.is_bf16(bit), form)
}

fn norm_mx8_emits(_: &NormMxfp8, form: &dyn Stored) -> bool {
    form.name() == "mxfp8"
}

fn norm_mx8_emits_at(k: &NormMxfp8, output: usize, form: &dyn Stored) -> bool {
    match output {
        0 => form.name() == "mxfp8",
        1 if k.has(SAVE_SUM) => x_accepts(k.is_bf16(4), form),
        _ => form.name() == "f32",
    }
}

fn norm_mx8_load(k: &NormMxfp8, _: usize) -> Option<Load> {
    let elems = (k.rows * k.width) as f64;
    let elem = |bit| if k.is_bf16(bit) { 2.0 } else { 4.0 };
    let mut reads = vec![elem(0) * elems];
    if k.has(ADD2) {
        reads.push(elem(1) * elems);
    }
    if k.has(ADD3) {
        reads.push(elem(2) * elems);
    }
    if k.has(WEIGHTED) {
        reads.push(elem(3) * k.width as f64);
    }
    let mut written = (33.0 / 32.0) * elems;
    if k.has(SAVE_SUM) {
        written += elem(4) * elems;
    }
    Some(Load {
        threads: k.rows * 256,
        group: 256,
        trips: k.width.div_ceil(1024) as f64,
        reads,
        written,
        flight: 512.0,
        ..Load::default()
    })
}

fn norm_mx8_launches(k: &NormMxfp8, step: &Step) -> Vec<Launch> {
    let a = step.read(0);
    let b = if k.has(ADD2) { step.read(1) } else { a.clone() };
    let c = if k.has(ADD3) { step.read(2) } else { a.clone() };
    let w_idx = 1 + usize::from(k.has(ADD2)) + usize::from(k.has(ADD3));
    let w = if k.has(WEIGHTED) {
        step.read(w_idx)
    } else {
        a.clone()
    };
    let out = step.out();
    let sum_out = if k.has(SAVE_SUM) {
        step.out_at(1)
    } else {
        out.clone()
    };
    let args = vec![
        a,
        b,
        c,
        w,
        out,
        sum_out,
        Arg::count(k.rows),
        Arg::count(k.width),
        Arg::value(&k.eps),
        Arg::value(&k.flags),
    ];
    vec![Launch {
        args,
        shape: k.shape().expect("a grid"),
    }]
}

fn elem_add(tree: &Tree, t: T, len: usize) -> Option<(T, T)> {
    let node = tree.node(t);
    if node.term != Term::Stage(Stage::Add) {
        return None;
    }
    let [a, b] = node.args.as_slice() else {
        return None;
    };
    (tree.elements(*a) == len && tree.elements(*b) == len).then_some((*a, *b))
}

fn bit_bf16(tree: &Tree, t: T, bit: u8) -> u8 {
    u8::from(tree.node(t).form.name() == "bf16") << bit
}

fn propose_norm_mx8(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    if tree.node(at).form.name() != "mxfp8" {
        return None;
    }
    let site: sites::Stage = tree.site(at)?;
    let (eps, weighted) = site.rms()?;
    if !site.width.is_multiple_of(32) {
        return None;
    }
    let rows = site.len / site.width;
    let w = site.operand.map(|(o, _)| o);
    let w_bit = w.map_or(0, |o| bit_bf16(tree, o, 3));
    let w_flag = if weighted { WEIGHTED } else { 0 };
    let make = |xs: &[T], save: Option<T>, choice: usize| {
        let mut flags = w_flag;
        let mut mask = w_bit | bit_bf16(tree, xs[0], 0);
        if xs.len() >= 2 {
            flags |= ADD2;
            mask |= bit_bf16(tree, xs[1], 1);
        }
        if xs.len() >= 3 {
            flags |= ADD3;
            mask |= bit_bf16(tree, xs[2], 2);
        }
        if let Some(sum) = save {
            flags |= SAVE_SUM;
            let s_bf16 = bit_bf16(tree, sum, 4);
            mask |= s_bf16;
            if s_bf16 == 0 {
                flags |= RELOAD_SUM;
            }
        }
        let reads: Vec<T> = xs.iter().copied().chain(w).collect();
        let kernel = NormMxfp8 {
            rows,
            width: site.width,
            eps,
            flags,
            bf16_mask: mask,
        };
        let mut draft = Draft::new(kernel, reads, site.len).choice(choice);
        if let Some(sum) = save {
            draft = draft.also(sum, site.len);
        }
        draft
    };
    let mut drafts = vec![make(&[site.x], None, 0)];
    if let Some((a, b)) = elem_add(tree, site.x, site.len) {
        let save = tree.root(site.x).then_some(site.x);
        drafts.push(make(&[a, b], save, 1));
        let triple = (!tree.root(b))
            .then(|| elem_add(tree, b, site.len).map(|(u, v)| [u, v, a]))
            .flatten()
            .or_else(|| {
                (!tree.root(a))
                    .then(|| elem_add(tree, a, site.len).map(|(u, v)| [u, v, b]))
                    .flatten()
            });
        if let Some(xs) = triple {
            drafts.push(make(&xs, save, 2));
        }
    }
    Some(drafts)
}

cuda_rule!("fast_norm_mxfp8", propose_norm_mx8);
