//! Causal attention that walks the keys one at a time.

use crate::join::Join;
use codegen::kept;
use codegen::prelude::*;
use graph::Order;
use lower::kernels::{Map, MapStage};
use lower::sites;
use lower::{Arg, Attn, Draft, Kernel, Launch, Needs, Read, Shape, Source, Step, T, Tree};

/// Threads in a group: one group per query and head.
const THREADS: u32 = 256;

/// The widest head: two sums per thread.
const WIDEST: u32 = 2 * THREADS;

/// Every thread's share of a dot product, summed through `red`.
#[kernel]
fn group_dot(
    q: Buf<F32>,
    k: Buf<F32>,
    q_off: u64,
    k_off: u64,
    width: u32,
    red: &mut Shared<f32>,
) -> f32 {
    let mut partial = 0.0f32;
    for d in (lid()..width).by(THREADS) {
        partial += q[q_off + d as u64] * k[k_off + d as u64];
    }
    red[lid()] = partial;
    barrier();
    for level in 0..8 {
        let span = 128 >> level;
        if lid() < span {
            red[lid()] += red[lid() + span];
        }
        barrier();
    }
    let dot = red[0];
    barrier();
    dot
}

/// Folds a value row into `acc`, scaled by `corr` first.
#[kernel]
fn fold_value(acc: &mut [f32; 2], corr: f32, p: f32, v: Buf<F32>, v_off: u64, width: u32) {
    for i in 0..2 {
        let d = lid() + i * THREADS;
        if d < width {
            acc[i as usize] = acc[i as usize] * corr + p * v[v_off + d as u64];
        }
    }
}

/// Writes `acc` over `l` as the output row.
#[kernel]
fn write_row(acc: &mut [f32; 2], mut out: BufMut<F32>, q_off: u64, l: f32, width: u32) {
    let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
    for i in 0..2 {
        let d = lid() + i * THREADS;
        if d < width {
            out[q_off + d as u64] = acc[i as usize] * inv;
        }
    }
}

/// A group per query and head shares each dot product.
#[kernel]
fn attend(
    q: Buf<F32>,
    kv: Buf<F32>,
    mut out: BufMut<F32>,
    queries: u32,
    keys: u32,
    past: u32,
    heads: u32,
    kv_heads: u32,
    width: u32,
    back: u32,
    ahead: u32,
) {
    let group = gid() / THREADS;
    let tok = group / heads;
    let qh = group % heads;
    if tok >= queries {
        return;
    }
    let kvh = qh / (heads / kv_heads);
    let q_off = ((tok * heads + qh) * width) as u64;
    let mut red = shared::<f32>(THREADS);
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;
    let mut acc = [0.0f32; 2];
    // The query sees `back` keys behind it and `ahead` past.
    let position = past + tok;
    let first = if position > back { position - back } else { 0 };
    let last = if keys - position - 1 > ahead {
        position + ahead + 1
    } else {
        keys
    };
    for t in first..last {
        // A joined row: every head's key, then every head's value.
        let k_off = ((t * kv_heads * 2 + kvh) * width) as u64;
        let dot = group_dot(q, kv, q_off, k_off, width, &mut red);
        let m_new = max(m, dot);
        let corr = exp(m - m_new);
        let p = exp(dot - m_new);
        m = m_new;
        l = l * corr + p;
        let v_off = k_off + (kv_heads * width) as u64;
        fold_value(&mut acc, corr, p, kv, v_off, width);
    }
    write_row(&mut acc, out, q_off, l, width);
}

/// A group per query row and head, within its slot.
#[kernel]
fn attend_rows(
    q: Buf<F32>,
    k: Buf<F32>,
    v: Buf<F32>,
    mut out: BufMut<F32>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    rows: u32,
    keys: u32,
    heads: u32,
    kv_heads: u32,
    width: u32,
) {
    let group = gid() / THREADS;
    let row = group / heads;
    let qh = group % heads;
    if row >= rows {
        return;
    }
    let slot = slots[row as u64];
    let position = positions[row as u64];
    let kvh = qh / (heads / kv_heads);
    let q_off = ((row * heads + qh) * width) as u64;
    let mut red = shared::<f32>(THREADS);
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;
    let mut acc = [0.0f32; 2];
    // The row sees its slot up to its own position.
    let last = if position + 1 < keys {
        position + 1
    } else {
        keys
    };
    for t in 0..last {
        let kv_off = ((kept_row(slot, t) * kv_heads + kvh) * width) as u64;
        let dot = group_dot(q, k, q_off, kv_off, width, &mut red);
        let m_new = max(m, dot);
        let corr = exp(m - m_new);
        let p = exp(dot - m_new);
        m = m_new;
        l = l * corr + p;
        fold_value(&mut acc, corr, p, v, kv_off, width);
    }
    write_row(&mut acc, out, q_off, l, width);
}

/// Scaled query rows over kept keys, by slot and position.
#[derive(Clone, Copy, Debug)]
pub struct AttendRows(pub Attn);

impl Kernel for AttendRows {
    over_attn!("attend_rows");

    fn source(&self) -> Source {
        let body = kept::attach(attend_rows_ir(), self.0.kept);
        Source::named(body, format!("plan_attend_rows_{}", self.0.kept.name()))
    }

    /// One launch over every row.
    fn launches(&self, step: &Step) -> Vec<Launch> {
        let a = &self.0;
        let mut args = a.rows_args(step);
        args.push(Arg::count(a.width));
        args.extend(a.kept_args());
        vec![Launch {
            args,
            shape: Shape::Grid {
                across: a.queries * a.heads,
                down: 1,
                threads: THREADS as usize,
            },
        }]
    }

    fn work(&self, _: usize) -> f64 {
        let a = &self.0;
        (a.heads * a.queries * a.keys.most() * a.width) as f64
    }

    fn needs(&self) -> Needs {
        Needs {
            group_bytes: 4 * THREADS as usize,
            group_threads: THREADS as usize,
            ..Needs::default()
        }
    }
}

/// Reads scaled queries, and keys joined with values. Any mask.
#[derive(Clone, Copy, Debug)]
pub struct Attend(pub Attn);

impl Kernel for Attend {
    over_attn!("attend");

    fn source(&self) -> Source {
        Source::named(attend_ir(), "plan_attend")
    }

    /// One launch per sequence, each at its own offset.
    fn launches(&self, step: &Step) -> Vec<Launch> {
        let a = &self.0;
        let keys = a.keys.most();
        let (q, kv) = (
            a.queries * a.heads * a.width,
            keys * a.kv_heads * a.width * 2,
        );
        let buffers = |s: usize| {
            [
                Arg::at(step.reads[0], s * q),
                Arg::at(step.reads[1], s * kv),
                Arg::at(step.write, s * q),
            ]
        };
        let across = a.queries * a.heads;
        let launch = |s| a.launch(buffers(s), [Arg::count(a.width)], across, THREADS as usize);
        (0..a.seqs).map(launch).collect()
    }

    fn work(&self, _: usize) -> f64 {
        let a = &self.0;
        (a.seqs * a.heads * a.queries * a.seen() * a.width) as f64
    }

    fn needs(&self) -> Needs {
        Needs {
            group_bytes: 4 * THREADS as usize,
            group_threads: THREADS as usize,
            ..Needs::default()
        }
    }
}

/// Scale the queries, join keys with values, then attend.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Attend = tree.site(at)?;
    if site.sizes.width > WIDEST as usize {
        return None;
    }
    // Ragged rows read the kept keys and values in place.
    if site.rows.is_some() {
        return site.ragged(AttendRows(site.sizes));
    }
    let sites::Attend {
        scale,
        q,
        k,
        v,
        sizes,
        ..
    } = site;
    let out = site.out();
    // Sequences would sit at moving offsets as the keys grow.
    if sizes.keys.runtime() && sizes.seqs > 1 {
        return None;
    }
    let side = sizes.kv_heads * sizes.width;
    let out = sizes.seqs * out;
    let join = Join {
        left: side,
        right: side,
        rows: sizes.keys.times(sizes.seqs),
    };
    let joined = 2 * sizes.seqs * sizes.keys.most() * side;
    Some(vec![
        Draft::new(Map::plain(vec![MapStage::Scale(scale)]), [q], out),
        Draft::new(join, [k, v], joined),
        Draft::new(Attend(sizes), [Read::Draft(0), Read::Draft(1)], out),
    ])
}

lower::rule!("attend", propose);
