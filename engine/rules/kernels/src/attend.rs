//! Causal attention that walks the keys one at a time.

use crate::join::Join;
use codegen::prelude::*;
use graph::Order;
use lower::kernels::{Map, MapStage};
use lower::sites;
use lower::{Arg, Attn, Draft, Kernel, Launch, Needs, Read, Shape, Source, Step, T, Tree};

/// Threads in a group: one group per query and head.
const THREADS: u32 = 256;

/// The widest head: two sums per thread.
const WIDEST: u32 = 2 * THREADS;

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
    // The query sees every key up to its own position.
    let last = if past + tok + 1 < keys {
        past + tok + 1
    } else {
        keys
    };
    for t in 0..last {
        // A joined row: every head's key, then every head's value.
        let k_off = ((t * kv_heads * 2 + kvh) * width) as u64;
        let mut partial = 0.0f32;
        for d in (lid()..width).by(THREADS) {
            partial += q[q_off + d as u64] * kv[k_off + d as u64];
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
        let m_new = max(m, dot);
        let corr = exp(m - m_new);
        let p = exp(dot - m_new);
        m = m_new;
        l = l * corr + p;
        let v_off = k_off + (kv_heads * width) as u64;
        for i in 0..2 {
            let d = lid() + i * THREADS;
            if d < width {
                acc[i as usize] = acc[i as usize] * corr + p * kv[v_off + d as u64];
            }
        }
    }
    let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
    for i in 0..2 {
        let d = lid() + i * THREADS;
        if d < width {
            out[q_off + d as u64] = acc[i as usize] * inv;
        }
    }
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
    stride: u32,
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
        let kv_off = (((slot * stride + t) * kv_heads + kvh) * width) as u64;
        let mut partial = 0.0f32;
        for d in (lid()..width).by(THREADS) {
            partial += q[q_off + d as u64] * k[kv_off + d as u64];
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
        let m_new = max(m, dot);
        let corr = exp(m - m_new);
        let p = exp(dot - m_new);
        m = m_new;
        l = l * corr + p;
        for i in 0..2 {
            let d = lid() + i * THREADS;
            if d < width {
                acc[i as usize] = acc[i as usize] * corr + p * v[kv_off + d as u64];
            }
        }
    }
    let inv = if l > 0.0 { 1.0 / l } else { 0.0 };
    for i in 0..2 {
        let d = lid() + i * THREADS;
        if d < width {
            out[q_off + d as u64] = acc[i as usize] * inv;
        }
    }
}

/// Scaled query rows over kept keys, by slot and position.
#[derive(Clone, Copy, Debug)]
pub struct AttendRows(pub Attn);

impl Kernel for AttendRows {
    fn name(&self) -> &'static str {
        "attend_rows"
    }

    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn sizes(&self) -> Vec<usize> {
        self.0.sizes()
    }

    fn source(&self) -> Source {
        let mut body = attend_rows_ir();
        body.entry = "plan_attend_rows".into();
        Source::Made(body)
    }

    /// One launch over every row.
    fn launches(&self, step: &Step) -> Vec<Launch> {
        let a = &self.0;
        let mut args = a.rows_args(step);
        args.push(Arg::count(a.width));
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

/// Reads scaled queries, and keys joined with values.
#[derive(Clone, Copy, Debug)]
pub struct Attend(pub Attn);

impl Kernel for Attend {
    fn name(&self) -> &'static str {
        "attend"
    }

    fn order(&self) -> Order {
        Order::PerConfig
    }

    fn sizes(&self) -> Vec<usize> {
        self.0.sizes()
    }

    fn source(&self) -> Source {
        let mut body = attend_ir();
        body.entry = "plan_attend".into();
        Source::Made(body)
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
                Arg::Buffer(step.reads[0], s * q),
                Arg::Buffer(step.reads[1], s * kv),
                Arg::Buffer(step.write, s * q),
            ]
        };
        let width = std::iter::once(Arg::count(a.width));
        let launch = |s: usize| Launch {
            args: buffers(s)
                .into_iter()
                .chain(a.args())
                .chain(width.clone())
                .collect(),
            shape: Shape::Grid {
                across: a.queries * a.heads,
                down: 1,
                threads: THREADS as usize,
            },
        };
        (0..a.seqs).map(launch).collect()
    }

    fn work(&self, _: usize) -> f64 {
        let a = &self.0;
        (a.seqs * a.heads * a.queries * a.keys.most() * a.width) as f64
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
        Draft::new(Map(vec![MapStage::Scale(scale)]), [q], out),
        Draft::new(join, [k, v], joined),
        Draft::new(Attend(sizes), [Read::Draft(0), Read::Draft(1)], out),
    ])
}

lower::rule!("attend", propose);
