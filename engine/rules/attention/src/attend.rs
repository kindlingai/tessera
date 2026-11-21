//! Causal attention that walks the keys one at a time.

use tessera_codegen::kept;
use tessera_codegen::prelude::*;
use tessera_graph::Order;
use tessera_lower::kernels::Join;
use tessera_lower::sites;
use tessera_lower::{Arg, Attn, Draft, Kernel, Launch, Needs, Read, Shape, Source, Step, T, Tree};

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

/// Folds one key into the peak, sum and `acc`.
#[kernel]
fn take_key(
    state: &mut [f32; 2],
    acc: &mut [f32; 2],
    dot: f32,
    v: Buf<F32>,
    v_off: u64,
    width: u32,
) {
    let m_new = max(state[0], dot);
    let corr = exp(state[0] - m_new);
    let p = exp(dot - m_new);
    state[0] = m_new;
    state[1] = state[1] * corr + p;
    for i in 0..2 {
        let d = lid() + i * THREADS;
        if d < width {
            acc[i as usize] = acc[i as usize] * corr + p * v[v_off + d as u64];
        }
    }
}

/// Writes `acc` over the sum in `state` as the row.
#[kernel]
fn write_row(state: &[f32; 2], acc: &[f32; 2], mut out: BufMut<F32>, q_off: u64, width: u32) {
    let inv = if state[1] > 0.0 { 1.0 / state[1] } else { 0.0 };
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
    let mut state = [0.0f32; 2];
    state[0] = f32::NEG_INFINITY;
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
        let v_off = k_off + (kv_heads * width) as u64;
        take_key(&mut state, &mut acc, dot, kv, v_off, width);
    }
    write_row(&state, &acc, out, q_off, width);
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
    back: u32,
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
    let mut state = [0.0f32; 2];
    state[0] = f32::NEG_INFINITY;
    let mut acc = [0.0f32; 2];
    // The row sees its own key and `back` before it.
    let last = if position + 1 < keys {
        position + 1
    } else {
        keys
    };
    let first = if position > back { position - back } else { 0 };
    for t in first..last {
        let kv_off = ((kept_row(slot, t) * kv_heads + kvh) * width) as u64;
        let dot = group_dot(q, k, q_off, kv_off, width, &mut red);
        take_key(&mut state, &mut acc, dot, v, kv_off, width);
    }
    write_row(&state, &acc, out, q_off, width);
}

/// Scaled query rows over kept keys, by slot and position.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "attend_rows",
    order = Order::PerConfig,
    sizes = self.0.sizes(),
    work = self.0.heads * self.0.queries * self.0.keys.most() * self.0.width,
    needs = group_needs(),
    source = rows_source,
    launches = rows_launches,
)]
pub(crate) struct AttendRows(pub Attn);

fn rows_source(k: &AttendRows) -> Source {
    let body = kept::attach(attend_rows_ir(), k.0.kept);
    Source::named(body, format!("plan_attend_rows_{}", k.0.kept.name()))
}

/// One launch over every row.
fn rows_launches(k: &AttendRows, step: &Step) -> Vec<Launch> {
    let a = &k.0;
    let mut args = a.rows_args(step);
    args.extend([Arg::count(a.width), a.back()]);
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

/// Reads scaled queries, and keys joined with values. Any mask.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    attend,
    name = "attend",
    order = Order::PerConfig,
    sizes = self.0.sizes(),
    work = self.0.seqs * self.0.heads * self.0.queries * self.0.seen() * self.0.width,
    needs = group_needs(),
    launches = attend_launches,
)]
pub(crate) struct Attend(pub Attn);

/// One launch per sequence, each at its own offset.
fn attend_launches(k: &Attend, step: &Step) -> Vec<Launch> {
    let a = &k.0;
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

fn group_needs() -> Needs {
    Needs {
        group_bytes: 4 * THREADS as usize,
        group_threads: THREADS as usize,
        ..Needs::default()
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
    let (scaled, out) = crate::scaled(&site)?;
    let sizes = site.sizes;
    let side = sizes.kv_heads * sizes.width;
    let join = Join {
        left: side,
        right: side,
        rows: sizes.keys.times(sizes.seqs),
    };
    let joined = 2 * sizes.seqs * sizes.keys.most() * side;
    Some(vec![
        scaled,
        Draft::new(join, [site.k, site.v], joined),
        Draft::new(Attend(sizes), [Read::Draft(0), Read::Draft(1)], out),
    ])
}

tessera_lower::rule!("attend", propose);
