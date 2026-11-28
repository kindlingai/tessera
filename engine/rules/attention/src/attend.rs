//! Causal attention that walks the keys one at a time.

use tessera_codegen::kept;
use tessera_codegen::prelude::*;
use tessera_graph::Order;
use tessera_lower::form::{F32_FORM, Load as Stored};
use tessera_lower::kernels::Join;
use tessera_lower::sites;
use tessera_lower::{
    Arg, Attn, Draft, Kernel, Launch, Needs, Read, Shape, Source, Step, T, Tree, knob,
};

/// Threads in a group: one group per query and head.
const THREADS: u32 = 256;

knob!("attend", "threads", 32, 1024);

/// Every thread's share of a dot product, summed through `red`.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn group_dot(
    q: Buf<F32>,
    #[form(k_form)] k: Buf<Dyn>,
    q_off: u64,
    k_off: u64,
    width: u32,
    red: &mut Shared<f32>,
    #[build] k_form: &'static dyn Stored,
    #[build] threads: u32,
    #[build] steps: u32,
    #[build] half: u32,
) -> f32 {
    let mut partial = 0.0f32;
    for d in (lid()..width).by(threads) {
        partial += q[q_off + d as u64] * k[k_off + d as u64];
    }
    red[lid()] = partial;
    barrier();
    for level in 0..steps {
        let span = half >> level;
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
    #[form(v_form)] v: Buf<Dyn>,
    v_off: u64,
    width: u32,
    #[build] v_form: &'static dyn Stored,
    #[build] threads: u32,
) {
    let m_new = max(state[0], dot);
    let corr = exp(state[0] - m_new);
    let p = exp(dot - m_new);
    state[0] = m_new;
    state[1] = state[1] * corr + p;
    for i in 0..2 {
        let d = lid() + i * threads;
        if d < width {
            acc[i as usize] = acc[i as usize] * corr + p * v[v_off + d as u64];
        }
    }
}

/// Writes `acc` over the sum in `state` as the row.
#[kernel]
fn write_row(
    state: &[f32; 2],
    acc: &[f32; 2],
    #[form(out_form)] mut out: BufMut<Dyn>,
    q_off: u64,
    width: u32,
    #[build] out_form: &'static dyn Stored,
    #[build] threads: u32,
) {
    let inv = if state[1] > 0.0 { 1.0 / state[1] } else { 0.0 };
    for i in 0..2 {
        let d = lid() + i * threads;
        if d < width {
            out[q_off + d as u64] = acc[i as usize] * inv;
        }
    }
}

/// A group per query and head shares each dot product.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn attend(
    q: Buf<F32>,
    #[form(F32_FORM)] kv: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    queries: u32,
    keys: u32,
    past: u32,
    heads: u32,
    kv_heads: u32,
    width: u32,
    back: u32,
    ahead: u32,
    #[build] out_form: &'static dyn Stored,
    #[build] threads: u32,
    #[build] steps: u32,
    #[build] half: u32,
) {
    let group = gid() / threads;
    let tok = group / heads;
    let qh = group % heads;
    if tok >= queries {
        return;
    }
    let kvh = qh / (heads / kv_heads);
    let q_off = ((tok * heads + qh) * width) as u64;
    let mut red = shared::<f32>(threads);
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
        let dot = group_dot(
            q, kv, q_off, k_off, width, &mut red, F32_FORM, threads, steps, half,
        );
        let v_off = k_off + (kv_heads * width) as u64;
        take_key(
            &mut state, &mut acc, dot, kv, v_off, width, F32_FORM, threads,
        );
    }
    write_row(&state, &acc, out, q_off, width, out_form, threads);
}

/// Attends rows over kept slots or packed bounds.
#[kernel]
#[allow(clippy::too_many_arguments)]
fn attend_indexed<const KEPT: bool, const CAUSAL: bool>(
    q: Buf<F32>,
    #[form(k_form)] k: Buf<Dyn>,
    #[form(v_form)] v: Buf<Dyn>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    bounds: Buf<U32>,
    rows: u32,
    keys: u32,
    heads: u32,
    kv_heads: u32,
    width: u32,
    back: u32,
    #[build] k_form: &'static dyn Stored,
    #[build] v_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
    #[build] threads: u32,
    #[build] steps: u32,
    #[build] half: u32,
) {
    let group = gid() / threads;
    let row = group / heads;
    let qh = group % heads;
    if row >= rows {
        return;
    }
    let kvh = qh / (heads / kv_heads);
    let q_off = ((row * heads + qh) * width) as u64;
    let mut red = shared::<f32>(threads);
    let mut state = [0.0f32; 2];
    state[0] = f32::NEG_INFINITY;
    let mut acc = [0.0f32; 2];
    let mut slot = 0u32;
    let mut first = 0u32;
    let mut last = 0u32;
    if KEPT {
        slot = slots[row as u64];
        let position = positions[row as u64];
        // The row sees its own key and `back` before it.
        last = if position + 1 < keys {
            position + 1
        } else {
            keys
        };
        first = if position > back { position - back } else { 0 };
    } else {
        first = bounds[(row * 2) as u64];
        let end = bounds[(row * 2 + 1) as u64];
        last = if CAUSAL && row + 1 < end {
            row + 1
        } else {
            end
        };
        if last > rows {
            last = rows;
        }
        if first > last {
            first = last;
        }
    }
    for t in first..last {
        let key_row = if KEPT { kept_row(slot, t) } else { t };
        let kv_off = ((key_row * kv_heads + kvh) * width) as u64;
        let dot = group_dot(
            q, k, q_off, kv_off, width, &mut red, k_form, threads, steps, half,
        );
        take_key(&mut state, &mut acc, dot, v, kv_off, width, v_form, threads);
    }
    write_row(&state, &acc, out, q_off, width, out_form, threads);
}

/// Packed rows over variable-length segments.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "attend_packed",
    order = Order::PerConfig,
    sizes = self.0.sizes(),
    work = self.0.heads * self.0.queries * self.0.queries * self.0.width,
    needs = group_needs(THREADS),
    accepts = packed_accepts,
    emits = packed_emits,
    source = packed_source,
    launches = packed_launches,
    shape = Shape::grid((self.0.queries * self.0.heads, 1), THREADS as usize),
)]
pub(crate) struct AttendPacked(pub Attn);

fn packed_accepts(k: &AttendPacked, input: usize, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == k.0.k_form.name(),
        2 => form.name() == k.0.v_form.name(),
        _ => !form.narrow(),
    }
}

fn packed_emits(k: &AttendPacked, form: &dyn Stored) -> bool {
    form.name() == k.0.out_form.name()
}

fn packed_source(k: &AttendPacked) -> Source {
    let a = &k.0;
    let causal = a.mask.causal();
    let (steps, half) = (THREADS.trailing_zeros(), THREADS / 2);
    let mut body = if causal {
        attend_indexed_ir::<false, true>(a.k_form, a.v_form, a.out_form, THREADS, steps, half)
    } else {
        attend_indexed_ir::<false, false>(a.k_form, a.v_form, a.out_form, THREADS, steps, half)
    };
    body.params
        .retain(|p| !matches!(p.name(), "slots" | "positions" | "keys" | "back"));
    let kind = if causal { "causal" } else { "full" };
    let entry = match (a.k_form.name(), a.v_form.name(), a.out_form.name()) {
        ("f32", "f32", "f32") => format!("plan_attend_packed_{kind}"),
        (kf, vf, of) => format!("plan_attend_packed_{kind}_{kf}_{vf}_{of}"),
    };
    Source::named(body, entry)
}

fn packed_launches(k: &AttendPacked, step: &Step) -> Vec<Launch> {
    let a = &k.0;
    let args = vec![
        step.read(0),
        step.read(1),
        step.read(2),
        step.out(),
        step.read(3),
        Arg::count(a.queries),
        Arg::count(a.heads),
        Arg::count(a.kv_heads),
        Arg::count(a.width),
    ];
    let shape = k.shape().expect("a grid");
    vec![Launch { args, shape }]
}

/// Scaled query rows over kept keys, by slot and position.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "attend_rows",
    order = Order::PerConfig,
    sizes = self.0.sizes(),
    work = self.0.heads * self.0.queries * self.0.keys.most() * self.0.width,
    needs = group_needs(self.1),
    accepts = rows_accepts,
    emits = rows_emits,
    source = rows_source,
    launches = rows_launches,
    shape = Shape::grid((self.0.queries * self.0.heads, 1), self.1 as usize),
)]
pub(crate) struct AttendRows(pub Attn, pub u32);

fn rows_accepts(k: &AttendRows, input: usize, form: &dyn Stored) -> bool {
    match input {
        1 => form.name() == k.0.k_form.name(),
        2 => form.name() == k.0.v_form.name(),
        _ => !form.narrow(),
    }
}

fn rows_emits(k: &AttendRows, form: &dyn Stored) -> bool {
    form.name() == k.0.out_form.name()
}

fn rows_source(k: &AttendRows) -> Source {
    let (a, threads) = (&k.0, k.1);
    let mut raw = attend_indexed_ir::<true, false>(
        a.k_form,
        a.v_form,
        a.out_form,
        threads,
        threads.trailing_zeros(),
        threads / 2,
    );
    raw.params.retain(|p| p.name() != "bounds");
    let body = kept::attach(raw, a.kept);
    let tag = if threads == THREADS {
        String::new()
    } else {
        format!("_t{threads}")
    };
    let entry = match (a.k_form.name(), a.v_form.name(), a.out_form.name()) {
        ("f32", "f32", "f32") => format!("plan_attend_rows_{}{tag}", a.kept.name()),
        (kf, vf, of) => format!("plan_attend_rows_{}_{kf}_{vf}_{of}{tag}", a.kept.name()),
    };
    Source::named(body, entry)
}

/// One launch over every row.
fn rows_launches(k: &AttendRows, step: &Step) -> Vec<Launch> {
    let a = &k.0;
    let mut args = a.rows_args(step);
    args.extend([Arg::count(a.width), a.back()]);
    args.extend(a.kept_args());
    let shape = k.shape().expect("a grid");
    vec![Launch { args, shape }]
}

/// Reads scaled queries, and keys joined with values. Any mask.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    attend(self.0.out_form, self.1, self.1.trailing_zeros(), self.1 / 2),
    name = "attend",
    order = Order::PerConfig,
    sizes = self.0.sizes(),
    work = self.0.seqs * self.0.heads * self.0.queries * self.0.seen() * self.0.width,
    needs = group_needs(self.1),
    emits = attend_emits,
    source = attend_source,
    launches = attend_launches,
)]
pub(crate) struct Attend(pub Attn, pub u32);

fn attend_emits(k: &Attend, form: &dyn Stored) -> bool {
    form.name() == k.0.out_form.name()
}

fn attend_source(k: &Attend) -> Source {
    let t = k.1;
    let tag = if t == THREADS {
        String::new()
    } else {
        format!("_t{t}")
    };
    let entry = match k.0.out_form.name() {
        "f32" => format!("plan_attend{tag}"),
        of => format!("plan_attend_{of}{tag}"),
    };
    Source::named(attend_ir(k.0.out_form, t, t.trailing_zeros(), t / 2), entry)
}

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
    let launch = |s| a.launch(buffers(s), [Arg::count(a.width)], across, k.1 as usize);
    (0..a.seqs).map(launch).collect()
}

fn group_needs(threads: u32) -> Needs {
    Needs {
        group_bytes: 4 * threads as usize,
        group_threads: threads as usize,
        ..Needs::default()
    }
}

/// Scale the queries, join keys with values, then attend.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Attend = tree.site(at)?;
    let threads = tree
        .knobs
        .get("attend", "threads")
        .map_or(THREADS, |t| t as u32);
    if !threads.is_power_of_two() || site.sizes.width > 2 * threads as usize {
        return None;
    }
    if site.packed.is_some() {
        return site.packed_draft(AttendPacked(site.sizes));
    }
    // Ragged rows read the kept keys and values in place.
    if site.rows.is_some() {
        return site.ragged(AttendRows(site.sizes, threads));
    }
    let (scaled, out) = crate::scaled(&site)?;
    let sizes = site.sizes;
    let side = sizes.kv_heads * sizes.width;
    let join = Join {
        left: side,
        right: side,
        rows: sizes.keys.times(sizes.seqs),
        a_form: sizes.k_form,
        b_form: sizes.v_form,
        out_form: F32_FORM,
    };
    if !join.fits() {
        return None;
    }
    let joined = 2 * sizes.seqs * sizes.keys.most() * side;
    Some(vec![
        scaled,
        Draft::new(join, [site.k, site.v], joined),
        Draft::new(
            Attend(sizes, threads),
            [Read::Draft(0), Read::Draft(1)],
            out,
        ),
    ])
}

tessera_lower::rule!("attend", propose);

mod mla_metal;

tessera_lower::rule!("mla_metal", mla_metal::propose);
