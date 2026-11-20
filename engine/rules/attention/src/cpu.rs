//! Attention on the CPU, a thread per query and head.

use sensai_codegen::kept;
use sensai_codegen::prelude::*;
use sensai_lower::{Arg, Attn, Draft, Kernel, Launch, Step, T, Tree, ZoneKind, sites};
use sensai_lower::{GUESS, small, source};

/// One pass per query, over its keys or listed `ids`.
#[kernel]
fn cpu_attend<const LISTED: bool>(
    q: Buf<F32>,
    k: Buf<F32>,
    v: Buf<F32>,
    ids: Buf<U32>,
    mut out: BufMut<F32>,
    scale: f32,
    jobs: u32,
    #[build] queries: u32,
    #[build] keys: u32,
    #[build] heads: u32,
    #[build] kv_heads: u32,
    #[build] width: u32,
    #[build] before: u32,
    #[build] after: u32,
    #[build] picks: u32,
) {
    if gid() >= jobs {
        return;
    }
    let t = gid() % queries;
    let sh = gid() / queries;
    let s = sh / heads;
    let h = sh % heads;
    let qo = ((s * queries + t) * heads + h) as u64 * width as u64;
    // Query and key ends align, as in decode.
    let position = keys - queries + t;
    let from = if position > before {
        position - before
    } else {
        0
    };
    let reach = position + 1 + after;
    let to = if reach < keys { reach } else { keys };
    let kh = h / (heads / kv_heads);
    let list = (s * queries + t) as u64 * picks as u64;
    // Key 0 of this head. Keys step by `stride`.
    let first = (s * keys * kv_heads + kh) as u64 * width as u64;
    let stride = kv_heads * width;
    let listed = if LISTED { picks } else { 0 };
    out.attend(
        qo, q, k, v, first, stride, from, to, width, scale, ids, list, listed,
    );
}

/// Softmax attention over `[seqs, rows, heads, width]` tensors.
#[derive(Clone, Debug, Kernel)]
#[launch(
    crate = sensai_lower,
    name = "cpu_attend",
    sizes = self.sizes.sizes(),
    work = self.sizes.seqs * self.sizes.heads * self.sizes.queries * self.sizes.keys.most() * self.sizes.width,
    guess = GUESS,
    runs_on = ZoneKind::Cpu,
    source = source("cpu_attend", self.body()),
    launches = attend_launches,
)]
pub(crate) struct Attend {
    pub scale: f32,
    pub sizes: Attn,
    /// Keys listed per query, or zero for all.
    pub picks: usize,
}

impl Attend {
    /// Keys before and after a query's position that it sees.
    fn window(&self) -> (u32, u32) {
        let (before, after) = self.sizes.mask.window(self.sizes.keys.most());
        (before as u32, after as u32)
    }

    fn jobs(&self) -> usize {
        self.sizes.seqs * self.sizes.heads * self.sizes.queries
    }

    fn body(&self) -> sensai_codegen::ir::Body {
        let a = &self.sizes;
        let (before, after) = self.window();
        let sizes = (a.queries as u32, a.keys.most() as u32, a.heads as u32);
        let (queries, keys, heads) = sizes;
        let (kv, width, picks) = (a.kv_heads as u32, a.width as u32, self.picks as u32);
        match self.picks {
            0 => cpu_attend_ir::<false>(queries, keys, heads, kv, width, before, after, 0),
            _ => cpu_attend_ir::<true>(queries, keys, heads, kv, width, before, after, picks),
        }
    }
}

fn attend_launches(k: &Attend, step: &Step) -> Vec<Launch> {
    let mut launches = vec![Launch::plain(step, &[k.jobs()], small(k.jobs()))];
    // The scale sits before the counts.
    let at = launches[0].args.len() - 1;
    launches[0].args.insert(at, Arg::value(&k.scale));
    launches
}

/// One launch over every row, then the kept layout sizes.
#[kernel]
fn cpu_attend_rows(
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
    if gid() >= rows * heads {
        return;
    }
    let row = gid() / heads;
    let qh = gid() % heads;
    let slot = slots[row as u64];
    let position = positions[row as u64];
    let kvh = qh / (heads / kv_heads);
    let q_off = ((row * heads + qh) * width) as u64;
    let mut most = -1.0e30f32;
    let mut total = 0.0f32;
    // The row sees its slot up to its own position.
    let last = if position + 1 < keys {
        position + 1
    } else {
        keys
    };
    for t in 0..last {
        let kv_off = ((kept_row(slot, t) * kv_heads + kvh) * width) as u64;
        let score = q.dot(q_off, k, kv_off, width);
        let top = max(most, score);
        let keep = exp(most - top);
        let p = exp(score - top);
        total = total * keep + p;
        out.mix(q_off, keep, p, v, kv_off, width);
        most = top;
    }
    out.mix(q_off, 1.0 / total, 0.0, v, 0, width);
}

/// Scaled query rows over kept keys, by slot and position.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = sensai_lower,
    name = "cpu_attend_rows",
    sizes = self.0.sizes(),
    work = self.0.heads * self.0.queries * self.0.keys.most() * self.0.width,
    guess = GUESS,
    runs_on = ZoneKind::Cpu,
    source = source("cpu_attend_rows", kept::attach(cpu_attend_rows_ir(), self.0.kept)),
    launches = rows_launches,
)]
pub(crate) struct AttendRows(pub Attn);

/// One launch over every row, the kept layout's sizes last.
fn rows_launches(k: &AttendRows, step: &Step) -> Vec<Launch> {
    let a = &k.0;
    let mut args = a.rows_args(step);
    args.push(Arg::count(a.width));
    args.extend(a.kept_args());
    vec![Launch {
        args,
        shape: small(a.queries * a.heads),
    }]
}

/// Whole sequences: all keys, or the `ids` each query lists.
fn whole(site: &sites::Attend, ids: T, picks: usize) -> Vec<Draft> {
    let out = site.sizes.seqs * site.out();
    let kernel = Attend {
        scale: site.scale,
        sizes: site.sizes,
        picks,
    };
    vec![Draft::new(kernel, [site.q, site.k, site.v, ids], out)]
}

/// Whole sequences with fixed keys, or rows over kept keys.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Attend = tree.site(at)?;
    if site.rows.is_some() {
        return site.ragged(AttendRows(site.sizes));
    }
    if site.sizes.keys.runtime() {
        return None;
    }
    // Only the listed kernel reads its fourth buffer.
    Some(whole(&site, site.k, 0))
}

/// Whole sequences, each query reading its listed keys.
fn listed(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (site, ids, picks) = crate::fixed_list(tree, sites::Attend::listed(tree, at)?)?;
    Some(whole(&site, ids, picks))
}

sensai_lower::rule!("cpu_attend", propose);
sensai_lower::rule!("cpu_listed", listed);
