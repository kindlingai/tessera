//! Masked attention over listed keys, or computed keys by row.

use tessera_codegen::kept::{self, Dense, Layout};
use tessera_codegen::prelude::*;
use tessera_graph::{Op, Order};
use tessera_lower::form::{Load as Stored, plain, writable};
use tessera_lower::sites::Attend;
use tessera_lower::{
    Arg, Attn, Draft, Extent, Kernel, Launch, Needs, Read, Shape, Source, Step, T, Term, Tree,
};

/// Elements one wide load brings per lane.
const WIDE: u32 = 4;

/// SIMD groups sharing one query row and head.
const WARPS: u32 = 8;

/// Threads in one fused attention group.
const THREADS: u32 = WARPS * LANES as u32;

/// Widest head handled by the fused warp kernel.
const MAX_WIDTH: usize = 640;

/// Per-lane register slots for `MAX_WIDTH` floats.
const MAX_SLOTS: usize = MAX_WIDTH / LANES;

/// Unroll bound for per-lane wide-load loops.
const MAX_QUADS: u32 = (MAX_SLOTS / WIDE as usize) as u32;

/// Fused split-K attention: one group per query and head.
#[kernel]
fn listed_attend(
    #[form(q_form)] q: Buf<Dyn>,
    #[form(k_form)] k: Buf<Dyn>,
    #[form(v_form)] v: Buf<Dyn>,
    ids: Buf<U32>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    scale: f32,
    lists: u32,
    queries: u32,
    keys: u32,
    heads: u32,
    kv_heads: u32,
    picks: u32,
    ragged: u32,
    from_list: u32,
    before: u32,
    after: u32,
    #[build] width: u32,
    #[build] q_form: &'static dyn Stored,
    #[build] k_form: &'static dyn Stored,
    #[build] v_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    let list = gid() / THREADS;
    if list >= lists {
        return;
    }
    let head = list % heads;
    let row = list / heads;
    let warp = lid() / LANES as u32;
    let l = lane();
    let kv_head = head / (heads / kv_heads);
    let mut slot = row / queries;
    let mut position = keys - queries + row % queries;
    if ragged == 1 {
        slot = slots[row as u64];
        position = positions[row as u64];
    }
    let from = if position > before {
        position - before
    } else {
        0
    };
    let to = if position + after < keys {
        position + after + 1
    } else {
        keys
    };
    let q_at = list as u64 * width as u64;
    let quads = (width + LANES as u32 * WIDE - 1) / (LANES as u32 * WIDE);
    let mut qr = [0.0f32; MAX_SLOTS];
    let mut acc = [0.0f32; MAX_SLOTS];
    for s in (0..quads).unroll(MAX_QUADS) {
        let d = s * (LANES as u32 * WIDE) + l * WIDE;
        let b = (s * WIDE) as usize;
        if d < width {
            let qv = q.load4(q_at + d as u64);
            qr[b] = qv.x;
            qr[b + 1] = qv.y;
            qr[b + 2] = qv.z;
            qr[b + 3] = qv.w;
        }
        acc[b] = 0.0;
        acc[b + 1] = 0.0;
        acc[b + 2] = 0.0;
        acc[b + 3] = 0.0;
    }
    let mut peak = f32::NEG_INFINITY;
    let mut total = 0.0f32;
    let start = if from_list == 0 { from } else { 0 };
    let mut stop = if from_list == 0 && to < picks {
        to
    } else {
        picks
    };
    if from_list == 1 && picks > 0 {
        let base = row as u64 * picks as u64;
        if ids[base + (picks - 1) as u64] >= keys {
            let mut lo = 0u32;
            let mut hi = picks;
            for step in 0..16 {
                let mid = (lo + hi) / 2;
                if mid < hi {
                    if ids[base + mid as u64] < keys {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
            }
            stop = lo;
        }
    }
    for p in (start + warp..stop).by(WARPS) {
        let mut j = p;
        if from_list == 1 {
            j = ids[row as u64 * picks as u64 + p as u64];
        }
        if j >= from && j < to {
            let k_at = (kept_row(slot, j) * kv_heads + kv_head) as u64 * width as u64;
            let mut dot = 0.0f32;
            for s in (0..quads).unroll(MAX_QUADS) {
                let d = s * (LANES as u32 * WIDE) + l * WIDE;
                let b = (s * WIDE) as usize;
                if d < width {
                    let kv = k.load4(k_at + d as u64);
                    dot += qr[b] * kv.x + qr[b + 1] * kv.y + qr[b + 2] * kv.z + qr[b + 3] * kv.w;
                }
            }
            dot = sum32(dot);
            let score = dot * scale;
            let next = max(peak, score);
            let corr = if total > 0.0 { exp(peak - next) } else { 0.0 };
            let wt = exp(score - next);
            peak = next;
            total = total * corr + wt;
            for s in (0..quads).unroll(MAX_QUADS) {
                let d = s * (LANES as u32 * WIDE) + l * WIDE;
                let b = (s * WIDE) as usize;
                if d < width {
                    let vv = v.load4(k_at + d as u64);
                    acc[b] = acc[b] * corr + wt * vv.x;
                    acc[b + 1] = acc[b + 1] * corr + wt * vv.y;
                    acc[b + 2] = acc[b + 2] * corr + wt * vv.z;
                    acc[b + 3] = acc[b + 3] * corr + wt * vv.w;
                }
            }
        }
    }
    let mut m_s = shared::<f32>(WARPS);
    let mut l_s = shared::<f32>(WARPS);
    let mut o_s = shared::<f32>(WARPS * width);
    if l == 0 {
        m_s[warp] = peak;
        l_s[warp] = total;
    }
    for s in (0..quads).unroll(MAX_QUADS) {
        let d = s * (LANES as u32 * WIDE) + l * WIDE;
        let b = (s * WIDE) as usize;
        if d < width {
            let o = warp * width + d;
            o_s[o] = acc[b];
            o_s[o + 1] = acc[b + 1];
            o_s[o + 2] = acc[b + 2];
            o_s[o + 3] = acc[b + 3];
        }
    }
    barrier();
    let mut best = f32::NEG_INFINITY;
    for w in 0..WARPS {
        if l_s[w] > 0.0 {
            best = max(best, m_s[w]);
        }
    }
    let mut denom = 0.0f32;
    for w in 0..WARPS {
        if l_s[w] > 0.0 {
            denom += l_s[w] * exp(m_s[w] - best);
        }
    }
    let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
    for d in (lid()..width).by(THREADS) {
        let mut sum = 0.0f32;
        for w in 0..WARPS {
            if l_s[w] > 0.0 {
                sum += o_s[w * width + d] * exp(m_s[w] - best);
            }
        }
        out[q_at + d as u64] = sum * inv;
    }
}

/// One thread per pick: its scaled score, or -inf.
#[kernel]
fn listed_scores(
    #[form(q_form)] q: Buf<Dyn>,
    #[form(k_form)] k: Buf<Dyn>,
    ids: Buf<U32>,
    slots: Buf<U32>,
    positions: Buf<U32>,
    mut out: BufMut<F32>,
    scale: f32,
    len: u32,
    queries: u32,
    keys: u32,
    heads: u32,
    kv_heads: u32,
    width: u32,
    picks: u32,
    ragged: u32,
    from_list: u32,
    before: u32,
    after: u32,
    #[build] q_form: &'static dyn Stored,
    #[build] k_form: &'static dyn Stored,
) {
    if gid() >= len {
        return;
    }
    let p = gid() % picks;
    let head = (gid() / picks) % heads;
    let row = gid() / (picks * heads);
    // Whole sequences end together. Ragged rows carry slot and position.
    let mut slot = row / queries;
    let mut position = keys - queries + row % queries;
    if ragged == 1 {
        slot = slots[row as u64];
        position = positions[row as u64];
    }
    // Keys the mask lets this query see.
    let from = if position > before {
        position - before
    } else {
        0
    };
    let to = if position + after < keys {
        position + after + 1
    } else {
        keys
    };
    let mut j = p;
    if from_list == 1 {
        j = ids[row as u64 * picks as u64 + p as u64];
    }
    let mut score = f32::NEG_INFINITY;
    if j >= from && j < to {
        let kv_head = head / (heads / kv_heads);
        let k_at = (kept_row(slot, j) * kv_heads + kv_head) as u64 * width as u64;
        let q_at = (row * heads + head) as u64 * width as u64;
        let mut dot = 0.0f32;
        for d in 0..width {
            dot += q[q_at + d as u64] * k[k_at + d as u64];
        }
        score = dot * scale;
    }
    out[gid() as u64] = score;
}

/// One thread per list: peak, total, and kept picks.
#[kernel]
fn listed_stats(scores: Buf<F32>, mut out: BufMut<F32>, len: u32, picks: u32) {
    if gid() >= len {
        return;
    }
    let at = gid() as u64 * picks as u64;
    let mut peak = f32::NEG_INFINITY;
    let mut end = 0u32;
    for p in 0..picks {
        let score = scores[at + p as u64];
        peak = max(peak, score);
        if score > f32::NEG_INFINITY {
            end = p + 1;
        }
    }
    let mut total = 0.0f32;
    for p in 0..end {
        let score = scores[at + p as u64];
        if score > f32::NEG_INFINITY {
            total += exp(score - peak);
        }
    }
    let to = gid() as u64 * 3;
    out[to] = peak;
    out[to + 1] = total;
    out[to + 2] = end as f32;
}

/// One thread per output, summed up to the last pick.
#[kernel]
fn listed_mix(
    scores: Buf<F32>,
    stats: Buf<F32>,
    #[form(v_form)] v: Buf<Dyn>,
    ids: Buf<U32>,
    slots: Buf<U32>,
    #[form(out_form)] mut out: BufMut<Dyn>,
    len: u32,
    queries: u32,
    keys: u32,
    heads: u32,
    kv_heads: u32,
    width: u32,
    picks: u32,
    flags: u32,
    #[build] v_form: &'static dyn Stored,
    #[build] out_form: &'static dyn Stored,
) {
    if gid() >= len {
        return;
    }
    // Bit 0: ragged rows. Bit 1: picks from the list.
    let ragged = flags & 1;
    let from_list = flags / 2;
    let d = gid() % width;
    let head = (gid() / width) % heads;
    let row = gid() / (width * heads);
    let mut slot = row / queries;
    if ragged == 1 {
        slot = slots[row as u64];
    }
    let kv_head = head / (heads / kv_heads);
    let at = (row * heads + head) as u64 * picks as u64;
    let stat = (row * heads + head) as u64 * 3;
    let peak = stats[stat];
    let total = stats[stat + 1];
    let end = stats[stat + 2] as u32;
    let mut sum = 0.0f32;
    for p in 0..end {
        let score = scores[at + p as u64];
        if score > f32::NEG_INFINITY {
            let mut j = p;
            if from_list == 1 {
                j = ids[row as u64 * picks as u64 + p as u64];
            }
            let v_at = (kept_row(slot, j) * kv_heads + kv_head) as u64 * width as u64;
            sum += exp(score - peak) * v[v_at + d as u64];
        }
    }
    let mut o = 0.0f32;
    if total > 0.0 {
        o = sum / total;
    }
    out[gid() as u64] = o;
}

/// Query rows over `[Seqs, Keys, KvHeads, Width]` keys.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sizes {
    pub len: usize,
    pub seqs: usize,
    /// Queries per sequence. Ragged rows have one each.
    pub queries: usize,
    /// Each slot's keys, up to the launch's capacity.
    pub keys: Extent,
    /// Slot capacity in `k`/`v`, for `kept_row`.
    pub context: usize,
    pub kept: &'static dyn Layout,
    pub heads: usize,
    pub kv_heads: usize,
    pub width: usize,
    pub picks: usize,
    pub ragged: bool,
    /// Without a list, pick `p` is key `p`.
    pub from_list: bool,
    pub scale: f32,
    /// Keys a query sees before and after its position.
    pub window: (usize, usize),
    pub q_form: &'static dyn Stored,
    pub k_form: &'static dyn Stored,
    pub v_form: &'static dyn Stored,
    pub out_form: &'static dyn Stored,
}

impl Sizes {
    /// Suffix for narrow forms, empty when all are f32.
    fn form_suffix(forms: &[&'static dyn Stored]) -> String {
        if forms.iter().all(|f| !f.narrow()) {
            return String::new();
        }
        let mut out = String::new();
        for f in forms {
            out.push('_');
            out.push_str(f.name());
        }
        out
    }

    /// One score per row, head and pick.
    fn scores(&self) -> usize {
        self.lists() * self.picks
    }

    /// One list of picks per row and head.
    fn lists(&self) -> usize {
        self.len / self.width
    }

    /// Buffer arguments, plus any page table for `kept`.
    fn buffers(&self, step: &Step) -> Vec<Arg> {
        (0..step.reads.len())
            .map(|i| step.read(i))
            .chain([step.out()])
            .chain(step.kept(self.kept))
            .collect()
    }

    /// Trailing scalar arguments for `kept`.
    fn kept_scalars(&self) -> impl Iterator<Item = Arg> {
        self.kept
            .values(self.seqs, self.context)
            .into_iter()
            .map(|v| Arg::count(v as usize))
    }

    /// The launch of `step` over `len` threads, `scale` if given.
    fn launch(&self, step: &Step, len: usize, scale: Option<f32>) -> Vec<Launch> {
        let mut args = self.buffers(step);
        if let Some(scale) = scale {
            args.push(Arg::value(&scale));
        }
        args.push(Arg::count(len));
        args.push(Arg::count(self.queries));
        args.push(Arg::size(self.keys));
        let counts = [self.heads, self.kv_heads, self.width, self.picks];
        let (ragged, listed) = (usize::from(self.ragged), usize::from(self.from_list));
        let flags = match scale {
            Some(_) => vec![ragged, listed, self.window.0, self.window.1],
            None => vec![ragged | listed << 1],
        };
        args.extend(counts.into_iter().chain(flags).map(Arg::count));
        args.extend(self.kept_scalars());
        vec![Launch {
            args,
            shape: Shape::Line(len),
        }]
    }

    /// Three-pass listed attention for any backend.
    fn passes(self, [q, k, v, ids, slots, positions]: [T; 6]) -> Vec<Draft> {
        let mix = [
            Read::Draft(0),
            Read::Draft(1),
            Read::Node(v),
            Read::Node(ids),
            Read::Node(slots),
        ];
        vec![
            Draft::new(
                ListedScores(self),
                [q, k, ids, slots, positions],
                self.scores(),
            ),
            Draft::new(ListedStats(self), [Read::Draft(0)], self.lists() * 3),
            Draft::new(ListedMix(self), mix, self.len),
        ]
    }

    /// Checks whether `width` fits the single-launch split-K kernel.
    fn check_fused(&self) -> Result<(), String> {
        if !self.width.is_multiple_of(WIDE as usize) {
            return Err(format!(
                "head width {} is not a multiple of {WIDE}",
                self.width
            ));
        }
        if self.width > MAX_WIDTH {
            return Err(format!("head width {} exceeds max {MAX_WIDTH}", self.width));
        }
        Ok(())
    }

    /// Single-launch split-K attention when width fits.
    fn fused(self, reads: [T; 6]) -> Option<Vec<Draft>> {
        self.check_fused().ok()?;
        Some(vec![Draft::new(ListedAttend(self), reads, self.len)])
    }
}

/// Fused split-K attention over listed or ragged keys.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "listed_attend",
    sizes = [self.0.len, self.0.keys.most(), self.0.picks],
    work = self.0.scores() * self.0.width,
    needs = attend_needs(self),
    order = Order::PerConfig,
    accepts = attend_accepts,
    emits = attend_emits,
    source = attend_source,
    launches = attend_launches,
    shape = Shape::grid((self.0.lists(), 1), THREADS as usize),
)]
pub(crate) struct ListedAttend(pub Sizes);

fn attend_accepts(k: &ListedAttend, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.0.q_form.name(),
        1 => form.name() == k.0.k_form.name(),
        2 => form.name() == k.0.v_form.name(),
        _ => !form.narrow(),
    }
}

fn attend_emits(k: &ListedAttend, form: &dyn Stored) -> bool {
    form.name() == k.0.out_form.name()
}

fn attend_source(k: &ListedAttend) -> Source {
    let s = &k.0;
    let body = kept::attach(
        listed_attend_ir(s.width as u32, s.q_form, s.k_form, s.v_form, s.out_form),
        s.kept,
    );
    let suffix = Sizes::form_suffix(&[s.q_form, s.k_form, s.v_form, s.out_form]);
    Source::named(
        body,
        format!("plan_listed_attend_{}_{}{suffix}", s.kept.name(), s.width),
    )
}

fn attend_needs(k: &ListedAttend) -> Needs {
    Needs {
        lanes: LANES,
        group_bytes: 4 * (WARPS as usize * (k.0.width + 2)),
        group_threads: THREADS as usize,
        ..Needs::default()
    }
}

fn attend_launches(k: &ListedAttend, step: &Step) -> Vec<Launch> {
    let s = k.0;
    let (lists, shape) = (s.lists(), k.shape().expect("a grid"));
    let mut args = s.buffers(step);
    args.push(Arg::value(&s.scale));
    args.push(Arg::count(lists));
    args.push(Arg::count(s.queries));
    args.push(Arg::size(s.keys));
    let counts = [
        s.heads,
        s.kv_heads,
        s.picks,
        usize::from(s.ragged),
        usize::from(s.from_list),
        s.window.0,
        s.window.1,
    ];
    args.extend(counts.into_iter().map(Arg::count));
    args.extend(s.kept_scalars());
    vec![Launch { args, shape }]
}

/// Each listed key's score, for whole sequences or ragged rows.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "listed_scores",
    sizes = [self.0.len, self.0.keys.most(), self.0.picks],
    work = self.0.scores() * self.0.width,
    accepts = scores_accepts,
    source = scores_source,
    launches = scores_launches,
)]
pub(crate) struct ListedScores(pub Sizes);

fn scores_accepts(k: &ListedScores, input: usize, form: &dyn Stored) -> bool {
    match input {
        0 => form.name() == k.0.q_form.name(),
        1 => form.name() == k.0.k_form.name(),
        _ => !form.narrow(),
    }
}

fn scores_source(k: &ListedScores) -> Source {
    let s = &k.0;
    let body = kept::attach(listed_scores_ir(s.q_form, s.k_form), s.kept);
    let suffix = Sizes::form_suffix(&[s.q_form, s.k_form]);
    Source::named(
        body,
        format!("plan_listed_scores_{}{suffix}", s.kept.name()),
    )
}

fn scores_launches(k: &ListedScores, step: &Step) -> Vec<Launch> {
    k.0.launch(step, k.0.scores(), Some(k.0.scale))
}

/// Each row and head's peak score, total, and kept picks.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    listed_stats,
    name = "listed_stats",
    sizes = [self.0.len, self.0.picks],
    work = self.0.scores(),
    launches = stats_launches,
)]
pub(crate) struct ListedStats(pub Sizes);

fn stats_launches(k: &ListedStats, step: &Step) -> Vec<Launch> {
    let len = k.0.lists();
    vec![Launch::plain(step, &[len, k.0.picks], Shape::Line(len))]
}

/// The softmax of the scores over the listed values.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    name = "listed_mix",
    sizes = [self.0.len, self.0.keys.most(), self.0.picks],
    work = self.0.len * self.0.picks,
    accepts = mix_accepts,
    emits = mix_emits,
    source = mix_source,
    launches = mix_launches,
)]
pub(crate) struct ListedMix(pub Sizes);

fn mix_accepts(k: &ListedMix, input: usize, form: &dyn Stored) -> bool {
    match input {
        2 => form.name() == k.0.v_form.name(),
        _ => !form.narrow(),
    }
}

fn mix_emits(k: &ListedMix, form: &dyn Stored) -> bool {
    form.name() == k.0.out_form.name()
}

fn mix_source(k: &ListedMix) -> Source {
    let s = &k.0;
    let body = kept::attach(listed_mix_ir(s.v_form, s.out_form), s.kept);
    let suffix = Sizes::form_suffix(&[s.v_form, s.out_form]);
    Source::named(body, format!("plan_listed_mix_{}{suffix}", s.kept.name()))
}

fn mix_launches(k: &ListedMix, step: &Step) -> Vec<Launch> {
    k.0.launch(step, k.0.len, None)
}

/// Ragged rows over kept or computed keys.
fn rows(tree: &Tree, at: T) -> Option<(Sizes, [T; 6])> {
    let node = tree.node(at);
    let Term::Op(Op::Attend(a)) = &node.term else {
        return None;
    };
    let (q, k, v) = (node.args[0], node.args[1], node.args[2]);
    if !a.ragged {
        return None;
    }
    let wide = |t: T| &tree.node(t).shape;
    let (&[rows, heads, width], &[seqs, most, kv_heads, _]) = (&wide(q)[..], &wide(k)[..]) else {
        return None;
    };
    if wide(v).last() != Some(&width) {
        return None;
    }
    let (kept, context): (&'static dyn Layout, usize) =
        match (&tree.node(k).term, &tree.node(v).term) {
            (Term::Op(Op::Append(ki)), Term::Op(Op::Append(vi))) => {
                let (_, ctx) = tree.kept_rows(*ki)?;
                if tree.kept_rows(*vi)?.1 != ctx {
                    return None;
                }
                (tree.kept, ctx)
            }
            (Term::Op(Op::Append(_)), _) | (_, Term::Op(Op::Append(_))) => return None,
            _ => (&Dense, most),
        };
    let [slots, positions] = [node.args[3], node.args[4]];
    let (ids, picks) = match node.args.get(5) {
        Some(&ids) => (ids, *wide(ids).last()?),
        None => (positions, most),
    };
    let sizes = Sizes {
        len: rows * heads * width,
        seqs,
        queries: 1,
        keys: tree.node(k).extents[1],
        context,
        kept,
        heads,
        kv_heads,
        width,
        picks,
        ragged: true,
        from_list: a.picked,
        scale: a.scale,
        window: a.mask.window(most),
        q_form: plain(tree.node(q).form),
        k_form: plain(tree.node(k).form),
        v_form: plain(tree.node(v).form),
        out_form: writable(node.form),
    };
    Some((sizes, [q, k, v, ids, slots, positions]))
}

/// Sequences whose keys are all present, under any mask.
fn whole((site, ids, picks): (Attend, T, usize)) -> (Sizes, [T; 6]) {
    let a: Attn = site.sizes;
    let keys = a.keys.most();
    let sizes = Sizes {
        len: a.seqs * a.queries * a.heads * a.width,
        seqs: a.seqs,
        queries: a.queries,
        keys: Extent::Fixed(keys),
        context: keys,
        kept: &Dense,
        heads: a.heads,
        kv_heads: a.kv_heads,
        width: a.width,
        picks,
        ragged: false,
        from_list: true,
        scale: site.scale,
        window: a.mask.window(keys),
        q_form: site.q_form,
        k_form: a.k_form,
        v_form: a.v_form,
        out_form: a.out_form,
    };
    // Slots and positions go unread, so the list stands in.
    (sizes, [site.q, site.k, site.v, ids, ids, ids])
}

fn site(tree: &Tree, at: T) -> Option<(Sizes, [T; 6])> {
    match Attend::listed(tree, at) {
        Some(found) => Some(whole(crate::fixed_list(tree, found)?)),
        None => rows(tree, at),
    }
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (sizes, reads) = site(tree, at)?;
    Some(sizes.passes(reads))
}

fn propose_fused(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let (sizes, reads) = site(tree, at)?;
    sizes.fused(reads)
}

fn why_fused(tree: &Tree, at: T) -> Option<String> {
    let (sizes, _) = site(tree, at)?;
    sizes.check_fused().err()
}

tessera_lower::rule!("listed", propose);
tessera_lower::rule!("listed_fused", propose_fused, why = why_fused);
