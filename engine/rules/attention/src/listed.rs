//! Masked attention over listed keys, or computed keys by row.

use tessera_codegen::prelude::*;
use tessera_graph::Op;
use tessera_lower::sites::Attend;
use tessera_lower::{Attn, Draft, Extent, Kernel, Launch, Read, Shape, Step, T, Term, Tree};

/// One thread per pick: its scaled score, or -inf.
#[kernel]
fn listed_scores(
    q: Buf<F32>,
    k: Buf<F32>,
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
) {
    if gid() >= len {
        return;
    }
    let p = gid() % picks;
    let head = (gid() / picks) % heads;
    let row = gid() / (picks * heads);
    // Whole sequences end together. Ragged rows carry slot and position.
    let mut first = row / queries * keys;
    let mut position = keys - queries + row % queries;
    if ragged == 1 {
        first = slots[row as u64] * keys;
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
        let k_at = ((first + j) * kv_heads + kv_head) as u64 * width as u64;
        let q_at = (row * heads + head) as u64 * width as u64;
        score = q.dot(q_at, k, k_at, width) * scale;
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
    v: Buf<F32>,
    ids: Buf<U32>,
    slots: Buf<U32>,
    mut out: BufMut<F32>,
    len: u32,
    queries: u32,
    keys: u32,
    heads: u32,
    kv_heads: u32,
    width: u32,
    picks: u32,
    flags: u32,
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
    let mut first = row / queries * keys;
    if ragged == 1 {
        first = slots[row as u64] * keys;
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
            let v_at = ((first + j) * kv_heads + kv_head) as u64 * width as u64;
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
    /// Queries per sequence. Ragged rows have one each.
    pub queries: usize,
    /// Each slot's keys, up to the launch's capacity.
    pub keys: Extent,
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
}

impl Sizes {
    /// One score per row, head and pick.
    fn scores(&self) -> usize {
        self.lists() * self.picks
    }

    /// One list of picks per row and head.
    fn lists(&self) -> usize {
        self.len / self.width
    }

    /// The launch of `step` over `len` threads, `scale` if given.
    fn launch(&self, step: &Step, len: usize, scale: Option<f32>) -> Vec<Launch> {
        let mut launch = Launch::plain(step, &[len], Shape::Line(len));
        if let Some(scale) = scale {
            let at = launch.args.len() - 1;
            launch.args.insert(at, tessera_lower::Arg::value(&scale));
        }
        launch.args.push(tessera_lower::Arg::count(self.queries));
        launch.args.push(tessera_lower::Arg::size(self.keys));
        let counts = [self.heads, self.kv_heads, self.width, self.picks];
        let (ragged, listed) = (usize::from(self.ragged), usize::from(self.from_list));
        let flags = match scale {
            Some(_) => vec![ragged, listed, self.window.0, self.window.1],
            None => vec![ragged | listed << 1],
        };
        let args = counts
            .into_iter()
            .chain(flags)
            .map(tessera_lower::Arg::count);
        launch.args.extend(args);
        vec![launch]
    }

    /// Scores, their peaks and totals, then the mix.
    fn drafts(self, [q, k, v, ids, slots, positions]: [T; 6]) -> Vec<Draft> {
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
}

/// Each listed key's score, for whole sequences or ragged rows.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    listed_scores,
    name = "listed_scores",
    sizes = [self.0.len, self.0.keys.most(), self.0.picks],
    work = self.0.scores() * self.0.width,
    launches = scores_launches,
)]
pub(crate) struct ListedScores(pub Sizes);

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
    listed_mix,
    name = "listed_mix",
    sizes = [self.0.len, self.0.keys.most(), self.0.picks],
    work = self.0.len * self.0.picks,
    launches = mix_launches,
)]
pub(crate) struct ListedMix(pub Sizes);

fn mix_launches(k: &ListedMix, step: &Step) -> Vec<Launch> {
    k.0.launch(step, k.0.len, None)
}

/// Ragged rows over computed keys, not kept rows.
fn rows(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Op(Op::Attend(a)) = &node.term else {
        return None;
    };
    let (q, k, v) = (node.args[0], node.args[1], node.args[2]);
    if !a.ragged || tree.appends(k) || tree.appends(v) {
        return None;
    }
    let wide = |t: T| &tree.node(t).shape;
    let (&[rows, heads, width], &[_, most, kv_heads, _]) = (&wide(q)[..], &wide(k)[..]) else {
        return None;
    };
    if wide(v).last() != Some(&width) {
        return None;
    }
    let [slots, positions] = [node.args[3], node.args[4]];
    let (ids, picks) = match node.args.get(5) {
        Some(&ids) => (ids, *wide(ids).last()?),
        None => (positions, most),
    };
    let sizes = Sizes {
        len: rows * heads * width,
        queries: 1,
        keys: tree.node(k).extents[1],
        heads,
        kv_heads,
        width,
        picks,
        ragged: true,
        from_list: a.picked,
        scale: a.scale,
        window: a.mask.window(most),
    };
    Some(sizes.drafts([q, k, v, ids, slots, positions]))
}

/// Sequences whose keys are all present, under any mask.
fn whole((site, ids, picks): (Attend, T, usize)) -> Option<Vec<Draft>> {
    let a: Attn = site.sizes;
    let keys = a.keys.most();
    let sizes = Sizes {
        len: a.seqs * a.queries * a.heads * a.width,
        queries: a.queries,
        keys: Extent::Fixed(keys),
        heads: a.heads,
        kv_heads: a.kv_heads,
        width: a.width,
        picks,
        ragged: false,
        from_list: true,
        scale: site.scale,
        window: a.mask.window(keys),
    };
    // Slots and positions go unread, so the list stands in.
    Some(sizes.drafts([site.q, site.k, site.v, ids, ids, ids]))
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    match Attend::listed(tree, at) {
        Some(found) => whole(crate::fixed_list(tree, found)?),
        None => rows(tree, at),
    }
}

tessera_lower::rule!("listed", propose);
