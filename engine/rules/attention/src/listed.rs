//! Masked attention over listed keys, or computed keys by row.

use sensai_codegen::prelude::*;
use sensai_graph::Op;
use sensai_lower::sites::Attend;
use sensai_lower::{Attn, Draft, Extent, Kernel, Launch, Shape, Step, T, Term, Tree};

/// One thread per output: the peak, then the weighted sum.
#[kernel]
fn listed(
    q: Buf<F32>,
    k: Buf<F32>,
    v: Buf<F32>,
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
    let d = gid() % width;
    let head = (gid() / width) % heads;
    let row = gid() / (width * heads);
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
    let kv_head = head / (heads / kv_heads);
    let list = row as u64 * picks as u64;
    let q_at = (gid() - d) as u64;
    let mut peak = f32::NEG_INFINITY;
    for p in 0..picks {
        let mut j = p;
        if from_list == 1 {
            j = ids[list + p as u64];
        }
        if j >= from && j < to {
            let k_at = ((first + j) * kv_heads + kv_head) as u64 * width as u64;
            peak = max(peak, q.dot(q_at, k, k_at, width) * scale);
        }
    }
    let mut total = 0.0f32;
    let mut sum = 0.0f32;
    for p in 0..picks {
        let mut j = p;
        if from_list == 1 {
            j = ids[list + p as u64];
        }
        if j >= from && j < to {
            let k_at = ((first + j) * kv_heads + kv_head) as u64 * width as u64;
            let e = exp(q.dot(q_at, k, k_at, width) * scale - peak);
            total += e;
            sum += e * v[k_at + d as u64];
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
    /// Each slot's keys, as many as the launch holds.
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

/// Whole sequences over listed keys, or ragged rows by slot.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    listed,
    name = "listed",
    sizes = [self.0.len, self.0.keys.most(), self.0.picks],
    work = self.0.len * self.0.picks * self.0.width,
    launches = listed_launches,
)]
pub(crate) struct Listed(pub Sizes);

fn listed_launches(k: &Listed, step: &Step) -> Vec<Launch> {
    let s = &k.0;
    let flags = [
        usize::from(s.ragged),
        usize::from(s.from_list),
        s.window.0,
        s.window.1,
    ];
    let counts = [s.heads, s.kv_heads, s.width, s.picks];
    let mut launch = Launch::plain(step, &[s.len], Shape::Line(s.len));
    launch
        .args
        .insert(launch.args.len() - 1, sensai_lower::Arg::value(&s.scale));
    launch.args.push(sensai_lower::Arg::count(s.queries));
    launch.args.push(sensai_lower::Arg::size(s.keys));
    launch.args.extend(
        counts
            .into_iter()
            .chain(flags)
            .map(sensai_lower::Arg::count),
    );
    vec![launch]
}

/// Ragged rows over computed keys, not kept rows.
fn rows(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let node = tree.node(at);
    let Term::Op(Op::Attend(a)) = &node.term else {
        return None;
    };
    let (q, k, v) = (node.args[0], node.args[1], node.args[2]);
    let kept = |t: T| matches!(tree.node(t).term, Term::Op(Op::Append(_)));
    if !a.ragged || kept(k) || kept(v) {
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
        window: (most, 0),
    };
    let reads = [q, k, v, ids, slots, positions];
    Some(vec![Draft::new(Listed(sizes), reads, sizes.len)])
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
        window: crate::window(a.mask, keys),
    };
    // Slots and positions go unread, so the list stands in.
    let reads = [site.q, site.k, site.v, ids, ids, ids];
    Some(vec![Draft::new(Listed(sizes), reads, sizes.len)])
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    match Attend::listed(tree, at) {
        Some(found) => whole(crate::fixed_list(tree, found)?),
        None => rows(tree, at),
    }
}

sensai_lower::rule!("listed", propose);
