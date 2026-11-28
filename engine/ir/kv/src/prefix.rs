//! Which retained run a prompt continues.

use crate::{Pool, RunId, Tag};
use std::collections::HashMap;

struct Entry {
    run: RunId,
    history: Vec<u32>,
    usable: usize,
}

/// Edge storage that avoids `HashMap` allocations on low-fanout nodes.
enum Edges {
    Small(Vec<(u64, usize)>),
    Many(HashMap<u64, Vec<usize>>),
}

impl Edges {
    fn new() -> Self {
        Self::Small(Vec::new())
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Small(v) => v.is_empty(),
            Self::Many(m) => m.is_empty(),
        }
    }

    fn find(&self, hash: u64, mut matches: impl FnMut(usize) -> bool) -> Option<usize> {
        match self {
            Self::Small(v) => v
                .iter()
                .find_map(|&(h, cid)| (h == hash && matches(cid)).then_some(cid)),
            Self::Many(m) => m.get(&hash)?.iter().copied().find(|&cid| matches(cid)),
        }
    }

    fn push(&mut self, hash: u64, cid: usize) {
        if let Self::Small(v) = self {
            if v.len() < 8 {
                v.push((hash, cid));
                return;
            }
            let mut map: HashMap<u64, Vec<usize>> = HashMap::with_capacity(v.len() + 1);
            for (h, c) in v.drain(..) {
                map.entry(h).or_default().push(c);
            }
            map.entry(hash).or_default().push(cid);
            *self = Self::Many(map);
            return;
        }
        if let Self::Many(m) = self {
            m.entry(hash).or_default().push(cid);
        }
    }

    fn replace(&mut self, hash: u64, old_cid: usize, new_cid: usize) {
        match self {
            Self::Small(v) => {
                for (h, cid) in v.iter_mut() {
                    if *h == hash && *cid == old_cid {
                        *cid = new_cid;
                        return;
                    }
                }
            }
            Self::Many(m) => {
                if let Some(bucket) = m.get_mut(&hash)
                    && let Some(slot) = bucket.iter_mut().find(|c| **c == old_cid)
                {
                    *slot = new_cid;
                }
            }
        }
    }

    fn for_each(&self, mut f: impl FnMut(usize)) {
        match self {
            Self::Small(v) => {
                for &(_, cid) in v {
                    f(cid);
                }
            }
            Self::Many(m) => {
                for bucket in m.values() {
                    for &cid in bucket {
                        f(cid);
                    }
                }
            }
        }
    }
}

/// One compressed edge in the chunk prefix trie.
struct Node {
    owner: usize,
    start: usize,
    end: usize,
    max_idx: usize,
    edges: Edges,
    ends: Vec<usize>,
}

/// The tokens of retained runs, indexed by chunk prefixes.
pub struct Prefixes {
    grid: usize,
    kept: Vec<Entry>,
    nodes: Vec<Node>,
    path: Vec<(usize, usize)>,
    stack: Vec<usize>,
    probe: usize,
}

/// Fast 64-bit hash of one `grid`-token chunk.
fn hash_chunk(tokens: &[u32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let (pairs, rem) = tokens.as_chunks::<2>();
    for &[a, b] in pairs {
        h ^= u64::from(a) | (u64::from(b) << 32);
        h = h.wrapping_mul(0x100000001b3);
    }
    if let [last] = rem {
        h ^= u64::from(*last);
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^ (h >> 33)
}

/// Common chunk-aligned token count between `a` and `b`.
fn common_chunks(a: &[u32], b: &[u32], step: usize) -> usize {
    let limit = (a.len().min(b.len()) / step) * step;
    if a[..limit] == b[..limit] {
        return limit;
    }
    let mut off = 0usize;
    while off < limit && a[off..off + step] == b[off..off + step] {
        off += step;
    }
    off
}

fn empty_node(owner: usize, start: usize, end: usize, max_idx: usize) -> Node {
    Node {
        owner,
        start,
        end,
        max_idx,
        edges: Edges::new(),
        ends: Vec::new(),
    }
}

impl Prefixes {
    /// Skips come in whole `grid` steps.
    pub fn new(grid: usize) -> Self {
        Self {
            grid,
            kept: Vec::new(),
            nodes: vec![empty_node(0, 0, 0, 0)],
            path: Vec::new(),
            stack: Vec::new(),
            probe: 0,
        }
    }

    /// Records the tokens behind a retained run's `valid` rows.
    pub fn keep(&mut self, run: RunId, history: Vec<u32>, valid: usize) {
        let step = self.grid.max(1);
        let usable = (history.len().min(valid) / step) * step;
        let idx = self.kept.len();
        self.kept.push(Entry {
            run,
            history,
            usable,
        });
        if usable > 0 {
            self.insert(idx);
        }
    }

    /// The newest run with the longest skip.
    pub fn find(
        &mut self,
        pool: &Pool,
        tag: Tag,
        prompt: &[u32],
        limit: usize,
    ) -> Option<(RunId, usize)> {
        if self.kept.is_empty() {
            return None;
        }
        let mut stale = false;
        // Sample one retained run per query to reclaim quiet evictions.
        let check = self.probe % self.kept.len();
        self.probe = self.probe.wrapping_add(1);
        if pool.rows(self.kept[check].run).is_err() {
            self.prune(pool);
            if self.kept.is_empty() {
                return None;
            }
        }
        let step = self.grid.max(1);
        let cap = (prompt.len().min(limit) / step) * step;
        if cap == 0 {
            return None;
        }
        self.path.clear();
        let (mut cur, mut off) = (0usize, 0usize);
        while off < cap {
            let chunk = &prompt[off..off + step];
            let Some(cid) = self.child(cur, off, chunk, step) else {
                break;
            };
            let node = &self.nodes[cid];
            let edge = &self.kept[node.owner].history[off..node.end];
            let matched = off + common_chunks(edge, &prompt[off..cap], step);
            self.path.push((cid, matched));
            if matched < node.end || matched == cap {
                break;
            }
            cur = cid;
            off = matched;
        }
        let mut skip_child = None;
        let mut found = None;
        for i in (0..self.path.len()).rev() {
            let (nid, matched) = self.path[i];
            if let Some(idx) = self.newest_in(nid, skip_child, pool, tag, &mut stale) {
                found = Some((self.kept[idx].run, matched));
                break;
            }
            skip_child = Some(nid);
        }
        if stale {
            self.prune(pool);
        }
        found
    }

    /// Drops evicted runs and rebuilds the compressed trie.
    fn prune(&mut self, pool: &Pool) {
        self.kept.retain(|e| pool.rows(e.run).is_ok());
        self.nodes.clear();
        self.nodes.push(empty_node(0, 0, 0, 0));
        self.probe = 0;
        for idx in 0..self.kept.len() {
            if self.kept[idx].usable > 0 {
                self.insert(idx);
            }
        }
    }

    /// Child of `nid` whose leading `step` tokens match `chunk`.
    fn child(&self, nid: usize, off: usize, chunk: &[u32], step: usize) -> Option<usize> {
        let h = hash_chunk(chunk);
        self.nodes[nid].edges.find(h, |cid| {
            &self.kept[self.nodes[cid].owner].history[off..off + step] == chunk
        })
    }

    /// Inserts `idx` into the compressed chunk trie.
    fn insert(&mut self, idx: usize) {
        let step = self.grid.max(1);
        let usable = self.kept[idx].usable;
        let (mut cur, mut off) = (0usize, 0usize);
        loop {
            self.nodes[cur].max_idx = self.nodes[cur].max_idx.max(idx);
            if off == usable {
                self.nodes[cur].ends.push(idx);
                return;
            }
            let chunk_hash = hash_chunk(&self.kept[idx].history[off..off + step]);
            let found = self.nodes[cur].edges.find(chunk_hash, |cid| {
                let owner = self.nodes[cid].owner;
                self.kept[owner].history[off..off + step] == self.kept[idx].history[off..off + step]
            });
            let Some(cid) = found else {
                let leaf = self.nodes.len();
                let mut node = empty_node(idx, off, usable, idx);
                node.ends.push(idx);
                self.nodes.push(node);
                self.nodes[cur].edges.push(chunk_hash, leaf);
                return;
            };
            let (owner, end, old_max) = (
                self.nodes[cid].owner,
                self.nodes[cid].end,
                self.nodes[cid].max_idx,
            );
            let common = common_chunks(
                &self.kept[owner].history[off..end],
                &self.kept[idx].history[off..usable],
                step,
            );
            let split = off + common;
            if split == end {
                cur = cid;
                off = split;
                continue;
            }
            let mid = self.nodes.len();
            let old_hash = hash_chunk(&self.kept[owner].history[split..split + step]);
            let mut mid_node = empty_node(owner, off, split, old_max.max(idx));
            mid_node.edges.push(old_hash, cid);
            self.nodes[cid].start = split;
            if split == usable {
                mid_node.ends.push(idx);
                self.nodes.push(mid_node);
            } else {
                let leaf = mid + 1;
                let new_hash = hash_chunk(&self.kept[idx].history[split..split + step]);
                mid_node.edges.push(new_hash, leaf);
                let mut leaf_node = empty_node(idx, split, usable, idx);
                leaf_node.ends.push(idx);
                self.nodes.push(mid_node);
                self.nodes.push(leaf_node);
            }
            self.nodes[cur].edges.replace(chunk_hash, cid, mid);
            return;
        }
    }

    /// Newest entry in `root` matching `tag`, skipping `exclude`.
    fn newest_in(
        &mut self,
        root: usize,
        exclude: Option<usize>,
        pool: &Pool,
        tag: Tag,
        stale: &mut bool,
    ) -> Option<usize> {
        self.stack.clear();
        self.stack.push(root);
        let mut best: Option<usize> = None;
        while let Some(nid) = self.stack.pop() {
            let node = &self.nodes[nid];
            if best.is_some_and(|b| node.max_idx <= b) {
                continue;
            }
            for &idx in node.ends.iter().rev() {
                if best.is_some_and(|b| idx <= b) {
                    break;
                }
                match pool.tag(self.kept[idx].run) {
                    Ok(t) if t == tag => {
                        best = Some(idx);
                        break;
                    }
                    Ok(_) => {}
                    Err(_) => *stale = true,
                }
            }
            if node.edges.is_empty() || best.is_some_and(|b| node.max_idx <= b) {
                continue;
            }
            let start = self.stack.len();
            node.edges.for_each(|cid| {
                if Some(cid) != exclude && best.is_none_or(|b| self.nodes[cid].max_idx > b) {
                    self.stack.push(cid);
                }
            });
            // Visit newer subtrees first so `best` prunes older branches.
            if self.stack.len() - start > 1 {
                self.stack[start..].sort_unstable_by_key(|&cid| self.nodes[cid].max_idx);
            }
        }
        best
    }
}
