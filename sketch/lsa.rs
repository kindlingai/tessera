//! Latent sparse attention's key selector (arXiv 2610.42147). Sketch.

use crate::core::*;

/// Distilled projections into the index space; the base stays frozen.
#[derive(Weights)]
pub struct Lsa {
    pub wq: Linear,
    pub wk: Linear,
    pub tau: Scalar,
}

pub struct LsaShape {
    pub dim: usize,
    pub pool: usize,
    pub block: usize,
    pub bucket: usize,
    pub buckets_kept: usize,
    pub keys: usize,
    pub sinks: usize,
    pub window: usize,
}

/// Keys by bucket centroids, plus sinks and a recent window.
pub fn lsa_keys(m: &mut Model, w: &Lsa, s: &LsaShape, q: &Tensor, h: &Tensor, l: Layer) -> KeySet {
    // It sums all heads, so split heads make it Partial.
    let qi = q.sum_heads().matmul(&w.wq).reduce().l2_norm();
    let ki = h.matmul(&w.wk).l2_norm().pool_rows(s.pool);
    let index = m.search_index(l, index_shape(s), Rows(s.window));
    index.append(&ki);
    let scores = qi.dot_all(&index.centroids()).scale_by(&w.tau);
    let best = m.top_k(&scores, s.buckets_kept);
    let sizes = m.prefix(&index.bucket_sizes(&best), PrefixOp::Sum);
    let far = m.gather_until(&index.positions(&best), &sizes, s.keys);
    far.union(&m.sinks(s.sinks)).union(&m.recent(s.window))
}

fn index_shape(s: &LsaShape) -> Index {
    Index {
        method: ClusterMethod::Ward,
        pool: s.pool,
        block: s.block,
        bucket: s.bucket,
    }
}

/// Clusters finished blocks off the step path, before the deadline.
pub fn maintain(m: &mut Model, index: &SearchIndex) {
    for block in index.ready_blocks() {
        m.cluster(index, block);
    }
}
