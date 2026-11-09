//! DeepSeek V4.1 Flash: sparse MLA, mHC streams, Engram. Sketch.

use crate::core::*;
use crate::glm53::{AttnShape, Indexer, Mla};
use crate::library::{Moe, moe};
use crate::lsa::{Lsa, LsaShape, lsa_keys};

pub struct Config {
    pub layers: usize,
    pub top_k: usize,
    pub attn: AttnShape,
    /// Residual streams mixed by mHC.
    pub streams: usize,
    pub engram: EngramShape,
}

pub struct EngramShape {
    pub orders: Vec<usize>,
    pub cache: RowCache,
    /// Head candidates per sequence to prefetch for.
    pub prefetch_k: usize,
    pub prefetch_budget: Rows,
}

/// How a layer picks the keys its attention reads.
pub enum Selector {
    Indexer(Indexer),
    Lsa(Lsa, LsaShape),
    /// Reuses another layer's picks and cache.
    SameAs(Layer),
}

/// Hyper-connection mixing: read, write and stream-to-stream weights.
#[derive(Weights)]
pub struct Mhc {
    pub pre: Linear,
    pub post: Linear,
    pub res: Linear,
}

/// Hashed n-gram memory, gated into the stream.
#[derive(Weights)]
pub struct Engram {
    pub table: Embed,
    pub key: Linear,
    pub gate: Linear,
    pub norm: Norm,
}

#[derive(Weights)]
pub struct Block {
    pub attn: Mla,
    pub selector: Selector,
    pub attn_mhc: Mhc,
    pub engram: Option<Engram>,
    pub mlp_norm: Norm,
    pub moe: Moe<SwiGlu>,
    pub mlp_mhc: Mhc,
}

#[derive(Weights)]
pub struct Ds41 {
    #[split(vocab)]
    pub embed: Embed,
    pub blocks: Vec<Block>,
    pub norm: Norm,
    #[split(vocab)]
    pub head: Linear,
}

/// Vision graph output rows, scattered into the token rows.
pub struct Images {
    pub rows: Tensor,
    pub at: IndexSet,
}

pub fn forward(
    m: &mut Model,
    c: &Config,
    w: &Ds41,
    tokens: &Tensor,
    img: Option<Images>,
) -> Tensor {
    let mut x = m.embed(tokens, &w.embed).reduce();
    if let Some(i) = img {
        x = m.scatter_rows(&x, &i.rows, &i.at);
    }
    let mut s = x.expand_streams(c.streams);
    for (i, b) in w.blocks.iter().enumerate() {
        let l = Layer(i);
        if let Some(e) = &b.engram {
            s = s.add(&engram(m, c, e, tokens, &s));
        }
        let attn = Site::Writer(l, Sublayer::Attn);
        s = mhc(m, &b.attn_mhc, attn, &s, |m, h| attention(m, c, b, l, h));
        let mlp = Site::Writer(l, Sublayer::Mlp);
        s = mhc(m, &b.mlp_mhc, mlp, &s, |m, h| {
            moe(m, &b.moe, &h.rms_norm(&b.mlp_norm), c.top_k, l)
        });
        // The residual here is all streams, not one row.
        s = m.site(Site::Residual(l), s);
    }
    let h = m.site(Site::PreHead, s.merge_streams().rms_norm(&w.norm));
    h.matmul(&w.head)
}

/// Read one input from the streams, run `f`, mix back.
fn mhc(
    m: &mut Model,
    w: &Mhc,
    site: Site,
    s: &Tensor,
    f: impl FnOnce(&mut Model, &Tensor) -> Tensor,
) -> Tensor {
    let h = s.mix_streams(&w.pre);
    let out = m.site(site, f(m, &h).reduce());
    s.mix_streams(&w.res).add(&out.spread_streams(&w.post))
}

fn attention(m: &mut Model, c: &Config, b: &Block, l: Layer, x: &Tensor) -> Tensor {
    let w = &b.attn;
    let h = x.rms_norm(&w.norm);
    let q = h
        .matmul(&w.q_a)
        .rms_norm(&w.q_norm)
        .matmul(&w.q_b)
        .rope(c.attn.rope);
    let latent = h.matmul(&w.kv_a).rms_norm(&w.kv_norm);
    let kind = AttnKind::Mla {
        latent: c.attn.latent,
        rope: c.attn.rope,
    };
    let (cache, keys) = match &b.selector {
        Selector::SameAs(from) => (m.paged_kv_of(*from), m.keys_of(*from)),
        Selector::Indexer(i) => (
            m.paged_kv(l, kind, Sync::Eager),
            indexer_keys(m, c, i, &h, l),
        ),
        Selector::Lsa(i, shape) => (
            m.paged_kv(l, kind, Sync::Eager),
            lsa_keys(m, i, shape, &q, &h, l),
        ),
    };
    m.attend(kind, Mask::Causal, &q, &latent, &cache, Some(keys))
        .matmul(&w.o_proj)
}

fn indexer_keys(m: &mut Model, c: &Config, i: &Indexer, h: &Tensor, l: Layer) -> KeySet {
    let k = m.index_keys(l, &h.matmul(&i.wk));
    let scores = h.matmul(&i.wq).dot_all(&k).weighted(&h.matmul(&i.weights));
    m.top_k(&scores, c.attn.index_keep).keys()
}

/// Hash recent n-grams, gather table rows, gate them in.
fn engram(m: &mut Model, c: &Config, w: &Engram, tokens: &Tensor, s: &Tensor) -> Tensor {
    let ids = m.ngram_hash(tokens, &c.engram.orders);
    // Ids need only tokens, so the planner hoists this read.
    let mem = m.row_cache(&w.table, &c.engram.cache).fetch(&ids);
    let h = s.merge_streams().rms_norm(&w.norm);
    mem.matmul(&w.key)
        .gate(Act::Sigmoid, &h.matmul(&w.gate))
        .spread_streams_even()
}

/// Likely ids for the next round: drafts plus head candidates.
pub fn prefetch_next(m: &mut Model, w: &Ds41, c: &Config, logits: &Tensor, drafts: &Tensor) {
    let next = m.top_k(logits, c.engram.prefetch_k).ids.union(drafts);
    let ids = m.ngram_hash_with(&m.last_tokens_all(), &next, &c.engram.orders);
    for e in w.blocks.iter().filter_map(|b| b.engram.as_ref()) {
        let cache = m.row_cache(&e.table, &c.engram.cache);
        cache.prefetch(&ids, c.engram.prefetch_budget);
    }
}

/// At admission every id of a queued prompt is known.
pub fn prefetch_prompt(m: &mut Model, w: &Ds41, c: &Config, prompt: &Tensor) {
    let ids = m.ngram_hash(prompt, &c.engram.orders);
    for e in w.blocks.iter().filter_map(|b| b.engram.as_ref()) {
        m.row_cache(&e.table, &c.engram.cache).prefetch_all(&ids);
    }
}
