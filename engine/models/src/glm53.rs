//! GLM-5.3: low-rank attention over indexed keys, and MoE.

use crate::shared::{Stack, rows, swiglu, tokens};
use trace::DimName::*;
use trace::{
    Act, Composite, Dim, DimExpr, Embed, Graph, Linear, Linears, Mask, Model, Norm, Pairs, Role,
    Tensor, Vector, Weights,
};

pub struct Config {
    /// Its `ffn` is an expert's width. Values use `head_dim`.
    pub stack: Stack,
    /// Leading layers with a dense MLP.
    pub dense_layers: u64,
    /// The part of `head_dim` that is rotated.
    pub rope: u64,
    pub rope_theta: f32,
    pub q_latent: u64,
    pub latent: u64,
    pub dense_ffn: u64,
    pub experts: u64,
    pub top_k: u64,
    pub route_scale: f32,
    pub index_heads: u64,
    pub index_dim: u64,
    pub index_keep: u64,
    /// Past the dense layers, every n-th layer indexes.
    pub index_every: u64,
    pub eps: f32,
}

impl Config {
    /// The released checkpoint's sizes.
    pub fn release(max_seqs: u64, max_context: u64) -> Self {
        let stack = Stack {
            vocab: 154_880,
            hidden: 6144,
            layers: 78,
            heads: 64,
            head_dim: 256,
            ffn: 2048,
            max_seqs,
            max_context,
        };
        Self {
            stack,
            dense_layers: 3,
            rope: 64,
            rope_theta: 8e6,
            q_latent: 2048,
            latent: 512,
            dense_ffn: 12_288,
            experts: 256,
            top_k: 8,
            route_scale: 2.5,
            index_heads: 32,
            index_dim: 128,
            index_keep: 2048,
            index_every: 4,
            eps: 1e-5,
        }
    }

    /// Whether layer `l` picks keys. Others reuse the last pick.
    pub fn indexes(&self, l: u64) -> bool {
        let dense = self.dense_layers;
        l < dense || (l + 1 - dense).is_multiple_of(self.index_every)
    }

    pub fn dims(&self) -> Vec<Dim> {
        let layers = self.stack.layers;
        let indexing = (0..layers).filter(|&l| self.indexes(l)).count();
        self.stack.dims(&[
            (Nope, self.stack.head_dim - self.rope),
            (Rope, self.rope),
            (QLatent, self.q_latent),
            (Latent, self.latent),
            (DenseFfn, self.dense_ffn),
            (Experts, self.experts),
            (TopK, self.top_k),
            (IndexHeads, self.index_heads),
            (IndexDim, self.index_dim),
            (IndexKeep, self.index_keep),
            (DenseLayers, self.dense_layers),
            (MoeLayers, layers - self.dense_layers),
            (IndexLayers, indexing as u64),
        ])
    }
}

#[derive(Weights)]
pub struct Mla {
    #[shape(Hidden)]
    pub norm: Norm,
    #[shape(Hidden, QLatent)]
    pub q_a: Linear,
    #[shape(QLatent)]
    pub q_norm: Norm,
    #[shape(QLatent, Heads * HeadDim)]
    pub q_b: Linear,
    #[shape(Hidden, Latent + Rope)]
    pub kv_a: Linear,
    #[shape(Latent)]
    pub kv_norm: Norm,
    /// Per head: an unrotated key part, then a value.
    #[shape(Latent, Heads * (Nope + HeadDim))]
    pub kv_b: Linear,
    #[shape(Heads * HeadDim, Hidden)]
    pub o: Linear,
}

/// Scores every key cheaply, so attention reads only the best.
#[derive(Weights)]
pub struct Indexer {
    #[shape(QLatent, IndexHeads * IndexDim)]
    pub wq: Linear,
    #[shape(Hidden, IndexDim)]
    pub wk: Linear,
    #[shape(IndexDim)]
    pub k_norm: Norm,
    #[shape(IndexDim)]
    pub k_bias: Vector,
    #[shape(Hidden, IndexHeads)]
    pub weights: Linear,
}

#[derive(Weights)]
pub struct DenseMlp {
    #[shape(Hidden, DenseFfn)]
    pub gate: Linear,
    #[shape(Hidden, DenseFfn)]
    pub up: Linear,
    #[shape(DenseFfn, Hidden)]
    pub down: Linear,
}

/// The expert every row uses.
#[derive(Weights)]
pub struct SharedMlp {
    #[shape(Hidden, Ffn)]
    pub gate: Linear,
    #[shape(Hidden, Ffn)]
    pub up: Linear,
    #[shape(Ffn, Hidden)]
    pub down: Linear,
}

#[derive(Weights)]
pub struct Moe {
    #[shape(Hidden, Experts)]
    pub router: Linear,
    /// Added to scores for the pick only.
    #[shape(Experts)]
    pub router_bias: Vector,
    #[shape(Experts, Hidden, Ffn)]
    pub gate: Linears,
    #[shape(Experts, Hidden, Ffn)]
    pub up: Linears,
    #[shape(Experts, Ffn, Hidden)]
    pub down: Linears,
    pub shared: SharedMlp,
}

/// Layer `l` uses `dense[l]`, or `moe[l - DenseLayers]`.
#[derive(Weights)]
pub struct Glm53 {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(Layers)]
    pub attn: Vec<Mla>,
    /// One per indexing layer, in layer order.
    #[shape(IndexLayers)]
    pub indexers: Vec<Indexer>,
    #[shape(Layers, Hidden)]
    pub mlp_norm: Vec<Norm>,
    #[shape(DenseLayers)]
    pub dense: Vec<DenseMlp>,
    #[shape(MoeLayers)]
    pub moe: Vec<Moe>,
    #[shape(Hidden)]
    pub norm: Norm,
    #[shape(Hidden, Vocab)]
    pub head: Linear,
}

pub fn forward(m: &Model, c: &Config, w: &Glm53, tokens: &Tensor, positions: &Tensor) -> Tensor {
    let mut x = m.embed(tokens, &w.embed);
    let mut indexers = w.indexers.iter();
    let mut keys: Option<Tensor> = None;
    for (l, a) in w.attn.iter().enumerate() {
        let h = x.rms_norm(&a.norm, c.eps);
        let at = [&h, positions];
        let reused = keys.clone().filter(|_| !c.indexes(l as u64));
        let attn = match reused {
            Some(k) => m.composite(Composite::Attention, || [mla(m, c, a, at, |_| k).0]),
            None => {
                let ix = indexers.next().expect("an indexer per indexing layer");
                let pick = |qr: &Tensor| pick_keys(c, ix, qr, &h);
                let [out, k] = m.composite(Composite::Attention, || mla(m, c, a, at, pick).into());
                keys = Some(k);
                [out]
            }
        };
        x = x.add(&attn[0]);
        let h = x.rms_norm(&w.mlp_norm[l], c.eps);
        let mlp = match l.checked_sub(w.dense.len()) {
            None => {
                let d = &w.dense[l];
                m.composite(Composite::Mlp, || [swiglu(&h, [&d.gate, &d.up, &d.down])])
            }
            Some(i) => m.composite(Composite::Moe, || [moe(m, c, &w.moe[i], &h)]),
        };
        x = x.add(&mlp[0]);
    }
    x.rms_norm(&w.norm, c.eps).matmul(&w.head)
}

/// Returns the output and the keys each query read.
fn mla(
    m: &Model,
    c: &Config,
    w: &Mla,
    [h, positions]: [&Tensor; 2],
    keys: impl FnOnce(&Tensor) -> Tensor,
) -> (Tensor, Tensor) {
    let turn = |t: Tensor| t.rope(positions, c.rope_theta, Pairs::Adjacent);
    let qr = h.matmul(&w.q_a).rms_norm(&w.q_norm, c.eps);
    let keys = keys(&qr);
    let q = qr.matmul(&w.q_b).reshape([Seqs, Context, Heads, HeadDim]);
    let q = q.slice(3, 0u64, Nope).concat(&turn(q.slice(3, Nope, Rope)));
    let kv = h.matmul(&w.kv_a);
    let latent = kv.slice(2, 0u64, Latent).rms_norm(&w.kv_norm, c.eps);
    // Every head shares one rotary key part.
    let shared = turn(kv.slice(2, Latent, Rope))
        .reshape(rows([DimExpr::Const(1), Rope.into()]))
        .expand(2, Heads);
    let per_head = latent
        .matmul(&w.kv_b)
        .reshape(rows([Heads.into(), Nope + HeadDim]));
    let k = per_head.slice(3, 0u64, Nope).concat(&shared);
    let v = per_head.slice(3, Nope, HeadDim);
    let scale = 1.0 / (c.stack.head_dim as f32).sqrt();
    let out = m.attend(Mask::Causal, scale, [&q, &k, &v], Some(&keys));
    let out = out.reshape(rows([Heads * HeadDim])).matmul(&w.o);
    (out, keys)
}

/// Each query's best `IndexKeep` visible keys.
fn pick_keys(c: &Config, w: &Indexer, qr: &Tensor, h: &Tensor) -> Tensor {
    let q = qr
        .matmul(&w.wq)
        .reshape([Seqs, Context, IndexHeads, IndexDim]);
    let k = h.matmul(&w.wk).center().rms_norm(&w.k_norm, c.eps);
    // Per query: one score for each head and key.
    let hits = q.matmul_rows(&k.add_vector(&w.k_bias)).act(Act::Relu);
    let by_head = h
        .matmul(&w.weights)
        .reshape(rows([DimExpr::Const(1), IndexHeads.into()]));
    let scores = hits.permute(&[0, 1, 3, 2]).matmul_rows(&by_head);
    let scores = scores.reshape([Seqs, Context, Context]);
    scores.mask(Mask::Causal).top_k(IndexKeep)
}

/// Each row's top experts, weighted, plus the shared expert.
fn moe(m: &Model, c: &Config, w: &Moe, h: &Tensor) -> Tensor {
    let scores = h.matmul(&w.router).act(Act::Sigmoid);
    let ids = scores.add_vector(&w.router_bias).top_k(TopK);
    let gates = scores.take(&ids).unit_sum().scale(c.route_scale);
    // One copy of each row per picked expert.
    let slots = h
        .reshape(rows([DimExpr::Const(1), Hidden.into()]))
        .expand(2, TopK);
    let on = |set: &Linears, x: &Tensor| x.matmul_rows(&m.pick(set, &ids));
    let inner = on(&w.gate, &slots).act(Act::Silu).mul(&on(&w.up, &slots));
    let outs = on(&w.down, &inner).permute(&[0, 1, 3, 2]);
    let gates = gates.reshape(rows([DimExpr::Const(1), TopK.into()]));
    let routed = outs.matmul_rows(&gates).reshape([Seqs, Context, Hidden]);
    let s = &w.shared;
    routed.add(&swiglu(h, [&s.gate, &s.up, &s.down]))
}

/// One forward entry: tokens and positions in, logits out.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Glm53 = trace::declare(&m);
    let [tokens, positions] = tokens(&m);
    let logits = forward(&m, c, &w, &tokens, &positions);
    m.finish([(Role::Logits, &logits)])
}
