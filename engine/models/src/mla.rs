//! DeepSeek-style decoders: low-rank attention, MoE, indexed keys.

use crate::shared::{Attend, Stack, Steps, TapSink, pick, rows, step_ports, swiglu, tokens};
use tessera_trace::DimName::*;
use tessera_trace::{
    Act, Composite, DType, Dim, DimExpr, DimName, Embed, Graph, Kept, Linear, Linears, Mask, Model,
    Norm, Pairs, Role, Size, Tensor, TensorType, Vector, Weights,
};

pub struct Config {
    /// Its `ffn` is an expert's width. Values use `head_dim`.
    pub stack: Stack,
    /// Leading layers with a dense MLP.
    pub dense_layers: u64,
    /// A head's unrotated key width, and its rotated width.
    pub nope: u64,
    pub rope: u64,
    pub rope_theta: f32,
    pub q_latent: u64,
    pub latent: u64,
    pub dense_ffn: u64,
    pub experts: u64,
    pub top_k: u64,
    pub route_scale: f32,
    /// Some layers score keys first. None read them all.
    pub index: Option<Index>,
    pub eps: f32,
}

/// The indexer's shape, and which layers run one.
pub struct Index {
    pub heads: u64,
    pub dim: u64,
    pub keep: u64,
    /// Past the dense layers, every n-th layer indexes.
    pub every: u64,
}

impl Config {
    /// GLM-5.3 at its released sizes.
    pub fn glm53(max_seqs: u64, max_context: u64) -> Self {
        let sizes = [154_880, 6144, 78, 64, 256, 2048];
        let index = Index {
            heads: 32,
            dim: 128,
            keep: 2048,
            every: 4,
        };
        Self {
            stack: Stack::sized(sizes, max_seqs, max_context),
            dense_layers: 3,
            nope: 192,
            rope: 64,
            rope_theta: 8e6,
            q_latent: 2048,
            latent: 512,
            dense_ffn: 12_288,
            experts: 256,
            top_k: 8,
            route_scale: 2.5,
            index: Some(index),
            eps: 1e-5,
        }
    }

    /// DeepSeek-V3 at its released sizes. Experts are picked without groups.
    pub fn deepseek_v3(max_seqs: u64, max_context: u64) -> Self {
        let sizes = [129_280, 7168, 61, 128, 128, 2048];
        Self {
            stack: Stack::sized(sizes, max_seqs, max_context),
            dense_layers: 3,
            nope: 128,
            rope: 64,
            rope_theta: 1e4,
            q_latent: 1536,
            latent: 512,
            dense_ffn: 18_432,
            experts: 256,
            top_k: 8,
            route_scale: 2.5,
            index: None,
            eps: 1e-6,
        }
    }

    /// Whether layer `l` picks keys. Others reuse the last pick.
    pub fn indexes(&self, l: u64) -> bool {
        let Some(index) = &self.index else {
            return false;
        };
        let dense = self.dense_layers;
        l < dense || (l + 1 - dense).is_multiple_of(index.every)
    }

    pub fn dims(&self) -> Vec<Dim> {
        let layers = self.stack.layers;
        let indexing = (0..layers).filter(|&l| self.indexes(l)).count();
        let mut own = vec![
            (Nope, self.nope),
            (Rope, self.rope),
            (QLatent, self.q_latent),
            (Latent, self.latent),
            (DenseFfn, self.dense_ffn),
            (Experts, self.experts),
            (TopK, self.top_k),
            (DenseLayers, self.dense_layers),
            (MoeLayers, layers - self.dense_layers),
            (IndexLayers, indexing as u64),
        ];
        if let Some(i) = &self.index {
            own.extend([
                (IndexHeads, i.heads),
                (IndexDim, i.dim),
                (IndexKeep, i.keep),
            ]);
        }
        self.stack.dims(&own)
    }
}

#[derive(Weights)]
pub struct Attention {
    #[shape(Hidden)]
    pub norm: Norm,
    #[shape(Hidden, QLatent)]
    pub q_a: Linear,
    #[shape(QLatent)]
    pub q_norm: Norm,
    #[shape(QLatent, Heads * (Nope + Rope))]
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
pub struct Mla<R: Routed> {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(Layers)]
    pub attn: Vec<Attention>,
    /// One per indexing layer, in layer order.
    #[shape(IndexLayers)]
    pub indexers: Vec<Indexer>,
    #[shape(Layers, Hidden)]
    pub mlp_norm: Vec<Norm>,
    #[shape(DenseLayers)]
    pub dense: Vec<DenseMlp>,
    #[shape(MoeLayers)]
    pub moe: Vec<R>,
    #[shape(Hidden)]
    pub norm: Norm,
    #[shape(Hidden, Vocab)]
    pub head: Linear,
}

/// One layer's kept latent, rotary key and index keys.
struct Cache {
    latent: Kept,
    rope: Kept,
    index: Option<Kept>,
}

/// The keys a layer reads.
enum Keys<'a> {
    All,
    /// Keys an earlier layer picked.
    Given(Tensor),
    /// Keys this indexer picks.
    Pick(&'a Indexer),
}

/// State one pass over the layers shares.
struct Pass<'a> {
    m: &'a Model,
    c: &'a Config,
    positions: &'a Tensor,
    /// When stepping: each row's slot, and every layer's kept rows.
    step: Option<(&'a Tensor, Vec<Cache>)>,
}

pub fn forward<R: Routed>(
    m: &Model,
    c: &Config,
    w: &Mla<R>,
    tokens: &Tensor,
    positions: &Tensor,
) -> Tensor {
    let pass = Pass {
        m,
        c,
        positions,
        step: None,
    };
    pass.logits(w, tokens)
}

impl Pass<'_> {
    fn logits<R: Routed>(&self, w: &Mla<R>, tokens: &Tensor) -> Tensor {
        self.logits_taps(w, tokens, &[]).0
    }

    /// Logits, and joined tapped layers when requested.
    fn logits_taps<R: Routed>(
        &self,
        w: &Mla<R>,
        tokens: &Tensor,
        layers: &[u64],
    ) -> (Tensor, Option<Tensor>) {
        if layers.is_empty() {
            let [logits] = self.m.composite(Composite::Model, || {
                let x = self.trunk(w, tokens, |_, _| {});
                [x.rms_norm(&w.norm, self.c.eps).matmul(&w.head)]
            });
            return (logits, None);
        }
        let [logits, tapped] = self.m.composite(Composite::Model, || {
            let mut sink = TapSink::new(layers);
            let x = self.trunk(w, tokens, |at, row| sink.record(at, row));
            let logits = x.rms_norm(&w.norm, self.c.eps).matmul(&w.head);
            [logits, sink.join().expect("tap layers")]
        });
        (logits, Some(tapped))
    }

    /// The residual rows. `tap` sees each layer's, counting from 1.
    fn trunk<R: Routed>(
        &self,
        w: &Mla<R>,
        tokens: &Tensor,
        mut tap: impl FnMut(u64, &Tensor),
    ) -> Tensor {
        let (m, c) = (self.m, self.c);
        let mut x = m.embed(tokens, &w.embed);
        let mut indexers = w.indexers.iter();
        let mut keys: Option<Tensor> = None;
        for l in 0..w.attn.len() {
            let reused = keys.clone().filter(|_| !c.indexes(l as u64));
            let read = match (reused, &c.index) {
                (Some(k), _) => Keys::Given(k),
                (None, None) => Keys::All,
                (None, Some(_)) => {
                    Keys::Pick(indexers.next().expect("an indexer per indexing layer"))
                }
            };
            let picks = matches!(read, Keys::Pick(_));
            let layer = || self.layer(w, l, &x, read);
            let (out, picked) = self.region(Composite::Layer, picks, layer);
            x = out;
            keys = picked.or(keys);
            tap(l as u64 + 1, &x);
        }
        x
    }

    /// One decoder layer, and the keys its attention picked.
    fn layer<R: Routed>(
        &self,
        w: &Mla<R>,
        l: usize,
        x: &Tensor,
        read: Keys,
    ) -> (Tensor, Option<Tensor>) {
        let (m, c, a) = (self.m, self.c, &w.attn[l]);
        let h = x.rms_norm(&a.norm, c.eps);
        let picks = matches!(read, Keys::Pick(_));
        let attend = || self.mla(l, a, &h, read);
        let (attn, keys) = self.region(Composite::Attention, picks, attend);
        let x = x.add(&attn);
        let h = x.rms_norm(&w.mlp_norm[l], c.eps);
        let [mlp] = match l.checked_sub(w.dense.len()) {
            None => {
                let d = &w.dense[l];
                m.composite(Composite::Mlp, || [swiglu(&h, [&d.gate, &d.up, &d.down])])
            }
            Some(i) => m.composite(Composite::Moe, || [w.moe[i].apply(m, c, &h)]),
        };
        (x.add(&mlp), keys)
    }

    /// `body` as a `kind` region. Picked keys are output two.
    fn region(
        &self,
        kind: Composite,
        picks: bool,
        body: impl FnOnce() -> (Tensor, Option<Tensor>),
    ) -> (Tensor, Option<Tensor>) {
        if !picks {
            let [out] = self.m.composite(kind, || [body().0]);
            return (out, None);
        }
        let [out, keys] = self.m.composite(kind, || {
            let (out, keys) = body();
            [out, keys.expect("picked keys")]
        });
        (out, Some(keys))
    }

    /// The axes before a row's features: `[Seqs, Context]`, or `[Rows]`.
    fn lead(&self, rest: impl IntoIterator<Item = DimExpr>) -> Vec<DimExpr> {
        let lead = match self.step {
            Some(_) => vec![Rows.into()],
            None => rows([]),
        };
        lead.into_iter().chain(rest).collect()
    }

    /// The output, and the keys it picked, if any.
    fn mla(&self, l: usize, w: &Attention, h: &Tensor, read: Keys) -> (Tensor, Option<Tensor>) {
        let (m, c) = (self.m, self.c);
        let feature = self.lead([]).len() as u32;
        let shape = [Heads.into(), Heads.into(), HeadDim.into()];
        let heads = Attend::new(&w.o, self.lead([]), shape);
        let turn = |t: Tensor| t.rope(self.positions, c.rope_theta, Pairs::Adjacent);
        let qr = h.matmul(&w.q_a).rms_norm(&w.q_norm, c.eps);
        let keys = match read {
            Keys::All => None,
            Keys::Given(k) => Some(k),
            Keys::Pick(ix) => Some(self.pick_keys(l, ix, &qr, h)),
        };
        let q = heads.split(qr.matmul(&w.q_b), Heads, Nope + Rope);
        let head = feature + 1;
        let (nope, rope) = (q.slice(head, 0u64, Nope), turn(q.slice(head, Nope, Rope)));
        let kv = h.matmul(&w.kv_a);
        let latent = kv.slice(feature, 0u64, Latent).rms_norm(&w.kv_norm, c.eps);
        // Every head shares one rotary key part.
        let shared = turn(kv.slice(feature, Latent, Rope));
        let scale = 1.0 / ((c.nope + c.rope) as f32).sqrt();
        let out = match &self.step {
            None => {
                let q = nope.concat(&rope);
                let [k, v] = keys_values(w, &latent, &shared);
                m.attend(Mask::Causal, scale, [&q, &k, &v], keys.as_ref())
            }
            Some((slots, caches)) => {
                let by = [*slots, self.positions];
                let latent = latent.append(&caches[l].latent, by);
                let k = latent.concat(&shared.append(&caches[l].rope, by));
                absorbed(m, w, [&nope, &rope], &k, (scale, by), keys.as_ref())
            }
        };
        (heads.merge(out), keys)
    }

    /// Each query's best `IndexKeep` visible keys.
    fn pick_keys(&self, l: usize, w: &Indexer, qr: &Tensor, h: &Tensor) -> Tensor {
        let c = self.c;
        let q = qr
            .matmul(&w.wq)
            .reshape(self.lead([IndexHeads.into(), IndexDim.into()]));
        let k = h.matmul(&w.wk).center().rms_norm(&w.k_norm, c.eps);
        let k = k.add_vector(&w.k_bias);
        // A step reads its slot's kept keys.
        let k = match &self.step {
            None => k,
            Some((slots, caches)) => {
                let kept = caches[l].index.as_ref().expect("an index cache");
                k.append(kept, [*slots, self.positions]).gather(slots)
            }
        };
        // Per query: one score for each head and key.
        let hits = q.matmul_rows(&k).act(Act::Relu);
        let by_head = h
            .matmul(&w.weights)
            .reshape(self.lead([DimExpr::Const(1), IndexHeads.into()]));
        let rank = self.lead([]).len() as u32 + 2;
        let swap: Vec<u32> = (0..rank - 2).chain([rank - 1, rank - 2]).collect();
        let scores = hits.permute(&swap).matmul_rows(&by_head);
        let scores = scores.reshape(self.lead([Context.into()]));
        let seen = match self.step {
            Some(_) => scores.causal(self.positions),
            None => scores.mask(Mask::Causal),
        };
        seen.top_k(IndexKeep)
    }
}

/// Per-head keys and values from latents and rotary keys.
fn keys_values(w: &Attention, latent: &Tensor, shared: &Tensor) -> [Tensor; 2] {
    let per_head = latent
        .matmul(&w.kv_b)
        .reshape(rows([Heads.into(), Nope + HeadDim]));
    let one = rows([DimExpr::Const(1), Rope.into()]);
    let shared = shared.reshape(one).expand(2, Heads);
    let k = per_head.slice(3, 0u64, Nope).concat(&shared);
    [k, per_head.slice(3, Nope, HeadDim)]
}

/// Attention over latent keys, with `kv_b` folded into each side.
fn absorbed(
    m: &Model,
    w: &Attention,
    [nope, rope]: [&Tensor; 2],
    keys: &Tensor,
    (scale, by): (f32, [&Tensor; 2]),
    picked: Option<&Tensor>,
) -> Tensor {
    // Per head: key rows, then value rows, one view each.
    let set = || {
        w.kv_b
            .stored()
            .reshape(vec![Heads.into(), Nope + HeadDim, Latent.into()])
    };
    let swap = [1, 0, 2];
    let to_latent = set().slice(1, 0u64, Nope).permute(&[0, 2, 1]);
    let q = nope.permute(&swap).matmul_rows(&to_latent).permute(&swap);
    let q = q.concat(rope);
    // One key head serves every query head.
    let k = keys.reshape(rows([DimExpr::Const(1), Latent + Rope]));
    let out = m.attend_rows(Mask::Causal, scale, [&q, &k, &k], by, picked);
    let out = out.slice(2, 0u64, Latent).permute(&swap);
    out.matmul_rows(&set().slice(1, Nope, HeadDim))
        .permute(&swap)
}

/// Routed experts and a shared one, however they are stored.
pub trait Routed: Weights {
    fn apply(&self, m: &Model, c: &Config, h: &Tensor) -> Tensor;
}

/// Each row's top experts, and their weights.
fn route(c: &Config, router: &Linear, bias: &Vector, h: &Tensor) -> [Tensor; 2] {
    let scores = h.matmul(router).act(Act::Sigmoid);
    let ids = scores.add_vector(bias).top_k(TopK);
    let gates = scores.take(&ids).unit_sum().scale(c.route_scale);
    [ids, gates]
}

impl Routed for Moe {
    /// Each row's top experts, weighted, plus the shared expert.
    fn apply(&self, m: &Model, c: &Config, h: &Tensor) -> Tensor {
        let [ids, gates] = route(c, &self.router, &self.router_bias, h);
        // One output row per pick, each on its own expert.
        let on = |set: &Linears, x: &Tensor| x.matmul_rows(&m.pick(set, &ids));
        let inner = on(&self.gate, h).act(Act::Silu).mul(&on(&self.up, h));
        let picks = h.ty().rank() as u32 - 1;
        let routed = on(&self.down, &inner).mix(picks, &gates);
        let s = &self.shared;
        routed.add(&swiglu(h, [&s.gate, &s.up, &s.down]))
    }
}

/// Experts as codes between 128-channel rotations.
#[derive(Weights)]
pub struct RotatedMoe {
    #[shape(Hidden, Experts)]
    pub router: Linear,
    #[shape(Experts)]
    pub router_bias: Vector,
    /// Signs on the stream into the experts, and out.
    #[shape(Hidden)]
    pub signs_in: Vector,
    #[shape(Hidden)]
    pub signs_out: Vector,
    #[shape(Experts, Hidden, Ffn)]
    pub gate: Linears,
    #[shape(Experts, Hidden)]
    pub gate_su: Embed,
    #[shape(Experts, Ffn)]
    pub gate_sv: Embed,
    #[shape(Experts, Hidden, Ffn)]
    pub up: Linears,
    #[shape(Experts, Hidden)]
    pub up_su: Embed,
    #[shape(Experts, Ffn)]
    pub up_sv: Embed,
    #[shape(Experts, Ffn, Hidden)]
    pub down: Linears,
    #[shape(Experts, Ffn)]
    pub down_su: Embed,
    #[shape(Experts, Hidden)]
    pub down_sv: Embed,
    pub shared: SharedMlp,
}

impl Routed for RotatedMoe {
    fn apply(&self, m: &Model, c: &Config, h: &Tensor) -> Tensor {
        let [ids, gates] = route(c, &self.router, &self.router_bias, h);
        let picks = h.ty().rank() as u32 - 1;
        // One copy of each signed row per picked expert.
        let lead: Vec<DimExpr> = h.ty().shape.0[..picks as usize].to_vec();
        let one = lead.into_iter().chain([DimExpr::Const(1), Hidden.into()]);
        let slots = h
            .mul_vector(&self.signs_in)
            .reshape(one.collect::<Vec<_>>());
        let slots = slots.expand(picks, TopK);
        let on = |x: &Tensor, codes: &Linears, [su, sv]: [&Embed; 2]| {
            let turned = x.mul(&m.embed(&ids, su)).hadamard(128);
            let product = turned.matmul_rows(&m.pick(codes, &ids));
            product.hadamard(128).mul(&m.embed(&ids, sv))
        };
        let gate = on(&slots, &self.gate, [&self.gate_su, &self.gate_sv]);
        let up = on(&slots, &self.up, [&self.up_su, &self.up_sv]);
        let inner = gate.act(Act::Silu).mul(&up);
        let out = on(&inner, &self.down, [&self.down_su, &self.down_sv]);
        let routed = out.mul_vector(&self.signs_out).mix(picks, &gates);
        let s = &self.shared;
        routed.add(&swiglu(h, [&s.gate, &s.up, &s.down]))
    }
}

/// One forward entry: tokens and positions in, logits out.
pub fn program<R: Routed>(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Mla<R> = tessera_trace::declare(&m);
    let [tokens, positions] = tokens(&m);
    let logits = forward(&m, c, &w, &tokens, &positions);
    m.finish([(Role::Logits, &logits)])
}

/// One step entry: rows in, each row's likeliest token out.
pub fn stepper<R: Routed>(c: &Config) -> Graph {
    let mut dims = c.dims();
    let size = Size::Bounded(c.stack.max_context);
    dims.push(Dim { name: Rows, size });
    let m = Model::new(dims);
    let w: Mla<R> = tessera_trace::declare(&m);
    let ([tokens, positions, slots], allowed) = step_ports(&m, c.stack.steps == Steps::Masked);
    let kept = |width: DimName| m.kept(TensorType::new(rows([width.into()]), DType::F32));
    let cache = |l| Cache {
        latent: kept(Latent),
        rope: kept(Rope),
        index: c.indexes(l).then(|| kept(IndexDim)),
    };
    let pass = Pass {
        m: &m,
        c,
        positions: &positions,
        step: Some((&slots, (0..c.stack.layers).map(cache).collect())),
    };
    let (logits, taps) = pass.logits_taps(&w, &tokens, &c.stack.taps);
    pick(&m, &logits, allowed.as_ref(), taps.as_ref())
}
