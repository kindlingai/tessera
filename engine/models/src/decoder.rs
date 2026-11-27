//! Decoders: grouped-query attention, RoPE, a dense or routed MLP.

use crate::keys::{At, Keys};
pub use crate::shared::Swiglu;
use crate::shared::{
    Attend, Dense, ExpertSet, PlainExperts, Side, Stack, Steps, TapSink, with_head,
};
use serde_json::json;
use tessera_trace::DimName::*;
use tessera_trace::hf::*;
use tessera_trace::{
    Act, Composite, DType, Dim, DimExpr, DimName, Embed, Graph, Kept, Linear, Linears, Mask, Model,
    Norm, Pairs, Role, Size, Tensor, TensorType, Vector, Weights, listed, pick, sparse_ports,
    step_ports, tokens,
};

/// Steps a query or key head passes through before RoPE.
pub trait HeadNorm: Weights {
    fn apply(&self, x: Tensor, eps: f32) -> Tensor;
}

/// Qwen3 norms each head.
impl HeadNorm for Norm {
    fn apply(&self, x: Tensor, eps: f32) -> Tensor {
        x.rms_norm(self, eps)
    }
}

/// Llama does not.
impl HeadNorm for () {
    fn apply(&self, x: Tensor, _: f32) -> Tensor {
        x
    }
}

/// Optional additive bias on Q, K, and V projections.
pub trait QkvBias: Weights {
    fn add(&self, x: Tensor) -> Tensor;
}

/// Qwen2 adds a bias vector to Q, K, and V.
impl QkvBias for Vector {
    fn add(&self, x: Tensor) -> Tensor {
        x.add_vector(self)
    }
}

/// Llama, Qwen3, and Granite omit Q/K/V bias.
impl QkvBias for () {
    fn add(&self, x: Tensor) -> Tensor {
        x
    }
}

/// The second half of a layer, over the normed rows.
pub trait Mlp: Weights {
    const KIND: Composite;
    fn apply(&self, m: &Model, h: &Tensor) -> Tensor;
}

impl<D: Dense> Mlp for Swiglu<D> {
    const KIND: Composite = Composite::Mlp;

    fn apply(&self, _: &Model, h: &Tensor) -> Tensor {
        self.eval(h)
    }
}

/// Each row's `TopK` of `Experts` SwiGLU MLPs, softmax-weighted.
#[derive(Weights)]
pub struct Moe {
    #[shape(Hidden, Experts)]
    pub router: Linear,
    /// Each expert's gate rows, then its up rows.
    #[shape(Experts, Hidden, Ffn * 2)]
    pub input: Linears,
    #[shape(Experts, Ffn, Hidden)]
    pub output: Linears,
}

impl Mlp for Moe {
    const KIND: Composite = Composite::Moe;

    fn apply(&self, m: &Model, h: &Tensor) -> Tensor {
        let scores = h.matmul(&self.router);
        let ids = scores.top_k(TopK);
        let gates = scores.take(&ids).softmax();
        // One output row per pick, each on its own expert.
        let on = |set: &Linears, x: &Tensor| x.matmul_rows(&m.pick(set, &ids));
        let gate = on(&self.input.rows(0u64, Ffn), h).act(Act::Silu);
        let inner = gate.mul(&on(&self.input.rows(Ffn, Ffn), h));
        let picks = h.ty().rank() as u32 - 1;
        on(&self.output, &inner).mix(picks, &gates)
    }
}

/// Each row's `TopK` of `Experts` SwiGLU MLPs with separate matrices.
#[derive(Weights)]
pub struct SparseMoe {
    #[shape(Hidden, Experts)]
    pub router: Linear,
    pub experts: PlainExperts,
}

impl Mlp for SparseMoe {
    const KIND: Composite = Composite::Moe;

    fn apply(&self, m: &Model, h: &Tensor) -> Tensor {
        let probs = h.matmul(&self.router).softmax();
        let ids = probs.top_k(TopK);
        let gates = probs.take(&ids).unit_sum();
        self.experts.apply(m, h, [&ids, &gates])
    }
}

/// Multipliers on the stream. One is a no-op.
#[derive(Clone, Copy, Debug)]
pub struct Scales {
    pub embed: f32,
    /// On the scores, in place of `1 / sqrt(head_dim)`.
    pub attention: Option<f32>,
    pub residual: f32,
    pub logits: f32,
}

impl Default for Scales {
    fn default() -> Self {
        Self {
            embed: 1.0,
            attention: None,
            residual: 1.0,
            logits: 1.0,
        }
    }
}

pub struct Config {
    pub stack: Stack,
    pub kv_heads: u64,
    pub rope_theta: f32,
    pub eps: f32,
    /// Experts per layer and per row. Zero means dense.
    pub experts: u64,
    pub top_k: u64,
    pub scales: Scales,
    /// The embedding table is also the output head.
    pub tied: bool,
}

/// `rope_theta` at the root, else under `rope_parameters`.
pub(crate) fn rope_theta(k: &mut Keys) -> f32 {
    let top = At::top();
    k.read(&top.clone().or(top.under(ROPE_PARAMETERS)), ROPE_THETA)
}

impl Config {
    /// The decoder `k` describes, for these launch bounds.
    pub fn read(k: &mut Keys, max_seqs: u64, max_context: u64) -> Self {
        let top = At::top();
        let kind = k.read_or(&top, MODEL_TYPE, String::new());
        k.fix(&top, HIDDEN_ACT.name, json!("silu"));
        k.fix(&top, ATTENTION_BIAS.name, json!(kind == "qwen2"));
        k.fix(&top, MLP_BIAS.name, json!(false));
        k.fix(&top, ROPE_SCALING.name, json!(null));
        k.fix(&top, USE_SLIDING_WINDOW.name, json!(false));
        let types = k.read_or(&top, LAYER_TYPES, Vec::new());
        if types.iter().any(|t| t != "full_attention") {
            k.misfit(
                LAYER_TYPES.name,
                "the graph attends to every row in each layer",
            );
        }
        let mlp_only: Vec<u64> = k.or(&["mlp_only_layers"], Vec::new());
        if !mlp_only.is_empty() {
            k.misfit("mlp_only_layers", "every layer must be routed");
        }
        let sparse_step: u64 = k.or(&["decoder_sparse_step"], 1);
        if sparse_step != 1 {
            k.misfit("decoder_sparse_step", "every layer must be routed");
        }
        if let Some(false) = k.or::<Option<bool>>(&["norm_topk_prob"], None) {
            k.misfit("norm_topk_prob", "sparse MoE requires normalized top-k");
        }
        let [vocab, hidden, layers, heads, ffn] =
            [VOCAB, HIDDEN, LAYERS, HEADS, FFN].map(|key| k.read(&top, key));
        let experts = k.or(&["num_local_experts", "num_experts"], 0);
        let ffn = match experts > 0 {
            true => k.or(&["moe_intermediate_size"], ffn),
            false => ffn,
        };
        let head_dim = k.read_or(&top, HEAD_DIM, hidden.checked_div(heads).unwrap_or(0));
        let sizes = [vocab, hidden, layers, heads, head_dim, ffn];
        Self {
            stack: Stack::sized(sizes, max_seqs, max_context),
            kv_heads: k.read(&top, KV_HEADS),
            rope_theta: rope_theta(k),
            eps: k.read(&top, EPS),
            experts,
            top_k: k.read_or(&top, TOP_K, 0),
            scales: Scales {
                embed: k.or(&["embedding_multiplier"], 1.0),
                attention: k.or(&["attention_multiplier"], None),
                residual: k.or(&["residual_multiplier"], 1.0),
                logits: 1.0 / k.or(&["logits_scaling"], 1.0),
            },
            tied: k.read_or(&top, TIED, false),
        }
    }

    /// Qwen2 14B sizes. Q, K, and V carry additive biases.
    pub fn qwen2(max_seqs: u64, max_context: u64) -> Self {
        Self {
            stack: Stack::sized([152_064, 5120, 48, 40, 128, 13_824], max_seqs, max_context),
            kv_heads: 8,
            rope_theta: 1e6,
            eps: 1e-6,
            experts: 0,
            top_k: 0,
            scales: Scales::default(),
            tied: false,
        }
    }

    /// Qwen3-0.6B's sizes. Its heads are normed.
    pub fn qwen3(max_seqs: u64, max_context: u64) -> Self {
        Self {
            stack: Stack::sized([151_936, 1024, 28, 16, 128, 3072], max_seqs, max_context),
            kv_heads: 8,
            rope_theta: 1e6,
            eps: 1e-6,
            experts: 0,
            top_k: 0,
            scales: Scales::default(),
            tied: true,
        }
    }

    /// SmolLM2-360M's sizes. A Llama: its heads are not normed.
    pub fn smollm2(max_seqs: u64, max_context: u64) -> Self {
        Self {
            stack: Stack::sized([49_152, 960, 32, 15, 64, 2560], max_seqs, max_context),
            kv_heads: 5,
            rope_theta: 1e5,
            eps: 1e-5,
            ..Self::qwen3(max_seqs, max_context)
        }
    }

    /// Granite 3.1 1B-A400M: a Llama routing 8 of 32 experts.
    pub fn granite(max_seqs: u64, max_context: u64) -> Self {
        Self {
            stack: Stack::sized([49_155, 1024, 24, 16, 64, 512], max_seqs, max_context),
            kv_heads: 8,
            rope_theta: 1.5e6,
            eps: 1e-6,
            experts: 32,
            top_k: 8,
            scales: Scales {
                embed: 12.0,
                attention: Some(1.0 / 64.0),
                residual: 0.22,
                logits: 1.0 / 6.0,
            },
            tied: true,
        }
    }

    /// Gemma 3 270M sizes on Qwen3's layout. Mostly head weights.
    pub fn gemma3(max_seqs: u64, max_context: u64) -> Self {
        Self {
            stack: Stack::sized([262_144, 640, 18, 4, 256, 2048], max_seqs, max_context),
            kv_heads: 1,
            ..Self::qwen3(max_seqs, max_context)
        }
    }

    pub fn qwen2_small() -> Self {
        Self {
            tied: false,
            ..Self::qwen3_small()
        }
    }

    pub fn qwen3_small() -> Self {
        Self {
            stack: Stack::small(64, 32, 128),
            kv_heads: 1,
            rope_theta: 1e4,
            eps: 1e-6,
            experts: 0,
            top_k: 0,
            scales: Scales::default(),
            tied: true,
        }
    }

    /// Four experts, two per row, normalized top-k weights.
    pub fn qwen3_moe_small() -> Self {
        Self {
            kv_heads: 2,
            experts: 4,
            top_k: 2,
            tied: false,
            ..Self::qwen3_small()
        }
    }

    pub fn smollm2_small() -> Self {
        Self {
            eps: 1e-5,
            ..Self::qwen3_small()
        }
    }

    /// Four experts, two per row, two KV heads.
    pub fn granite_small() -> Self {
        Self {
            kv_heads: 2,
            experts: 4,
            top_k: 2,
            scales: Scales {
                embed: 2.0,
                attention: Some(0.1),
                residual: 0.5,
                logits: 0.25,
            },
            ..Self::qwen3_small()
        }
    }

    /// A layout small enough to run in tests.
    pub fn toy(max_seqs: u64, max_context: u64) -> Self {
        Self {
            stack: Stack::sized([32, 16, 2, 4, 8, 32], max_seqs, max_context),
            kv_heads: 2,
            ..Self::qwen3_small()
        }
    }

    pub fn dims(&self) -> Vec<Dim> {
        let mut own = vec![(KvHeads, self.kv_heads)];
        if self.experts > 0 {
            own.extend([(Experts, self.experts), (TopK, self.top_k)]);
        }
        self.stack.dims(&own)
    }
}

#[derive(Weights)]
pub struct Layer<N: HeadNorm, M: Mlp, D: Dense = Linear, B: QkvBias = ()> {
    #[shape(Hidden)]
    pub ln1: Norm,
    #[shape(Hidden, Heads * HeadDim)]
    pub q: D,
    #[shape(Hidden, KvHeads * HeadDim)]
    pub k: D,
    #[shape(Hidden, KvHeads * HeadDim)]
    pub v: D,
    #[shape(Heads * HeadDim)]
    pub q_bias: B,
    #[shape(KvHeads * HeadDim)]
    pub k_bias: B,
    #[shape(KvHeads * HeadDim)]
    pub v_bias: B,
    #[shape(HeadDim)]
    pub q_norm: N,
    #[shape(HeadDim)]
    pub k_norm: N,
    #[shape(Heads * HeadDim, Hidden)]
    pub o: D,
    #[shape(Hidden)]
    pub ln2: Norm,
    pub mlp: M,
}

/// The trunk's weights. Untied heads are declared beside them.
#[derive(Weights)]
pub struct Decoder<N: HeadNorm, M: Mlp, B: QkvBias = ()> {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(Layers)]
    pub layers: Vec<Layer<N, M, Linear, B>>,
    #[shape(Hidden)]
    pub norm: Norm,
}

/// One layer's kept keys and values.
type Cache = [Kept; 2];

/// State one pass over the layers shares.
struct Pass<'a> {
    m: &'a Model,
    c: &'a Config,
    /// The table rows unembed against.
    head: &'a Embed,
    positions: &'a Tensor,
    /// The axis this launch's rows lie on.
    rows: DimName,
    /// Each row's kept-row lane: the rows are ragged.
    slots: Option<&'a Tensor>,
    /// Each row's keys, in place of the causal mask.
    keys: Option<&'a Tensor>,
}

/// `x` times `by`, with nothing traced for one.
fn scaled(x: Tensor, by: f32) -> Tensor {
    match by == 1.0 {
        true => x,
        false => x.scale(by),
    }
}

pub fn forward<N: HeadNorm, M: Mlp, B: QkvBias>(
    m: &Model,
    c: &Config,
    (w, head): (&Decoder<N, M, B>, Option<&Embed>),
    tokens: &Tensor,
    positions: &Tensor,
) -> Tensor {
    let rows = Context;
    let pass = Pass {
        m,
        c,
        head: head.unwrap_or(&w.embed),
        positions,
        rows,
        slots: None,
        keys: None,
    };
    pass.logits(w, tokens, None)
}

impl Pass<'_> {
    /// With `kept`, each layer also reads earlier launches' rows.
    fn logits<N: HeadNorm, M: Mlp, B: QkvBias>(
        &self,
        w: &Decoder<N, M, B>,
        tokens: &Tensor,
        kept: Option<&[Cache]>,
    ) -> Tensor {
        self.logits_taps(w, tokens, kept, &[]).0
    }

    /// Logits, and joined tapped layers when requested.
    fn logits_taps<N: HeadNorm, M: Mlp, B: QkvBias>(
        &self,
        w: &Decoder<N, M, B>,
        tokens: &Tensor,
        kept: Option<&[Cache]>,
        layers: &[u64],
    ) -> (Tensor, Option<Tensor>) {
        if layers.is_empty() {
            let [logits] = self.m.composite(Composite::Model, || {
                let h = self.hidden(w, tokens, kept);
                [scaled(h.unembed(self.head), self.c.scales.logits)]
            });
            return (logits, None);
        }
        let [logits, tapped] = self.m.composite(Composite::Model, || {
            let mut sink = TapSink::new(layers);
            let x = self.trunk(w, tokens, kept, |at, row| sink.record(at, row));
            let h = x.rms_norm(&w.norm, self.c.eps);
            let logits = scaled(h.unembed(self.head), self.c.scales.logits);
            [logits, sink.join().expect("tap layers")]
        });
        (logits, Some(tapped))
    }

    /// Each row's normed state, before the unembed.
    fn hidden<N: HeadNorm, M: Mlp, B: QkvBias>(
        &self,
        w: &Decoder<N, M, B>,
        tokens: &Tensor,
        kept: Option<&[Cache]>,
    ) -> Tensor {
        let x = self.trunk(w, tokens, kept, |_, _| {});
        x.rms_norm(&w.norm, self.c.eps)
    }

    /// The residual rows. `tap` sees each layer's, counting from 1.
    fn trunk<N: HeadNorm, M: Mlp, B: QkvBias>(
        &self,
        w: &Decoder<N, M, B>,
        tokens: &Tensor,
        kept: Option<&[Cache]>,
        mut tap: impl FnMut(u64, &Tensor),
    ) -> Tensor {
        let (m, c) = (self.m, self.c);
        let s = &c.scales;
        let mut x = scaled(m.embed(tokens, &w.embed), s.embed);
        for (i, l) in w.layers.iter().enumerate() {
            let [out] = m.composite(Composite::Layer, || {
                let h = x.rms_norm(&l.ln1, c.eps);
                let attend = || [self.attention(l, &h, kept.map(|k| &k[i]))];
                let [attn] = m.composite(Composite::Attention, attend);
                let x = scaled(attn, s.residual).add(&x);
                let h = x.rms_norm(&l.ln2, c.eps);
                let [mlp] = m.composite(M::KIND, || [l.mlp.apply(m, &h)]);
                [scaled(mlp, s.residual).add(&x)]
            });
            x = out;
            tap(i as u64 + 1, &x);
        }
        x
    }

    /// Queries and keys pass through `N` per head, then rotate.
    fn attention<N: HeadNorm, M: Mlp, B: QkvBias>(
        &self,
        l: &Layer<N, M, Linear, B>,
        h: &Tensor,
        kept: Option<&Cache>,
    ) -> Tensor {
        let (c, rows) = (self.c, self.rows);
        let lead: Vec<DimExpr> = match self.slots {
            Some(_) => vec![rows.into()],
            None => vec![Seqs.into(), rows.into()],
        };
        let shape = [Heads.into(), KvHeads.into(), HeadDim.into()];
        let heads =
            Attend::new(&l.o, lead, shape).rotary(self.positions, c.rope_theta, Pairs::Halves);
        let turn = |t: Tensor, side: Side| match side.of([&l.q_norm, &l.k_norm]) {
            Some(norm) => norm.apply(t, c.eps),
            None => t,
        };
        let root = 1.0 / (c.stack.head_dim as f32).sqrt();
        let scale = c.scales.attention.unwrap_or(root);
        let proj = |w: &Linear, b: &B, side: Side| {
            let count: DimExpr = match side {
                Side::Query => Heads.into(),
                Side::Key | Side::Value => KvHeads.into(),
            };
            let t = turn(heads.split(b.add(w.apply(h)), count, HeadDim), side);
            match side {
                Side::Query | Side::Key => heads.rotate(t),
                Side::Value => t,
            }
        };
        let q = proj(&l.q, &l.q_bias, Side::Query);
        let k = proj(&l.k, &l.k_bias, Side::Key);
        let v = proj(&l.v, &l.v_bias, Side::Value);
        let out = match (kept, self.slots) {
            (Some([keys, values]), Some(slots)) => {
                let by = [slots, self.positions];
                let (k, v) = (k.append(keys, by), v.append(values, by));
                (self.m).attend_rows(Mask::Causal, scale, [&q, &k, &v], by, None)
            }
            _ => match self.keys {
                Some(keys) => self
                    .m
                    .attend(Mask::Bidirectional, scale, [&q, &k, &v], Some(keys)),
                None => self.m.attend(Mask::Causal, scale, [&q, &k, &v], None),
            },
        };
        heads.merge(out)
    }
}

crate::preset::preset!(
    "qwen2",
    small: || stepper_with::<(), Swiglu, Vector>(&Config::qwen2_small()),
    step: |n| stepper_with::<(), Swiglu, Vector>(&Config::qwen2(1, n)),
    forward: |n| program_with::<(), Swiglu, Vector>(&Config::qwen2(1, n)),
);
crate::preset::traced!("qwen3", Config::qwen3_small, Config::qwen3, [Norm, Swiglu]);
crate::preset::traced!(
    "smollm2",
    Config::smollm2_small,
    Config::smollm2,
    [(), Swiglu]
);
crate::preset::traced!("granite", Config::granite_small, Config::granite, [(), Moe]);

/// One forward entry: tokens and positions in, logits out.
pub fn program<N: HeadNorm, M: Mlp>(c: &Config) -> Graph {
    program_with::<N, M, ()>(c)
}

/// Forward entry parameterized by optional Q/K/V bias `B`.
pub fn program_with<N: HeadNorm, M: Mlp, B: QkvBias>(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let (w, head) = with_head::<Decoder<N, M, B>>(&m, c.tied);
    let [tokens, positions] = tokens(&m);
    let logits = forward(&m, c, (&w, head.as_ref()), &tokens, &positions);
    m.finish([(Role::Logits, &logits)])
}

/// Residual taps parameterized by optional Q/K/V bias `B`.
pub fn taps_with<N: HeadNorm, M: Mlp, B: QkvBias>(c: &Config, layers: &[u64]) -> Graph {
    let m = Model::new(c.dims());
    let (w, _) = with_head::<Decoder<N, M, B>>(&m, c.tied);
    // Key lists keep padding out of each row's attention.
    let [tokens, positions, keys] = listed(&m);
    let pass = Pass {
        m: &m,
        c,
        head: &w.embed,
        positions: &positions,
        rows: Context,
        slots: None,
        keys: Some(&keys),
    };
    let mut sink = TapSink::new(layers);
    pass.trunk(&w, &tokens, None, |at, x| sink.record(at, x));
    let joined = sink.join().expect("at least one tap layer");
    m.finish([(Role::Logits, &joined)])
}

/// One step entry: rows in, each row's likeliest token out.
pub fn stepper<N: HeadNorm, M: Mlp>(c: &Config) -> Graph {
    stepper_with::<N, M, ()>(c)
}

/// Step entry parameterized by optional Q/K/V bias `B`.
pub fn stepper_with<N: HeadNorm, M: Mlp, B: QkvBias>(c: &Config) -> Graph {
    let mut dims = c.dims();
    let size = Size::Bounded(c.stack.max_context);
    dims.push(Dim { name: Rows, size });
    let m = Model::new(dims);
    let (w, head) = with_head::<Decoder<N, M, B>>(&m, c.tied);
    let masked = c.stack.steps == Steps::Masked;
    let ([tokens, positions, slots], allowed) = step_ports(&m, masked);
    let ty = TensorType::new([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let cache = |_: &Layer<N, M, Linear, B>| [m.kept(ty.clone()), m.kept(ty.clone())];
    let kept: Vec<Cache> = w.layers.iter().map(cache).collect();
    let (positions, rows) = (&positions, Rows);
    let pass = Pass {
        m: &m,
        c,
        head: head.as_ref().unwrap_or(&w.embed),
        positions,
        rows,
        slots: Some(&slots),
        keys: None,
    };
    let (logits, taps) = pass.logits_taps(&w, &tokens, Some(&kept), &c.stack.taps);
    let full = pick(&m, &logits, allowed.as_ref(), taps.as_ref());
    if c.stack.steps == Steps::Plain {
        return full;
    }
    // A second entry picks among candidate ids.
    let [tokens, positions, slots, ids] = sparse_ports(&m);
    let pass = Pass {
        positions: &positions,
        slots: Some(&slots),
        keys: None,
        ..pass
    };
    let one = DimExpr::Const(1);
    let h = pass.hidden(&w, &tokens, Some(&kept));
    let h = h.reshape(vec![Rows.into(), one, Hidden.into()]);
    let rows = m.embed(&ids, pass.head);
    let logits = h.matmul_rows(&rows).reshape([Rows, Candidates]);
    let logits = scaled(logits, c.scales.logits);
    let at = logits.top_k(1u64).reshape([Rows]);
    m.finish([(Role::Tokens, &at), (Role::Logits, &logits)])
}
