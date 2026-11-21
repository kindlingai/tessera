//! Decoders: grouped-query attention, RoPE, a dense or routed MLP.

use crate::keys::Keys;
use crate::shared::{
    Attend, Side, Stack, Steps, listed, pick, sparse_ports, step_ports, swiglu, tokens,
};
use serde_json::json;
use tessera_trace::DimName::*;
use tessera_trace::{
    Act, Composite, DType, Dim, DimExpr, DimName, Embed, Graph, Kept, Linear, Linears, Mask, Model,
    Norm, Pairs, Role, Size, Tensor, TensorType, Weights,
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

/// The second half of a layer, over the normed rows.
pub trait Mlp: Weights {
    const KIND: Composite;
    fn apply(&self, m: &Model, h: &Tensor) -> Tensor;
}

/// A SwiGLU MLP every row passes through.
#[derive(Weights)]
pub struct Swiglu {
    #[shape(Hidden, Ffn)]
    pub gate: Linear,
    #[shape(Hidden, Ffn)]
    pub up: Linear,
    #[shape(Ffn, Hidden)]
    pub down: Linear,
}

impl Mlp for Swiglu {
    const KIND: Composite = Composite::Mlp;

    fn apply(&self, _: &Model, h: &Tensor) -> Tensor {
        swiglu(h, [&self.gate, &self.up, &self.down])
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
}

impl Config {
    /// The decoder `k` describes, for these launch bounds.
    pub fn read(k: &mut Keys, max_seqs: u64, max_context: u64) -> Self {
        k.fixed("hidden_act", json!("silu"));
        k.fixed("attention_bias", json!(false));
        k.fixed("mlp_bias", json!(false));
        k.fixed("rope_scaling", json!(null));
        k.fixed("use_sliding_window", json!(false));
        let types: Vec<String> = k.or(&["layer_types"], Vec::new());
        if types.iter().any(|t| t != "full_attention") {
            k.misfit(
                "layer_types",
                "the graph attends to every row in each layer",
            );
        }
        let [vocab, hidden, layers, heads, ffn] = [
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
        ]
        .map(|key| k.need::<u64>(&[key]));
        let head_dim = k.or(&["head_dim"], hidden.checked_div(heads).unwrap_or(0));
        let sizes = [vocab, hidden, layers, heads, head_dim, ffn];
        Self {
            stack: Stack::sized(sizes, max_seqs, max_context),
            kv_heads: k.need(&["num_key_value_heads"]),
            rope_theta: k.need(&["rope_theta", "rope_parameters.rope_theta"]),
            eps: k.need(&["rms_norm_eps"]),
            experts: k.or(&["num_local_experts"], 0),
            top_k: k.or(&["num_experts_per_tok"], 0),
            scales: Scales {
                embed: k.or(&["embedding_multiplier"], 1.0),
                attention: k.or(&["attention_multiplier"], None),
                residual: k.or(&["residual_multiplier"], 1.0),
                logits: 1.0 / k.or(&["logits_scaling"], 1.0),
            },
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
        }
    }

    /// SmolLM2-360M's sizes. A Llama: its heads are not normed.
    pub fn smollm2(max_seqs: u64, max_context: u64) -> Self {
        Self {
            stack: Stack::sized([49_152, 960, 32, 15, 64, 2560], max_seqs, max_context),
            kv_heads: 5,
            rope_theta: 1e5,
            eps: 1e-5,
            experts: 0,
            top_k: 0,
            scales: Scales::default(),
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
        }
    }

    /// A layout small enough to run in tests.
    pub fn toy(max_seqs: u64, max_context: u64) -> Self {
        Self {
            stack: Stack::sized([32, 16, 2, 4, 8, 32], max_seqs, max_context),
            kv_heads: 2,
            rope_theta: 1e4,
            eps: 1e-6,
            experts: 0,
            top_k: 0,
            scales: Scales::default(),
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
pub struct Layer<N: HeadNorm, M: Mlp> {
    #[shape(Hidden)]
    pub ln1: Norm,
    #[shape(Hidden, Heads * HeadDim)]
    pub q: Linear,
    #[shape(Hidden, KvHeads * HeadDim)]
    pub k: Linear,
    #[shape(Hidden, KvHeads * HeadDim)]
    pub v: Linear,
    #[shape(HeadDim)]
    pub q_norm: N,
    #[shape(HeadDim)]
    pub k_norm: N,
    #[shape(Heads * HeadDim, Hidden)]
    pub o: Linear,
    #[shape(Hidden)]
    pub ln2: Norm,
    pub mlp: M,
}

/// The embedding table is also the output head.
#[derive(Weights)]
pub struct Decoder<N: HeadNorm, M: Mlp> {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(Layers)]
    pub layers: Vec<Layer<N, M>>,
    #[shape(Hidden)]
    pub norm: Norm,
}

/// One layer's kept keys and values.
type Cache = [Kept; 2];

/// State one pass over the layers shares.
struct Pass<'a> {
    m: &'a Model,
    c: &'a Config,
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

pub fn forward<N: HeadNorm, M: Mlp>(
    m: &Model,
    c: &Config,
    w: &Decoder<N, M>,
    tokens: &Tensor,
    positions: &Tensor,
) -> Tensor {
    let rows = Context;
    let pass = Pass {
        m,
        c,
        positions,
        rows,
        slots: None,
        keys: None,
    };
    pass.logits(w, tokens, None)
}

impl Pass<'_> {
    /// With `kept`, each layer also reads earlier launches' rows.
    fn logits<N: HeadNorm, M: Mlp>(
        &self,
        w: &Decoder<N, M>,
        tokens: &Tensor,
        kept: Option<&[Cache]>,
    ) -> Tensor {
        let [logits] = self.m.composite(Composite::Model, || {
            let h = self.hidden(w, tokens, kept);
            [scaled(h.unembed(&w.embed), self.c.scales.logits)]
        });
        logits
    }

    /// Each row's normed state, before the unembed.
    fn hidden<N: HeadNorm, M: Mlp>(
        &self,
        w: &Decoder<N, M>,
        tokens: &Tensor,
        kept: Option<&[Cache]>,
    ) -> Tensor {
        let x = self.trunk(w, tokens, kept, |_, _| {});
        x.rms_norm(&w.norm, self.c.eps)
    }

    /// The residual rows. `tap` sees each layer's, counting from 1.
    fn trunk<N: HeadNorm, M: Mlp>(
        &self,
        w: &Decoder<N, M>,
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
    fn attention<N: HeadNorm, M: Mlp>(
        &self,
        l: &Layer<N, M>,
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
        heads.apply([&l.q, &l.k, &l.v], h, turn, |[q, k, v]| {
            match (kept, self.slots) {
                (Some([keys, values]), Some(slots)) => {
                    let by = [slots, self.positions];
                    let (k, v) = (k.append(keys, by), v.append(values, by));
                    self.m.attend_rows(scale, [q, &k, &v], by, None)
                }
                _ => match self.keys {
                    Some(keys) => self
                        .m
                        .attend(Mask::Bidirectional, scale, [q, k, v], Some(keys)),
                    None => self.m.attend(Mask::Causal, scale, [q, k, v], None),
                },
            }
        })
    }
}

/// One forward entry: tokens and positions in, logits out.
pub fn program<N: HeadNorm, M: Mlp>(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Decoder<N, M> = tessera_trace::declare(&m);
    let [tokens, positions] = tokens(&m);
    let logits = forward(&m, c, &w, &tokens, &positions);
    m.finish([(Role::Logits, &logits)])
}

/// Residual rows after `layers`, joined on the last axis.
pub fn taps<N: HeadNorm, M: Mlp>(c: &Config, layers: &[u64]) -> Graph {
    let m = Model::new(c.dims());
    let w: Decoder<N, M> = tessera_trace::declare(&m);
    // Key lists keep padding out of each row's attention.
    let [tokens, positions, keys] = listed(&m);
    let pass = Pass {
        m: &m,
        c,
        positions: &positions,
        rows: Context,
        slots: None,
        keys: Some(&keys),
    };
    let mut taken: Vec<Tensor> = Vec::new();
    pass.trunk(&w, &tokens, None, |at, x| {
        if layers.contains(&at) {
            taken.push(x.clone());
        }
    });
    assert_eq!(taken.len(), layers.len(), "taps {layers:?} past the layers");
    let first = taken[0].clone();
    let joined = taken[1..].iter().fold(first, |all, t| all.concat(t));
    m.finish([(Role::Logits, &joined)])
}

/// One step entry: rows in, each row's likeliest token out.
pub fn stepper<N: HeadNorm, M: Mlp>(c: &Config) -> Graph {
    let mut dims = c.dims();
    let size = Size::Bounded(c.stack.max_context);
    dims.push(Dim { name: Rows, size });
    let m = Model::new(dims);
    let w: Decoder<N, M> = tessera_trace::declare(&m);
    let masked = c.stack.steps == Steps::Masked;
    let ([tokens, positions, slots], allowed) = step_ports(&m, masked);
    let ty = TensorType::new([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let cache = |_: &Layer<N, M>| [m.kept(ty.clone()), m.kept(ty.clone())];
    let kept: Vec<Cache> = w.layers.iter().map(cache).collect();
    let (positions, rows) = (&positions, Rows);
    let pass = Pass {
        m: &m,
        c,
        positions,
        rows,
        slots: Some(&slots),
        keys: None,
    };
    let logits = pass.logits(&w, &tokens, Some(&kept));
    let full = pick(&m, &logits, allowed.as_ref());
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
    let rows = m.embed(&ids, &w.embed);
    let logits = h.matmul_rows(&rows).reshape([Rows, Candidates]);
    let logits = scaled(logits, c.scales.logits);
    let at = logits.top_k(1u64).reshape([Rows]);
    m.finish([(Role::Tokens, &at), (Role::Logits, &logits)])
}
