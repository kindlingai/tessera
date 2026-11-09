//! Qwen3: grouped-query attention with RoPE, and SwiGLU MLPs.

use crate::shared::{Stack, ports, swiglu, tokens};
use trace::DimName::*;
use trace::{
    Composite, DType, Dim, DimExpr, DimName, Embed, EntryKind, Graph, Kept, Linear, Mask, Model,
    Norm, Pairs, PathSeg, Role, Size, Tensor, TensorType, Weights,
};

pub struct Config {
    pub stack: Stack,
    pub kv_heads: u64,
    pub rope_theta: f32,
    pub eps: f32,
}

impl Config {
    /// The 0.6B checkpoint's sizes.
    pub fn small(max_seqs: u64, max_context: u64) -> Self {
        let stack = Stack {
            vocab: 151_936,
            hidden: 1024,
            layers: 28,
            heads: 16,
            head_dim: 128,
            ffn: 3072,
            max_seqs,
            max_context,
        };
        Self {
            stack,
            kv_heads: 8,
            rope_theta: 1e6,
            eps: 1e-6,
        }
    }

    /// The same layout, small enough to run in tests.
    pub fn toy(max_seqs: u64, max_context: u64) -> Self {
        let stack = Stack {
            vocab: 32,
            hidden: 16,
            layers: 2,
            heads: 4,
            head_dim: 8,
            ffn: 32,
            max_seqs,
            max_context,
        };
        Self {
            stack,
            kv_heads: 2,
            rope_theta: 1e4,
            eps: 1e-6,
        }
    }

    pub fn dims(&self) -> Vec<Dim> {
        self.stack.dims(&[(KvHeads, self.kv_heads)])
    }
}

#[derive(Weights)]
pub struct Layer {
    #[shape(Hidden)]
    pub ln1: Norm,
    #[shape(Hidden, Heads * HeadDim)]
    pub q: Linear,
    #[shape(Hidden, KvHeads * HeadDim)]
    pub k: Linear,
    #[shape(Hidden, KvHeads * HeadDim)]
    pub v: Linear,
    #[shape(HeadDim)]
    pub q_norm: Norm,
    #[shape(HeadDim)]
    pub k_norm: Norm,
    #[shape(Heads * HeadDim, Hidden)]
    pub o: Linear,
    #[shape(Hidden)]
    pub ln2: Norm,
    #[shape(Hidden, Ffn)]
    pub gate: Linear,
    #[shape(Hidden, Ffn)]
    pub up: Linear,
    #[shape(Ffn, Hidden)]
    pub down: Linear,
}

/// The embedding table is also the output head.
#[derive(Weights)]
pub struct Qwen3 {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(Layers)]
    pub layers: Vec<Layer>,
    #[shape(Hidden)]
    pub norm: Norm,
}

/// One layer's kept keys and values.
type Cache = [Kept; 2];

/// What one pass over the layers shares.
struct Pass<'a> {
    m: &'a Model,
    c: &'a Config,
    positions: &'a Tensor,
    /// The axis this launch's rows lie on.
    rows: DimName,
}

pub fn forward(m: &Model, c: &Config, w: &Qwen3, tokens: &Tensor, positions: &Tensor) -> Tensor {
    let rows = Context;
    let pass = Pass {
        m,
        c,
        positions,
        rows,
    };
    pass.logits(w, tokens, None)
}

impl Pass<'_> {
    /// With `kept`, each layer also reads earlier launches' rows.
    fn logits(&self, w: &Qwen3, tokens: &Tensor, kept: Option<&[Cache]>) -> Tensor {
        let (m, c) = (self.m, self.c);
        let mut x = m.embed(tokens, &w.embed);
        for (i, l) in w.layers.iter().enumerate() {
            let h = x.rms_norm(&l.ln1, c.eps);
            let attend = || [self.attention(l, &h, kept.map(|k| &k[i]))];
            let [attn] = m.composite(Composite::Attention, attend);
            x = x.add(&attn);
            let h = x.rms_norm(&l.ln2, c.eps);
            let [mlp] = m.composite(Composite::Mlp, || [swiglu(&h, [&l.gate, &l.up, &l.down])]);
            x = x.add(&mlp);
        }
        x.rms_norm(&w.norm, c.eps).unembed(&w.embed)
    }

    /// Queries and keys are normed per head, then rotated.
    fn attention(&self, l: &Layer, h: &Tensor, kept: Option<&Cache>) -> Tensor {
        let (c, rows) = (self.c, self.rows);
        let split = |w: &Linear, heads: DimName| h.matmul(w).reshape([Seqs, rows, heads, HeadDim]);
        let turn = |t: Tensor, norm: &Norm| {
            let normed = t.rms_norm(norm, c.eps);
            normed.rope(self.positions, c.rope_theta, Pairs::Halves)
        };
        let q = turn(split(&l.q, Heads), &l.q_norm);
        let mut k = turn(split(&l.k, KvHeads), &l.k_norm);
        let mut v = split(&l.v, KvHeads);
        if let Some([keys, values]) = kept {
            (k, v) = (k.append(keys), v.append(values));
        }
        let scale = 1.0 / (c.stack.head_dim as f32).sqrt();
        let out = self.m.attend(Mask::Causal, scale, [&q, &k, &v], None);
        let flat: Vec<DimExpr> = vec![Seqs.into(), rows.into(), Heads * HeadDim];
        out.reshape(flat).matmul(&l.o)
    }
}

/// One forward entry: tokens and positions in, logits out.
pub fn program(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Qwen3 = trace::declare(&m);
    let [tokens, positions] = tokens(&m);
    let logits = forward(&m, c, &w, &tokens, &positions);
    m.finish([(Role::Logits, &logits)])
}

/// One step entry: rows in, each row's likeliest token out.
pub fn stepper(c: &Config) -> Graph {
    let mut dims = c.dims();
    let size = Size::Bounded(c.stack.max_context);
    dims.push(Dim { name: Rows, size });
    let m = Model::new(dims);
    let w: Qwen3 = trace::declare(&m);
    let [tokens, positions] = ports(&m, EntryKind::Step, Rows);
    let ty = TensorType::new([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let cache = |_: &Layer| [m.kept(ty.clone()), m.kept(ty.clone())];
    let kept: Vec<Cache> = w.layers.iter().map(cache).collect();
    let (positions, rows) = (&positions, Rows);
    let pass = Pass {
        m: &m,
        c,
        positions,
        rows,
    };
    let logits = pass.logits(&w, &tokens, Some(&kept));
    let ids = logits.top_k(1u64).reshape([Seqs, Rows]);
    m.finish([(Role::Tokens, &ids), (Role::Logits, &logits)])
}

/// The released checkpoint's tensor name for a weight path.
pub fn tensor_name(path: &[PathSeg]) -> Option<String> {
    use PathSeg::{Field, Index};
    let name = match path {
        [Field(f)] if f == "embed" => "embed_tokens".into(),
        [Field(f)] if f == "norm" => "norm".into(),
        [Field(layers), Index(i), Field(f)] if layers == "layers" => {
            let part = match f.as_str() {
                "ln1" => "input_layernorm",
                "ln2" => "post_attention_layernorm",
                "q_norm" | "k_norm" => &format!("self_attn.{f}"),
                "q" | "k" | "v" | "o" => &format!("self_attn.{f}_proj"),
                "gate" | "up" | "down" => &format!("mlp.{f}_proj"),
                _ => return None,
            };
            format!("layers.{i}.{part}")
        }
        _ => return None,
    };
    Some(format!("model.{name}.weight"))
}
