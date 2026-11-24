//! Gemma 4 text: a causal decoder with per-layer embeddings.

use crate::eg2::{Attention, FULL_HEAD_DIM, Layer, PER_LAYER, after, context, full_layers};
use crate::keys::Keys;
use crate::shared::{Attend, Dense, Side, Stack};
use serde_json::json;
use tessera_trace as trace;
use trace::DimName::{Context, HeadDim, Heads, KvHeads, Rows, Seqs};
use trace::{Act, Composite, Dim, DimExpr, Embed, Graph, Kept, Mask, Model, Norm, Pairs, PathSeg};
use trace::{Role, Size, Tensor, TensorType, Weights};

pub struct Config {
    /// Its `head_dim` is a sliding layer's.
    pub stack: Stack,
    pub kv_heads: u64,
    /// Every n-th layer attends to the whole text.
    pub full_every: u64,
    pub full_head_dim: u64,
    /// Pairs a full layer's rotary turns.
    pub full_turned: u32,
    pub per_layer: u64,
    /// Keys a sliding layer sees, its own included.
    pub window: u64,
    pub theta: f32,
    pub full_theta: f32,
    /// Layers from this one on read an earlier layer's keys.
    pub first_shared: u64,
    pub eps: f32,
    /// Logits become `cap * tanh(logits / cap)`.
    pub cap: Option<f32>,
}

impl Config {
    /// The config `k` holds, for these launch bounds.
    pub fn read(k: &mut Keys, max_seqs: u64, max_context: u64) -> Self {
        let fixed = [
            ("model_type", json!("gemma4_text")),
            ("hidden_activation", json!("gelu_pytorch_tanh")),
            ("attention_bias", json!(false)),
            ("attention_k_eq_v", json!(false)),
            ("enable_moe_block", json!(false)),
            ("use_double_wide_mlp", json!(false)),
            ("tie_word_embeddings", json!(true)),
            (
                "rope_parameters.sliding_attention.rope_type",
                json!("default"),
            ),
            (
                "rope_parameters.full_attention.rope_type",
                json!("proportional"),
            ),
        ];
        for (key, want) in fixed {
            k.fixed(&format!("text_config.{key}"), want);
        }
        let sizes = [
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "head_dim",
            "intermediate_size",
        ]
        .map(|key| k.need(&[&format!("text_config.{key}")]));
        let [vocab, _, layers, ..] = sizes;
        k.fixed("text_config.vocab_size_per_layer_input", json!(vocab));
        let kv_heads = k.need(&["text_config.num_key_value_heads"]);
        let full_kv: Option<u64> = k.or(&["text_config.num_global_key_value_heads"], None);
        if full_kv.is_some_and(|n| n != kv_heads) {
            let why = "the graph gives full layers the sliding layers' key heads";
            k.misfit("text_config.num_global_key_value_heads", why);
        }
        let full_head_dim: u64 = k.need(&["text_config.global_head_dim"]);
        let part: f32 = k.or(
            &["text_config.rope_parameters.full_attention.partial_rotary_factor"],
            1.0,
        );
        let shared: u64 = k.need(&["text_config.num_kv_shared_layers"]);
        let full_every = full_layers(k, layers);
        let first_shared = layers.saturating_sub(shared);
        // Each kind needs an own layer to keep its keys.
        if first_shared < full_every {
            let why = "a kind of layer has no earlier layer to read keys from";
            k.misfit("text_config.num_kv_shared_layers", why);
        }
        Self {
            stack: Stack::sized(sizes, max_seqs, max_context),
            kv_heads,
            full_every,
            full_head_dim,
            full_turned: (part * full_head_dim as f32) as u32 / 2,
            per_layer: k.need(&["text_config.hidden_size_per_layer_input"]),
            window: k.need(&["text_config.sliding_window"]),
            theta: k.need(&["text_config.rope_parameters.sliding_attention.rope_theta"]),
            full_theta: k.need(&["text_config.rope_parameters.full_attention.rope_theta"]),
            first_shared,
            eps: k.need(&["text_config.rms_norm_eps"]),
            cap: k.or(&["text_config.final_logit_softcapping"], None),
        }
    }

    /// Whether layer `l` attends to the whole text.
    pub fn full(&self, l: u64) -> bool {
        (l + 1).is_multiple_of(self.full_every)
    }

    /// Whether layer `l` is its kind's last own layer.
    pub fn hands_on(&self, l: u64) -> bool {
        let later = (l + 1..self.first_shared).any(|n| self.full(n) == self.full(l));
        l < self.first_shared && !later
    }

    /// Layer `l`'s head width.
    pub fn width(&self, l: u64) -> DimExpr {
        match self.full(l) {
            true => FULL_HEAD_DIM.into(),
            false => HeadDim.into(),
        }
    }

    pub fn dims(&self) -> Vec<Dim> {
        self.stack.dims(&[
            (KvHeads, self.kv_heads),
            (FULL_HEAD_DIM, self.full_head_dim),
            (PER_LAYER, self.per_layer),
        ])
    }
}

/// A layer that reads another's keys: `S0` head width.
#[derive(Weights)]
pub struct Queries<D: Dense> {
    #[shape(Hidden)]
    pub ln: Norm,
    #[shape(Hidden, Heads * S0)]
    pub q: D,
    #[shape(S0)]
    pub q_norm: Norm,
    #[shape(Heads * S0, Hidden)]
    pub o: D,
    #[shape(Hidden)]
    pub post: Norm,
}

/// One layer's attention weights.
pub enum Block<D: Dense> {
    Own(Attention<D>),
    Shared(Queries<D>),
}

/// The weights outside attention.
#[derive(Weights)]
struct Gemma4<D: Dense> {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    /// Each token's per-layer input, the layers side by side.
    #[shape(Vocab, Layers * PER_LAYER)]
    pub ple_embed: Embed,
    #[shape(Layers)]
    pub layers: Vec<Layer<D>>,
    #[shape(PER_LAYER)]
    pub ple_norm: Norm,
    #[shape(Hidden)]
    pub norm: Norm,
}

/// Each layer's attention, shaped for its kind.
pub fn attention<D: Dense>(m: &Model, c: &Config) -> Vec<Block<D>> {
    let mut path = vec![PathSeg::Field("attention".into())];
    let mut out = Vec::new();
    for l in 0..c.stack.layers {
        path.push(PathSeg::Index(l as u32));
        let width = c.width(l);
        let heads = match l < c.first_shared {
            true => Block::Own(Attention::declare(m, &mut path, &[width, KvHeads.into()])),
            false => Block::Shared(Queries::declare(m, &mut path, &[width])),
        };
        path.pop();
        out.push(heads);
    }
    out
}

/// One layer's kept keys and values.
type Cache = [Kept; 2];

/// Keys and values a layer attends over.
type Kv = [Tensor; 2];

/// State one pass over the layers shares.
struct Pass<'a> {
    m: &'a Model,
    c: &'a Config,
    positions: &'a Tensor,
    /// Each row's slot and the kept rows, for ragged rows.
    step: Option<(&'a Tensor, &'a [Cache])>,
}

impl Pass<'_> {
    /// Axes before the hidden one.
    fn lead(&self) -> Vec<DimExpr> {
        match self.step {
            Some(_) => vec![Rows.into()],
            None => vec![Seqs.into(), Context.into()],
        }
    }

    /// The residual rows after the last layer.
    fn trunk<D: Dense>(&self, w: &Gemma4<D>, heads: &[Block<D>], tokens: &Tensor) -> Tensor {
        let (m, c) = (self.m, self.c);
        let (eps, hidden, per) = (c.eps, c.stack.hidden as f32, c.per_layer);
        let inputs = m.embed(tokens, &w.embed).scale(hidden.sqrt());
        let table = m.embed(tokens, &w.ple_embed).scale((per as f32).sqrt());
        let last = self.lead().len() as u32;
        let mut x = inputs.clone();
        // Each kind's latest own keys and values, sliding first.
        let mut kept: [Option<Kv>; 2] = [None, None];
        for (l, layer) in w.layers.iter().enumerate() {
            let own = table.slice(last, l as u64 * per, per);
            let ple = context(layer, &inputs, &w.ple_norm, [hidden, eps]).add(&own);
            let ple = ple.scale(0.5f32.sqrt());
            let (l, full) = (l as u64, c.full(l as u64) as usize);
            let shared = kept[full].clone();
            let attend = || self.attend(l, &heads[l as usize], &x, shared);
            // Each kind's last own layer hands its keys on.
            let attn = match c.hands_on(l) {
                true => {
                    let [attn, k, v] = m.composite(Composite::Attention, attend);
                    kept[full] = Some([k, v]);
                    attn
                }
                false => {
                    let [attn] = m.composite(Composite::Attention, || [attend()[0].clone()]);
                    attn
                }
            };
            x = after(m, eps, &w.layers[l as usize], x.add(&attn), &ple);
        }
        x
    }

    /// The output, then the keys and values it read.
    fn attend<D: Dense>(
        &self,
        l: u64,
        heads: &Block<D>,
        x: &Tensor,
        shared: Option<Kv>,
    ) -> [Tensor; 3] {
        let c = self.c;
        let full = c.full(l);
        let (theta, mask, turned) = match full {
            true => (c.full_theta, Mask::Causal, Some(c.full_turned)),
            false => (c.theta, Mask::Back(c.window - 1), None),
        };
        let (ln, q, q_norm, o, post) = match heads {
            Block::Own(a) => (&a.ln, &a.q, &a.q_norm, &a.o, &a.post),
            Block::Shared(a) => (&a.ln, &a.q, &a.q_norm, &a.o, &a.post),
        };
        let h = x.rms_norm(ln, c.eps);
        let shape = [Heads.into(), KvHeads.into(), c.width(l)];
        let at = Attend::new(o, self.lead(), shape);
        let at = at
            .rotary(self.positions, theta, Pairs::Halves)
            .turning(turned);
        let turn = |t: Tensor, side: Side| match side {
            Side::Query => t.rms_norm(q_norm, c.eps),
            Side::Key => match heads {
                Block::Own(a) => t.rms_norm(&a.k_norm, c.eps),
                Block::Shared(_) => t,
            },
            Side::Value => t.rms(c.eps),
        };
        let q = at.side(q, &h, Side::Query, turn);
        let [k, v] = match heads {
            Block::Own(a) => {
                let [k, v] = [(&a.k, Side::Key), (&a.v, Side::Value)];
                let [k, v] = [k, v].map(|(w, side)| at.side(w, &h, side, turn));
                match self.step {
                    Some((slots, caches)) => {
                        let by = [slots, self.positions];
                        let [keys, values] = &caches[l as usize];
                        [k.append(keys, by), v.append(values, by)]
                    }
                    None => [k, v],
                }
            }
            Block::Shared(_) => shared.expect("an earlier layer's keys"),
        };
        let out = match self.step {
            Some((slots, _)) => {
                let by = [slots, self.positions];
                self.m.attend_rows(mask, 1.0, [&q, &k, &v], by, None)
            }
            None => self.m.attend(mask, 1.0, [&q, &k, &v], None),
        };
        [at.merge(out).rms_norm(post, c.eps), k, v]
    }

    /// Logits through the tied embedding.
    fn logits<D: Dense>(&self, w: &Gemma4<D>, heads: &[Block<D>], tokens: &Tensor) -> Tensor {
        let [logits] = self.m.composite(Composite::Model, || {
            let x = self.trunk(w, heads, tokens);
            [x.rms_norm(&w.norm, self.c.eps).unembed(&w.embed)]
        });
        match self.c.cap {
            Some(cap) => logits.scale(cap.recip()).act(Act::Tanh).scale(cap),
            None => logits,
        }
    }
}

/// Four layers, sliding then full. The last two share keys.
pub fn small() -> Config {
    let mut stack = Stack::small(256, 128, 256);
    stack.layers = 4;
    Config {
        stack,
        kv_heads: 1,
        full_every: 2,
        full_head_dim: 256,
        full_turned: 32,
        per_layer: 128,
        window: 4,
        theta: 1e4,
        full_theta: 1e6,
        first_shared: 2,
        eps: 1e-6,
        cap: Some(30.0),
    }
}

crate::preset::preset!("gemma4", small: || stepper::<trace::Linear>(&small()));

/// One forward entry: tokens and positions in, logits out.
pub fn program<D: Dense>(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w: Gemma4<D> = trace::declare(&m);
    let heads = attention(&m, c);
    let [tokens, positions] = tessera_trace::tokens(&m);
    let pass = Pass {
        m: &m,
        c,
        positions: &positions,
        step: None,
    };
    let logits = pass.logits(&w, &heads, &tokens);
    m.finish([(Role::Logits, &logits)])
}

/// One step entry: rows in, each row's likeliest token out.
pub fn stepper<D: Dense>(c: &Config) -> Graph {
    let mut dims = c.dims();
    let size = Size::Bounded(c.stack.max_context);
    dims.push(Dim { name: Rows, size });
    let m = Model::new(dims);
    let w: Gemma4<D> = trace::declare(&m);
    let heads = attention(&m, c);
    let masked = c.stack.steps == crate::Steps::Masked;
    let ([tokens, positions, slots], allowed) = tessera_trace::step_ports(&m, masked);
    let mut caches = Vec::new();
    for l in 0..c.first_shared {
        let ty = TensorType::new(
            vec![Seqs.into(), Context.into(), KvHeads.into(), c.width(l)],
            trace::DType::F32,
        );
        caches.push([m.kept(ty.clone()), m.kept(ty)]);
    }
    let pass = Pass {
        m: &m,
        c,
        positions: &positions,
        step: Some((&slots, &caches)),
    };
    let logits = pass.logits(&w, &heads, &tokens);
    tessera_trace::pick(&m, &logits, allowed.as_ref(), None)
}
