//! Qwen3.8-Flash-Next (`qwen4_exp`): 4-stream HC, PLE, GDN, GQA and MoE.

use crate::keys::Keys;
use crate::qwen35::{
    DeltaNet, Full, KEY_DIM, KEY_HEADS, L2_EPS, Mixing, Recurrent, VALUE_DIM, VALUE_HEADS, WINDOW,
    either, mixing, norm, rotary, text,
};
use crate::shared::{
    Attend, CodedExperts, ExpertSet, NOPE, PlainExperts, ROPE, Stack, Steps, TapSink, part,
    routed_calib, rows, swiglu_tapped,
};
use serde_json::json;
use std::cell::RefCell;
use tessera_trace::DimName::*;
use tessera_trace::pick;
use tessera_trace::{
    Act, CalibTap, Composite, DType, Dim, DimExpr, DimName, Embed, EntryKind, Graph, Kept, Linear,
    Mask, Model, Pairs, PathSeg, Port, Role, Size, Taps, Tensor, TensorType, Vector, Weights,
};

const GDN_HIST: DimName = DimName::named("GdnHist");
const HC: DimName = DimName::named("Hc");
const HC_RANK: DimName = DimName::named("HcRank");
const SHARED_FFN: DimName = DimName::named("SharedFfn");
const PLE_DIM: DimName = DimName::named("PleDim");
const PLE_WINDOW: DimName = DimName::named("PleWindow");
const PLE_HIST: DimName = DimName::named("PleHist");

/// Layer PLE configuration.
#[derive(Clone, Copy, Debug)]
pub struct Ple {
    /// 0-based layer index receiving PLE.
    pub layer: u64,
    pub dim: u64,
    pub window: u64,
    /// The n-gram size, which is also the conv dilation.
    pub dilation: u32,
    /// Hash heads: one group per n-gram order past unigrams.
    pub heads: u64,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub stack: Stack,
    pub kv_heads: u64,
    pub rotary: u64,
    pub rope_theta: f32,
    pub eps: f32,
    pub hc_count: u64,
    pub hc_rank: u64,
    pub experts: u64,
    pub top_k: u64,
    pub shared_ffn: u64,
    pub linear: Recurrent,
    pub layers: Vec<Mixing>,
    pub ple: Option<Ple>,
    pub eos_token: i32,
    pub tied: bool,
}

impl Config {
    pub fn read(k: &mut Keys, max_seqs: u64, max_context: u64) -> Self {
        let fixed = [
            ("hidden_act", json!("silu")),
            ("attention_bias", json!(false)),
            ("output_gate_type", json!("sigmoid")),
        ];
        for (key, want) in fixed {
            k.fixed(&format!("text_config.{key}"), want.clone());
            k.fixed(key, want);
        }
        let sizes = [
            text::<u64>(k, "vocab_size"),
            text(k, "hidden_size"),
            text(k, "num_hidden_layers"),
            text(k, "num_attention_heads"),
            text(k, "head_dim"),
            text(k, "moe_intermediate_size"),
        ];
        let layers = mixing(k, sizes[2]);
        let rotary = rotary(k, sizes[4]);
        let linear = Recurrent {
            key_heads: text(k, "linear_num_key_heads"),
            value_heads: text(k, "linear_num_value_heads"),
            key_dim: text(k, "linear_key_head_dim"),
            value_dim: text(k, "linear_value_head_dim"),
            window: text(k, "linear_conv_kernel_dim"),
        };
        let ids = ["text_config.ple_layer_ids", "ple_layer_ids"];
        let ple_ids: Vec<u64> = k.or(&ids, Vec::new());
        if ple_ids.len() > 1 {
            k.misfit(
                ids[0],
                format!("one PLE layer is supported, not {ple_ids:?}"),
            );
        }
        let ple = ple_ids.first().map(|&id| {
            let ngram: u32 = text(k, "ngram_size");
            let per: u64 = text(k, "heads_per_ngram");
            Ple {
                layer: id.saturating_sub(1),
                dim: text(k, "ple_embed_dim"),
                window: text(k, "ple_conv_kernel_size"),
                dilation: ngram,
                heads: per * u64::from(ngram.saturating_sub(1)),
            }
        });
        Self {
            stack: Stack::sized(sizes, max_seqs, max_context),
            kv_heads: text(k, "num_key_value_heads"),
            rotary,
            rope_theta: either(k, "rope_parameters.rope_theta", "rope_theta", 1e7),
            eps: text(k, "rms_norm_eps"),
            hc_count: text(k, "hc_count"),
            hc_rank: text(k, "hc_lowrank"),
            experts: text(k, "num_experts"),
            top_k: text(k, "num_experts_per_tok"),
            shared_ffn: either(
                k,
                "shared_expert_intermediate_size",
                "shared_expert_intermediate_size",
                sizes[5],
            ),
            linear,
            layers,
            ple,
            eos_token: text(k, "eos_token_id"),
            tied: k.or(
                &["text_config.tie_word_embeddings", "tie_word_embeddings"],
                false,
            ),
        }
    }

    pub fn dims(&self) -> Vec<Dim> {
        let (width, l) = (self.stack.head_dim, &self.linear);
        let mut own = vec![
            (KvHeads, self.kv_heads),
            (KEY_HEADS, l.key_heads),
            (VALUE_HEADS, l.value_heads),
            (KEY_DIM, l.key_dim),
            (VALUE_DIM, l.value_dim),
            (WINDOW, l.window),
            (GDN_HIST, l.window.saturating_sub(1)),
            (HC, self.hc_count),
            (HC_RANK, self.hc_rank),
            (Experts, self.experts),
            (TopK, self.top_k),
            (SHARED_FFN, self.shared_ffn),
        ];
        if self.rotary < width {
            own.extend([(ROPE, self.rotary), (NOPE, width - self.rotary)]);
        }
        if let Some(p) = self.ple {
            let hist = p.window.saturating_sub(1) * u64::from(p.dilation);
            own.extend([(PLE_DIM, p.dim), (PLE_WINDOW, p.window), (PLE_HIST, hist)]);
        }
        self.stack.dims(&own)
    }

    fn partial(&self) -> bool {
        self.rotary < self.stack.head_dim
    }
}

/// Grouped zero-centered RMSNorm over `[..lead, HC, Hidden]`, flattened.
fn grouped_norm(x: &Tensor, gain: &Vector, lead: &[DimExpr], eps: f32) -> Tensor {
    let r = x.rms(eps).reshape(cat(lead, [HC * Hidden]));
    r.add(&r.mul_vector(gain))
}

fn cat<const N: usize>(lead: &[DimExpr], rest: [DimExpr; N]) -> Vec<DimExpr> {
    lead.iter().cloned().chain(rest).collect()
}

/// Low-rank 4-stream hyper-connection with residual injection.
#[derive(Weights)]
struct HyperConn {
    #[shape(HC * Hidden)]
    norm: Vector,
    #[shape(HC * Hidden, HC_RANK)]
    down: Linear,
    #[shape(HC_RANK, HC * Hidden)]
    up: Linear,
    #[shape(HC * Hidden, HC)]
    inject: Linear,
}

impl HyperConn {
    fn pre(&self, hyper: &Tensor, lead: &[DimExpr], eps: f32) -> [Tensor; 2] {
        let hn = grouped_norm(hyper, &self.norm, lead, eps);
        let down = hn.matmul(&self.down).scale(0.25).act(Act::Silu);
        let mix_w = down.matmul(&self.up).act(Act::Sigmoid);
        let mixed = hn
            .mul(&mix_w)
            .reshape(cat(lead, [HC.into(), Hidden.into()]))
            .mean(lead.len() as u32);
        let inj = hn
            .matmul(&self.inject)
            .scale(0.25)
            .act(Act::Sigmoid)
            .scale(2.0);
        [mixed, inj]
    }

    fn post(hyper: &Tensor, y: &Tensor, inj: &Tensor, lead: &[DimExpr]) -> Tensor {
        let axis = lead.len() as u32;
        let y_exp = y
            .reshape(cat(lead, [1u64.into(), Hidden.into()]))
            .expand(axis, HC);
        let i_exp = inj
            .reshape(cat(lead, [HC.into(), 1u64.into()]))
            .expand(axis + 1, Hidden);
        hyper.add(&y_exp.mul(&i_exp))
    }
}

/// Final hyper-connection mixer collapsing streams to one residual.
#[derive(Weights)]
struct FinalMixer {
    #[shape(HC * Hidden)]
    norm: Vector,
    #[shape(HC * Hidden, HC_RANK)]
    down: Linear,
    #[shape(HC_RANK, HC * Hidden)]
    up: Linear,
}

impl FinalMixer {
    fn apply(&self, hyper: &Tensor, lead: &[DimExpr], eps: f32) -> Tensor {
        let hn = grouped_norm(hyper, &self.norm, lead, eps);
        let down = hn.matmul(&self.down).scale(0.25).act(Act::Silu);
        let mix_w = down.matmul(&self.up).act(Act::Sigmoid);
        hn.mul(&mix_w)
            .reshape(cat(lead, [HC.into(), Hidden.into()]))
            .mean(lead.len() as u32)
    }
}

/// Per-layer n-gram embedding block injected before layer attention.
#[derive(Weights)]
struct PleBlock {
    #[shape(PLE_DIM, HC * Hidden)]
    key_proj: Linear,
    #[shape(PLE_DIM, Hidden)]
    value_proj: Linear,
    #[shape(HC * Hidden, PLE_WINDOW)]
    conv: Taps,
    #[shape(HC * Hidden)]
    norm_key: Vector,
    #[shape(HC * Hidden)]
    norm_query: Vector,
    #[shape(HC * Hidden)]
    norm_conv: Vector,
}

impl PleBlock {
    fn apply(
        &self,
        c: &Config,
        hyper: &Tensor,
        ple: &Tensor,
        lead: &[DimExpr],
        step: Option<(&Kept, [&Tensor; 2])>,
    ) -> Tensor {
        let axis = lead.len() as u32;
        let p = c.ple.expect("ple configured");
        let hc_shape = cat(lead, [HC.into(), Hidden.into()]);
        let key_raw = ple.matmul(&self.key_proj).reshape(hc_shape.clone());
        let key_n = grouped_norm(&key_raw, &self.norm_key, lead, c.eps).reshape(hc_shape.clone());
        let q_n = grouped_norm(hyper, &self.norm_query, lead, c.eps).reshape(hc_shape.clone());
        let val = ple.matmul(&self.value_proj);
        let dot = key_n
            .mul(&q_n)
            .mean(axis + 1)
            .scale((c.stack.hidden as f32).sqrt());
        let gate = dot.act(Act::SignedSqrt).act(Act::Sigmoid);
        let val_exp = val
            .reshape(cat(lead, [1u64.into(), Hidden.into()]))
            .expand(axis, HC);
        let gate_exp = gate
            .reshape(cat(lead, [HC.into(), 1u64.into()]))
            .expand(axis + 1, Hidden);
        let gv = val_exp.mul(&gate_exp);
        let gv_n = grouped_norm(&gv, &self.norm_conv, lead, c.eps);
        let conv = match step {
            Some((state, ids)) => gv_n.conv_rows(&self.conv, p.dilation, state, ids),
            None => gv_n.conv_dilated(&self.conv, p.dilation),
        };
        let conv = conv.act(Act::Silu).reshape(hc_shape);
        hyper.add(&gv).add(&conv)
    }
}

/// Applies full GQA attention, whole-sequence or ragged step.
fn apply_full(
    f: &Full,
    m: &Model,
    c: &Config,
    h: &Tensor,
    positions: &Tensor,
    lead: &[DimExpr],
    step: Option<(&Kept, &Kept, &Tensor)>,
) -> Tensor {
    let shape = [Heads.into(), KvHeads.into(), HeadDim.into()];
    let mut heads =
        Attend::new(&f.o, lead.to_vec(), shape).rotary(positions, c.rope_theta, Pairs::Halves);
    if c.partial() {
        heads = heads.partial([ROPE.into(), NOPE.into()]);
    }
    let both = heads.split(h.matmul(&f.q), Heads, HeadDim * 2u64);
    let last = both.ty().rank() as u32 - 1;
    let q = norm(&both.slice(last, 0u64, HeadDim), &f.q_norm, c.eps);
    let gate = both.slice(last, HeadDim, HeadDim).act(Act::Sigmoid);
    let k = norm(
        &heads.split(h.matmul(&f.k), KvHeads, HeadDim),
        &f.k_norm,
        c.eps,
    );
    let v = heads.split(h.matmul(&f.v), KvHeads, HeadDim);
    let (q, k) = (heads.rotate(q), heads.rotate(k));
    let scale = 1.0 / (c.stack.head_dim as f32).sqrt();
    let out = match step {
        Some((k_kept, v_kept, slots)) => {
            let k = k.append(k_kept, [slots, positions]);
            let v = v.append(v_kept, [slots, positions]);
            m.attend_rows(Mask::Causal, scale, [&q, &k, &v], [slots, positions], None)
        }
        None => m.attend(Mask::Causal, scale, [&q, &k, &v], None),
    };
    heads.merge(out.mul(&gate))
}

/// Applies GDN with sigmoid output gate, whole or ragged step.
fn apply_gdn(
    d: &DeltaNet,
    m: &Model,
    c: &Config,
    h: &Tensor,
    lead: &[DimExpr],
    step: Option<(&Kept, &Kept, [&Tensor; 2])>,
) -> Tensor {
    let keys = KEY_HEADS * KEY_DIM;
    let values = VALUE_HEADS * VALUE_DIM;
    let raw = h.matmul(&d.qkv);
    let mixed = match step {
        Some((conv_st, _, ids)) => raw.conv_rows(&d.conv, 1, conv_st, ids),
        None => raw.conv(&d.conv),
    }
    .act(Act::Silu);
    let axis = lead.len() as u32;
    let split = |start: DimExpr, len: DimExpr, pair: [DimExpr; 2]| {
        mixed.slice(axis, start, len).reshape(cat(lead, pair))
    };
    let key = [KEY_HEADS.into(), KEY_DIM.into()];
    let q = split(0u64.into(), keys.clone(), key.clone());
    let k = split(keys.clone(), keys.clone(), key);
    let v = split(
        keys * 2u64,
        values.clone(),
        [VALUE_HEADS.into(), VALUE_DIM.into()],
    );
    let width = c.linear.key_dim as f32;
    let q = q.rms(L2_EPS / width);
    let k = k.rms(L2_EPS / width).scale(width.sqrt().recip());
    let decay = h
        .matmul(&d.a)
        .add_vector(&d.dt_bias)
        .act(Act::Softplus)
        .mul(&d.a_log.stored().act(Act::Exp))
        .scale(-1.0);
    let rate = h.matmul(&d.b).act(Act::Sigmoid);
    let out = match step {
        Some((_, delta_st, ids)) => {
            m.delta_rows(width.recip(), delta_st, [&q, &k, &v], [&decay, &rate], ids)
        }
        None => m.delta(width.recip(), [&q, &k, &v], [&decay, &rate]),
    };
    let z = h
        .matmul(&d.z)
        .reshape(cat(lead, [VALUE_HEADS.into(), VALUE_DIM.into()]));
    let gated = out.rms_norm(&d.norm, c.eps).mul(&z.act(Act::Sigmoid));
    gated.reshape(cat(lead, [values])).matmul(&d.out)
}

/// Shared expert MLP with scalar sigmoid output gate.
#[derive(Weights)]
struct SharedMlp {
    #[shape(Hidden, SHARED_FFN)]
    gate: Linear,
    #[shape(Hidden, SHARED_FFN)]
    up: Linear,
    #[shape(SHARED_FFN, Hidden)]
    down: Linear,
    #[shape(Hidden, 1)]
    scale: Linear,
}

impl SharedMlp {
    fn apply_tapped(&self, h: &Tensor) -> [Tensor; 2] {
        let last = h.ty().rank() as u32 - 1;
        let g = h.matmul(&self.scale).act(Act::Sigmoid).expand(last, Hidden);
        let [out, inner] = swiglu_tapped(h, [&self.gate, &self.up, &self.down]);
        [out.mul(&g), inner]
    }
}

/// Softmax top-k router, normalized over selected experts.
fn route(router: &Linear, h: &Tensor) -> [Tensor; 2] {
    let probs = h.matmul(router).softmax();
    let ids = probs.top_k(TopK);
    let gates = probs.take(&ids).unit_sum();
    [ids, gates]
}

/// A layer's routed MoE block.
pub trait Routed: Weights {
    fn apply(&self, m: &Model, h: &Tensor) -> Tensor;
    /// Same as `apply`, recording calibration taps via `note`.
    fn apply_calib(
        &self,
        m: &Model,
        h: &Tensor,
        note: &mut dyn FnMut(CalibTap, &Tensor),
    ) -> Tensor {
        let _ = note;
        self.apply(m, h)
    }
    /// Rotates a tap into this layer's input gauge, when present.
    fn tap_gauge(&self, h: &Tensor) -> Tensor {
        h.clone()
    }
}

/// Routed experts stored as `E`, plus gated shared expert.
#[derive(Weights)]
pub struct MoeOf<E: ExpertSet> {
    #[shape(Hidden, Experts)]
    router: Linear,
    #[flatten]
    experts: E,
    shared: SharedMlp,
}

pub type Moe = MoeOf<PlainExperts>;
pub type RotatedMoe = MoeOf<CodedExperts>;

impl<E: ExpertSet> Routed for MoeOf<E> {
    fn apply(&self, m: &Model, h: &Tensor) -> Tensor {
        self.apply_calib(m, h, &mut |_, _| {})
    }

    fn apply_calib(
        &self,
        m: &Model,
        h: &Tensor,
        note: &mut dyn FnMut(CalibTap, &Tensor),
    ) -> Tensor {
        let picks = route(&self.router, h);
        routed_calib(
            m,
            &self.experts,
            h,
            picks,
            self.shared.apply_tapped(h),
            note,
        )
    }

    fn tap_gauge(&self, h: &Tensor) -> Tensor {
        self.experts.tap_gauge(h)
    }
}

enum Mixer {
    Linear(DeltaNet),
    Full(Full),
}

struct Layer<R: Routed> {
    ple: Option<PleBlock>,
    attn_hc: HyperConn,
    mixer: Mixer,
    mlp_hc: HyperConn,
    moe: R,
}

struct Qwen4Exp<R: Routed> {
    embed: Embed,
    layers: Vec<Layer<R>>,
    final_hc: FinalMixer,
    head: Option<Embed>,
}

impl<R: Routed> Qwen4Exp<R> {
    fn declare(m: &Model, c: &Config) -> Self {
        let shape = [Vocab.into(), Hidden.into()];
        let embed = part(m, &mut Vec::new(), "embed", &shape);
        let mut layers = Vec::with_capacity(c.layers.len());
        for (i, kind) in c.layers.iter().enumerate() {
            let mut path = vec![PathSeg::Field("layers".into()), PathSeg::Index(i as u32)];
            let ple = c
                .ple
                .filter(|p| p.layer == i as u64)
                .map(|_| part(m, &mut path, "ple", &[]));
            let attn_hc = part(m, &mut path, "attn_hc", &[]);
            let mixer = match kind {
                Mixing::Linear => Mixer::Linear(part(m, &mut path, "linear", &[])),
                Mixing::Full => Mixer::Full(part(m, &mut path, "attn", &[])),
            };
            let mlp_hc = part(m, &mut path, "mlp_hc", &[]);
            let moe = part(m, &mut path, "moe", &[]);
            layers.push(Layer {
                ple,
                attn_hc,
                mixer,
                mlp_hc,
                moe,
            });
        }
        let final_hc = part(m, &mut Vec::new(), "final_hc", &[]);
        let head = (!c.tied).then(|| part(m, &mut Vec::new(), "head", &shape));
        Self {
            embed,
            layers,
            final_hc,
            head,
        }
    }
}

enum LayerCache {
    Linear { conv: Kept, state: Kept },
    Full { k: Kept, v: Kept },
}

struct StepState<'a> {
    slots: &'a Tensor,
    ple: Option<Kept>,
    layers: Vec<LayerCache>,
}

/// Runs all layers and returns logits, with optional taps.
#[allow(clippy::too_many_arguments)]
fn run<R: Routed>(
    m: &Model,
    c: &Config,
    w: &Qwen4Exp<R>,
    [tokens, positions]: [&Tensor; 2],
    ple_in: Option<&Tensor>,
    step: Option<&StepState>,
    taps: &[u64],
    calib: Option<&RefCell<Vec<(Role, Tensor)>>>,
) -> (Tensor, Option<Tensor>) {
    let lead: Vec<DimExpr> = match step {
        Some(_) => vec![Rows.into()],
        None => vec![Seqs.into(), Context.into()],
    };
    let axis = lead.len() as u32;
    let body = || {
        let mut sink = TapSink::new(taps);
        let mut hyper = m
            .embed(tokens, &w.embed)
            .reshape(cat(&lead, [1u64.into(), Hidden.into()]))
            .expand(axis, HC);
        for (i, l) in w.layers.iter().enumerate() {
            let [next] = m.composite(Composite::Layer, || {
                let mut h_in = hyper.clone();
                if let Some(ple) = &l.ple {
                    let ple_emb = ple_in.expect("ple port present");
                    let ple_step =
                        step.and_then(|s| s.ple.as_ref().map(|k| (k, [s.slots, positions])));
                    h_in = ple.apply(c, &h_in, ple_emb, &lead, ple_step);
                }
                let [mixed, inj] = l.attn_hc.pre(&h_in, &lead, c.eps);
                let [attn_out] = m.composite(Composite::Attention, || {
                    let out = match (&l.mixer, step.map(|s| (&s.layers[i], s.slots))) {
                        (Mixer::Linear(net), Some((LayerCache::Linear { conv, state }, sl))) => {
                            apply_gdn(
                                net,
                                m,
                                c,
                                &mixed,
                                &lead,
                                Some((conv, state, [sl, positions])),
                            )
                        }
                        (Mixer::Linear(net), None) => apply_gdn(net, m, c, &mixed, &lead, None),
                        (Mixer::Full(attn), Some((LayerCache::Full { k, v }, sl))) => {
                            apply_full(attn, m, c, &mixed, positions, &lead, Some((k, v, sl)))
                        }
                        (Mixer::Full(attn), None) => {
                            apply_full(attn, m, c, &mixed, positions, &lead, None)
                        }
                        _ => unreachable!("layer cache matches mixer"),
                    };
                    [out]
                });
                let mid = HyperConn::post(&h_in, &attn_out, &inj, &lead);
                let [mlp_in, mlp_inj] = l.mlp_hc.pre(&mid, &lead, c.eps);
                let mut note = |tap: CalibTap, t: &Tensor| {
                    if let Some(buf) = calib {
                        buf.borrow_mut().push((Role::Calib(tap), t.clone()));
                    }
                };
                note(CalibTap::MlpIn, &mlp_in);
                let [mlp_out] = m.composite(Composite::Moe, || {
                    [l.moe.apply_calib(m, &mlp_in, &mut note)]
                });
                [HyperConn::post(&mid, &mlp_out, &mlp_inj, &lead)]
            });
            hyper = next;
            if !taps.is_empty() {
                let plain_id = i as u64 + 1;
                let gauge_id = 1000 + plain_id;
                if taps.contains(&plain_id) {
                    sink.record(plain_id, &hyper.mean(axis));
                }
                if taps.contains(&gauge_id) {
                    sink.record(gauge_id, &l.moe.tap_gauge(&hyper.mean(axis)));
                }
            }
        }
        let h = w.final_hc.apply(&hyper, &lead, c.eps);
        let logits = h.unembed(w.head.as_ref().unwrap_or(&w.embed));
        (logits, sink.join())
    };
    if taps.is_empty() {
        let [logits] = m.composite(Composite::Model, || [body().0]);
        return (logits, None);
    }
    let [logits, tapped] = m.composite(Composite::Model, || {
        let (l, t) = body();
        [l, t.expect("taps requested")]
    });
    (logits, Some(tapped))
}

fn forward_entry(m: &Model, c: &Config) -> (Tensor, Tensor, Option<Tensor>) {
    let tok_ty = TensorType::new([Seqs, Context], DType::U32);
    if c.ple.is_some() {
        let [t, p, ple] = m.entry(
            EntryKind::Forward,
            [
                Port {
                    role: Role::Tokens,
                    ty: tok_ty.clone(),
                },
                Port {
                    role: Role::Positions,
                    ty: tok_ty,
                },
                Port {
                    role: Role::Ple,
                    ty: TensorType::new([Seqs, Context, PLE_DIM], DType::F32),
                },
            ],
        );
        (t, p, Some(ple))
    } else {
        let [t, p] = m.entry(
            EntryKind::Forward,
            [
                Port {
                    role: Role::Tokens,
                    ty: tok_ty.clone(),
                },
                Port {
                    role: Role::Positions,
                    ty: tok_ty,
                },
            ],
        );
        (t, p, None)
    }
}

/// Whole-sequence forward graph.
pub fn program<R: Routed>(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w = Qwen4Exp::<R>::declare(&m, c);
    let (tokens, positions, ple) = forward_entry(&m, c);
    let (logits, _) = run(
        &m,
        c,
        &w,
        [&tokens, &positions],
        ple.as_ref(),
        None,
        &[],
        None,
    );
    m.finish([(Role::Logits, &logits)])
}

/// Whole-sequence forward graph with per-layer calibration taps.
pub fn calib_program<R: Routed>(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w = Qwen4Exp::<R>::declare(&m, c);
    let (tokens, positions, ple) = forward_entry(&m, c);
    let calib = RefCell::new(Vec::new());
    let (logits, _) = run(
        &m,
        c,
        &w,
        [&tokens, &positions],
        ple.as_ref(),
        None,
        &[],
        Some(&calib),
    );
    let taps = calib.into_inner();
    let mut outs: Vec<(Role, &Tensor)> = Vec::with_capacity(1 + taps.len());
    outs.push((Role::Logits, &logits));
    for (role, t) in &taps {
        outs.push((*role, t));
    }
    m.finish_many(&outs)
}

/// Ragged step graph with kept KV and recurrent states.
pub fn stepper<R: Routed>(c: &Config) -> Graph {
    let mut dims = c.dims();
    dims.push(Dim {
        name: Rows,
        size: Size::Bounded(c.stack.max_context),
    });
    let m = Model::new(dims);
    let w = Qwen4Exp::<R>::declare(&m, c);
    let row = |role| Port {
        role,
        ty: TensorType::new([Rows], DType::U32),
    };
    let ple_port = || Port {
        role: Role::Ple,
        ty: TensorType::new([Rows, PLE_DIM], DType::F32),
    };
    let mask_port = || Port {
        role: Role::Allowed,
        ty: TensorType::new([Rows, VocabWords], DType::U32),
    };
    let masked = c.stack.steps == Steps::Masked;
    let [t, p, s] = [Role::Tokens, Role::Positions, Role::Slots].map(row);
    let (tokens, positions, slots, ple_in, allowed) = match (c.ple.is_some(), masked) {
        (true, true) => {
            let [t, p, s, ple, mask] = m.entry(EntryKind::Step, [t, p, s, ple_port(), mask_port()]);
            (t, p, s, Some(ple), Some(mask))
        }
        (true, false) => {
            let [t, p, s, ple] = m.entry(EntryKind::Step, [t, p, s, ple_port()]);
            (t, p, s, Some(ple), None)
        }
        (false, true) => {
            let [t, p, s, mask] = m.entry(EntryKind::Step, [t, p, s, mask_port()]);
            (t, p, s, None, Some(mask))
        }
        (false, false) => {
            let [t, p, s] = m.entry(EntryKind::Step, [t, p, s]);
            (t, p, s, None, None)
        }
    };
    let ple_state = c.ple.map(|_| {
        m.kept(TensorType::new(
            vec![Seqs.into(), PLE_HIST.into(), HC * Hidden],
            DType::F32,
        ))
    });
    let qkv_width = KEY_HEADS * KEY_DIM * 2u64 + VALUE_HEADS * VALUE_DIM;
    let mut layers = Vec::with_capacity(c.layers.len());
    for kind in &c.layers {
        layers.push(match kind {
            Mixing::Linear => LayerCache::Linear {
                conv: m.kept(TensorType::new(
                    vec![Seqs.into(), GDN_HIST.into(), qkv_width.clone()],
                    DType::F32,
                )),
                state: m.kept(TensorType::new(
                    vec![
                        Seqs.into(),
                        VALUE_HEADS.into(),
                        KEY_DIM.into(),
                        VALUE_DIM.into(),
                    ],
                    DType::F32,
                )),
            },
            Mixing::Full => LayerCache::Full {
                k: m.kept(TensorType::new(
                    rows([KvHeads.into(), HeadDim.into()]),
                    DType::F32,
                )),
                v: m.kept(TensorType::new(
                    rows([KvHeads.into(), HeadDim.into()]),
                    DType::F32,
                )),
            },
        });
    }
    let state = StepState {
        slots: &slots,
        ple: ple_state,
        layers,
    };
    let (logits, taps) = run(
        &m,
        c,
        &w,
        [&tokens, &positions],
        ple_in.as_ref(),
        Some(&state),
        &c.stack.taps,
        None,
    );
    pick(&m, &logits, allowed.as_ref(), taps.as_ref())
}
