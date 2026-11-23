//! DFlash block-diffusion drafter conditioned on tapped target states.

use crate::decoder::{HeadNorm, Layer, Mlp, Swiglu};
pub use crate::dflash_shard::shard;
use crate::keys::Keys;
use crate::shared::{Stack, join_rows, with_head};
use serde_json::json;
use tessera_trace::DimName::*;
use tessera_trace::{
    Act, Composite, DType, Dim, DimExpr, Embed, EntryKind, Graph, Kept, Linear, Mask, Model, Norm,
    Pairs, PathSeg, Port, Role, Size, Tensor, TensorType, Vector, Weights,
};

/// Dynamic grouped causal convolution dimensions for DFlash2 layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvConfig {
    pub taps: u64,
    pub group_size: u64,
}

/// Bilinear candidate path selector dimensions for DFlash2 outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectorConfig {
    pub rank: u64,
    pub top_k: u64,
}

/// Adjacent causal injection bridge dimensions for DSpine drafters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BridgeConfig {
    pub code_size: u64,
    pub rank: u64,
    pub top_k: u64,
    pub eps: f32,
}

/// Target model dimensions a drafter inherits when omitted.
#[derive(Clone, Copy, Debug)]
pub struct TargetSpec {
    pub vocab: u64,
    pub hidden: u64,
    pub layers: u64,
    pub tied: bool,
}

/// On-load quantization mode for BF16 drafter linears.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DraftQuant {
    /// Keep all BF16 weights in BF16.
    Bf16,
    /// Quantize layers and head, keep fc and selector BF16.
    #[default]
    MxFp8,
    /// Quantize all eligible 2D BF16 linears.
    MxFp8All,
}

/// DFlash drafter sizes and conditioning settings.
#[derive(Clone, Debug)]
pub struct Config {
    pub stack: Stack,
    pub kv_heads: u64,
    pub rope_theta: f32,
    pub eps: f32,
    /// The embedding table is also the output head.
    pub tied: bool,
    /// Anchor plus drafted tokens per parallel block.
    pub block_size: u64,
    pub mask_token: u32,
    pub target_hidden: u64,
    /// 0-based target layer indices whose states condition drafting.
    pub target_layers: Vec<u64>,
    /// Sliding attention window, when enabled by the checkpoint.
    pub sliding_window: Option<u64>,
    /// Whether draft queries attend causally within the block.
    pub causal: bool,
    /// Optional per-layer grouped dynamic causal convolutions (DFlash2).
    pub conv: Option<ConvConfig>,
    /// Optional candidate path selector over top-k logits (DFlash2).
    pub selector: Option<SelectorConfig>,
    /// Optional adjacent causal injection bridge (DSpine).
    pub bridge: Option<BridgeConfig>,
    pub output_multiplier: f32,
    pub logit_softcap: Option<f32>,
    /// On-load quantization mode for eligible BF16 linear weights.
    pub quant: DraftQuant,
}

/// Evenly spaced 0-based target layer indices for `draft_layers`.
pub fn default_layers(target_layers: u64, draft_layers: u64) -> Vec<u64> {
    if draft_layers == 0 || target_layers == 0 {
        return Vec::new();
    }
    if draft_layers == 1 {
        return vec![target_layers / 2];
    }
    let start = 1u64.min(target_layers - 1);
    let end = target_layers.saturating_sub(3).max(start);
    let span = (end - start) as f64;
    let denom = (draft_layers - 1) as f64;
    let mut out = Vec::with_capacity(draft_layers as usize);
    for i in 0..draft_layers {
        let pos = start + ((i as f64 * span) / denom).round() as u64;
        out.push(pos);
    }
    out
}

impl Config {
    /// Reads a DFlash `config.json` against its `target`.
    pub fn read(k: &mut Keys, target: &TargetSpec, max_seqs: u64, max_context: u64) -> Self {
        k.fixed("hidden_act", json!("silu"));
        k.fixed("attention_bias", json!(false));
        k.fixed("mlp_bias", json!(false));
        let vocab = k.or(&["vocab_size"], target.vocab);
        let hidden = k.or(&["hidden_size"], target.hidden);
        let layers: u64 = k.need(&["num_hidden_layers"]);
        let heads: u64 = k.need(&["num_attention_heads"]);
        let head_dim = k.or(&["head_dim"], hidden.checked_div(heads).unwrap_or(0));
        let ffn: u64 = k.need(&["intermediate_size"]);
        let kv_heads: u64 = k.need(&["num_key_value_heads"]);
        let rope_theta: f32 = k.need(&["rope_theta", "rope_parameters.rope_theta"]);
        let eps: f32 = k.need(&["rms_norm_eps"]);
        let tied = k.or(&["tie_word_embeddings"], target.tied);
        let block_size = k.or(&["block_size", "dflash_config.block_size"], 16u64);
        if block_size < 2 {
            k.misfit("block_size", "block_size must be at least 2");
        }
        let mask_token: u32 = k.need(&["dflash_config.mask_token_id", "mask_token_id"]);
        let total_target = k.or(&["num_target_layers"], target.layers);
        let explicit: Option<Vec<u64>> = k.or(
            &["dflash_config.target_layer_ids", "target_layer_ids"],
            None,
        );
        let target_layers = explicit.unwrap_or_else(|| default_layers(total_target, layers));
        if target_layers.is_empty() {
            k.misfit("target_layer_ids", "need at least one target layer");
        }
        if target_layers.iter().any(|&l| l >= target.layers) {
            k.misfit(
                "target_layer_ids",
                format!("layer id past target's {} layers", target.layers),
            );
        }
        let use_window = k.or(&["use_sliding_window"], false);
        let sliding_window: Option<u64> = if use_window {
            k.or(&["sliding_window"], None).filter(|&w| w > 1)
        } else {
            None
        };
        let explicit_causal = k
            .or::<Option<bool>>(&["dflash_config.causal"], None)
            .or_else(|| k.or::<Option<bool>>(&["dflash_config.draft_block_causal"], None))
            .or_else(|| k.or::<Option<bool>>(&["is_causal"], None));
        let taps: Option<u64> = k.or(&["dflash_config.conv_kernel_size"], None);
        let group_size: Option<u64> = k.or(&["dflash_config.conv_group_size"], None);
        let conv = match (taps, group_size) {
            (Some(taps), Some(group_size)) => {
                if taps == 0 || group_size == 0 || !hidden.is_multiple_of(group_size) {
                    k.misfit("conv_group_size", "conv_group_size must divide hidden_size");
                }
                Some(ConvConfig { taps, group_size })
            }
            (None, None) => None,
            _ => {
                k.misfit("conv_kernel_size", "conv needs both kernel and group size");
                None
            }
        };
        let rank: Option<u64> = k.or(&["dflash_config.selector_rank"], None);
        let top_k: Option<u64> = k.or(&["dflash_config.selector_top_k"], None);
        let selector = match (rank, top_k) {
            (Some(rank), Some(top_k)) => {
                if rank == 0 || top_k == 0 {
                    k.misfit("selector_top_k", "selector rank and top_k must be positive");
                }
                Some(SelectorConfig { rank, top_k })
            }
            (None, None) => None,
            _ => {
                k.misfit("selector_rank", "selector needs both rank and top_k");
                None
            }
        };
        let bridge_code: Option<u64> = k.or(
            &[
                "dflash_config.selected_prefix_code_bridge.code_size",
                "dspine_config.transfer_rank",
            ],
            None,
        );
        let bridge_rank: Option<u64> = k.or(
            &[
                "dflash_config.selected_prefix_code_bridge.rank",
                "dspine_config.message_size",
            ],
            None,
        );
        let bridge = match (bridge_code, bridge_rank) {
            (Some(code_size), Some(rank)) => {
                let top_k = k.or(
                    &[
                        "dflash_config.selected_prefix_last_write_refine.head_topk",
                        "dspine_config.top_k",
                    ],
                    16u64,
                );
                let b_eps = k.or(&["dflash_config.selected_prefix_code_bridge.eps"], eps);
                if code_size == 0 || rank == 0 || top_k == 0 {
                    k.misfit(
                        "selected_prefix_code_bridge",
                        "bridge sizes must be positive",
                    );
                }
                Some(BridgeConfig {
                    code_size,
                    rank,
                    top_k,
                    eps: b_eps,
                })
            }
            (None, None) => None,
            _ => {
                k.misfit(
                    "selected_prefix_code_bridge",
                    "bridge needs code_size and rank",
                );
                None
            }
        };
        let causal = explicit_causal.unwrap_or(bridge.is_some());
        let output_multiplier = k.or(&["dflash_config.output_multiplier"], 1.0f32);
        let logit_softcap = k
            .or::<Option<f32>>(&["dflash_config.final_logit_softcapping"], None)
            .filter(|&v| v > 0.0);
        let quant = match k
            .or::<Option<String>>(&["dflash_config.quant"], None)
            .as_deref()
        {
            None | Some("mxfp8") => DraftQuant::MxFp8,
            Some("bf16") => DraftQuant::Bf16,
            Some("mxfp8_all") => DraftQuant::MxFp8All,
            Some(other) => {
                k.misfit("quant", format!("unknown draft quant `{other}`"));
                DraftQuant::MxFp8
            }
        };
        let sizes = [vocab, hidden, layers, heads, head_dim, ffn];
        Self {
            stack: Stack::sized(sizes, max_seqs, max_context),
            kv_heads,
            rope_theta,
            eps,
            tied,
            block_size,
            mask_token,
            target_hidden: target.hidden,
            target_layers,
            sliding_window,
            causal,
            conv,
            selector,
            bridge,
            output_multiplier,
            logit_softcap,
            quant,
        }
    }

    /// A small layout for tests against `target`.
    pub fn toy(
        target: &TargetSpec,
        block_size: u64,
        mask_token: u32,
        max_seqs: u64,
        max_context: u64,
    ) -> Self {
        let target_layers: Vec<u64> = (0..target.layers).collect();
        let sizes = [target.vocab, target.hidden, 1, 4, 8, 32];
        Self {
            stack: Stack::sized(sizes, max_seqs, max_context),
            kv_heads: 2,
            rope_theta: 1e4,
            eps: 1e-6,
            tied: target.tied,
            block_size,
            mask_token,
            target_hidden: target.hidden,
            target_layers,
            sliding_window: None,
            causal: false,
            conv: None,
            selector: None,
            bridge: None,
            output_multiplier: 1.0,
            logit_softcap: None,
            quant: DraftQuant::MxFp8,
        }
    }

    /// 1-based layer positions for `Stack::taps`.
    pub fn tap_layers(&self) -> Vec<u64> {
        self.target_layers.iter().map(|&l| l + 1).collect()
    }

    /// Concatenated target tap width fed to `fc`.
    pub fn tap_width(&self) -> u64 {
        self.target_hidden * self.target_layers.len() as u64
    }

    pub fn dims(&self) -> Vec<Dim> {
        let mut dims = self
            .stack
            .dims(&[(KvHeads, self.kv_heads), (Embedding, self.tap_width())]);
        dims.push(Dim {
            name: Rows,
            size: Size::Bounded(self.stack.max_context),
        });
        dims
    }
}

/// Drafter weights. Untied output heads sit beside them.
#[derive(Weights)]
pub struct Drafter<N: HeadNorm> {
    #[shape(Vocab, Hidden)]
    pub embed: Embed,
    #[shape(Embedding, Hidden)]
    pub fc: Linear,
    #[shape(Hidden)]
    pub hidden_norm: Norm,
    #[shape(Layers)]
    pub layers: Vec<Layer<N, Swiglu>>,
    #[shape(Hidden)]
    pub norm: Norm,
}

/// One draft layer's kept keys and values.
type Cache = [Kept; 2];

/// Input ports for one DFlash draft launch.
struct Ports {
    block_tokens: Tensor,
    block_pos: Tensor,
    block_slots: Tensor,
    attend_pos: Tensor,
    ctx_taps: Tensor,
    kv_pos: Tensor,
    kv_slots: Tensor,
}

fn entry_ports(m: &Model, block: u64) -> Ports {
    let port = |role, shape: Vec<DimExpr>, dtype| Port {
        role,
        ty: TensorType::new(shape, dtype),
    };
    let blk: DimExpr = block.into();
    let kv: DimExpr = Rows + block;
    let [bt, bp, bs, ap, ct, kp, ks] = m.entry(
        EntryKind::Step,
        [
            port(Role::Tokens, vec![blk.clone()], DType::U32),
            port(Role::Positions, vec![blk.clone()], DType::U32),
            port(Role::Slots, vec![blk.clone()], DType::U32),
            port(Role::Positions, vec![blk], DType::U32),
            port(Role::Taps, vec![Rows.into(), Embedding.into()], DType::F32),
            port(Role::Positions, vec![kv.clone()], DType::U32),
            port(Role::Slots, vec![kv], DType::U32),
        ],
    );
    Ports {
        block_tokens: bt,
        block_pos: bp,
        block_slots: bs,
        attend_pos: ap,
        ctx_taps: ct,
        kv_pos: kp,
        kv_slots: ks,
    }
}

/// Non-causal block attention over cached context and noise KV.
fn attention<N: HeadNorm>(
    m: &Model,
    c: &Config,
    l: &Layer<N, Swiglu>,
    ctx: &Tensor,
    h: &Tensor,
    p: &Ports,
    [keys, values]: &Cache,
) -> Tensor {
    let blk: DimExpr = c.block_size.into();
    let kv_rows: DimExpr = Rows + c.block_size;
    let kv_in = join_rows(ctx, h);
    let q = h
        .matmul(&l.q)
        .reshape(vec![blk.clone(), Heads.into(), HeadDim.into()]);
    let q = l
        .q_norm
        .apply(q, c.eps)
        .rope(&p.block_pos, c.rope_theta, Pairs::Halves);
    let k = kv_in
        .matmul(&l.k)
        .reshape(vec![kv_rows.clone(), KvHeads.into(), HeadDim.into()]);
    let k = l
        .k_norm
        .apply(k, c.eps)
        .rope(&p.kv_pos, c.rope_theta, Pairs::Halves);
    let v = kv_in
        .matmul(&l.v)
        .reshape(vec![kv_rows, KvHeads.into(), HeadDim.into()]);
    let by = [&p.kv_slots, &p.kv_pos];
    let (k, v) = (k.append(keys, by), v.append(values, by));
    let scale = 1.0 / (c.stack.head_dim as f32).sqrt();
    let mask = match c.sliding_window {
        Some(w) => Mask::Back(w - 1),
        None => Mask::Causal,
    };
    let attn = m.attend_rows(
        mask,
        scale,
        [&q, &k, &v],
        [&p.block_slots, &p.attend_pos],
        None,
    );
    attn.reshape(vec![blk, Heads * HeadDim]).matmul(&l.o)
}

/// Weights for one DFlash2 grouped dynamic causal convolution.
struct ConvWeights {
    base: Tensor,
    proj: Linear,
}

impl ConvWeights {
    fn declare(m: &Model, layer: u64, kind: &str, hidden: u64, conv: ConvConfig) -> Self {
        let groups = hidden / conv.group_size;
        let path = |leaf: &str| {
            vec![
                PathSeg::Field("layers".into()),
                PathSeg::Index(layer as u32),
                PathSeg::Field(kind.into()),
                PathSeg::Field(leaf.into()),
            ]
        };
        let base = m.weight(
            &path("base_kernel"),
            TensorType::new(
                vec![2u64.into(), conv.taps.into(), Hidden.into()],
                DType::F32,
            ),
        );
        let proj = Linear::of(m.weight(
            &path("kernel_projection"),
            TensorType::new(
                vec![(2 * conv.taps * groups).into(), Hidden.into()],
                DType::F32,
            ),
        ));
        Self { base, proj }
    }

    fn convolve(
        &self,
        x: &Tensor,
        delta: &Tensor,
        side: u64,
        block: u64,
        hidden: u64,
        conv: ConvConfig,
    ) -> Tensor {
        let groups = hidden / conv.group_size;
        let gs = conv.group_size;
        let x_g = x.reshape(vec![block.into(), groups.into(), gs.into()]);
        let base = self.base.scale(1.0);
        let mut out: Option<Tensor> = None;
        for tap in 0..conv.taps {
            let b = base
                .slice(0, side, 1u64)
                .slice(1, tap, 1u64)
                .reshape(vec![1u64.into(), groups.into(), gs.into()])
                .expand(0, block);
            let d = delta
                .slice(1, tap, 1u64)
                .reshape(vec![block.into(), groups.into(), 1u64.into()])
                .expand(2, gs);
            let shifted = if tap == 0 {
                x_g.clone()
            } else if tap < block {
                let zeros = x.slice(0, 0u64, tap).scale(0.0);
                let prev = x.slice(0, 0u64, block - tap);
                join_rows(&zeros, &prev).reshape(vec![block.into(), groups.into(), gs.into()])
            } else {
                x_g.scale(0.0)
            };
            let term = b.add(&d).mul(&shifted);
            out = Some(match out {
                Some(acc) => acc.add(&term),
                None => term,
            });
        }
        out.expect("conv taps >= 1")
            .reshape(vec![block.into(), Hidden.into()])
    }

    fn prepare(&self, x: &Tensor, block: u64, hidden: u64, conv: ConvConfig) -> (Tensor, Tensor) {
        let groups = hidden / conv.group_size;
        let coeff = x.matmul(&self.proj).reshape(vec![
            block.into(),
            2u64.into(),
            conv.taps.into(),
            groups.into(),
        ]);
        let d0 =
            coeff
                .slice(1, 0u64, 1u64)
                .reshape(vec![block.into(), conv.taps.into(), groups.into()]);
        let d1 =
            coeff
                .slice(1, 1u64, 1u64)
                .reshape(vec![block.into(), conv.taps.into(), groups.into()]);
        (self.convolve(x, &d0, 0, block, hidden, conv), d1)
    }

    fn finish(
        &self,
        x: &Tensor,
        delta: &Tensor,
        block: u64,
        hidden: u64,
        conv: ConvConfig,
    ) -> Tensor {
        self.convolve(x, delta, 1, block, hidden, conv)
    }
}

/// Weights for the DFlash2 bilinear candidate path selector.
struct SelectorWeights {
    proj: Linear,
    pred: Tensor,
    succ: Tensor,
}

impl SelectorWeights {
    fn declare(m: &Model, sel: SelectorConfig) -> Self {
        let path = |leaf: &str| {
            vec![
                PathSeg::Field("candidate_selector".into()),
                PathSeg::Field(leaf.into()),
            ]
        };
        let proj = Linear::of(m.weight(
            &path("hidden_projection"),
            TensorType::new(vec![sel.rank.into(), Hidden.into()], DType::F32),
        ));
        let pred = m.weight(
            &path("predecessor_codebook"),
            TensorType::new(vec![Vocab.into(), sel.rank.into()], DType::F32),
        );
        let succ = m.weight(
            &path("successor_codebook"),
            TensorType::new(vec![Vocab.into(), sel.rank.into()], DType::F32),
        );
        Self { proj, pred, succ }
    }

    fn lattice(
        &self,
        c: &Config,
        sel: SelectorConfig,
        draft: &Tensor,
        logits: &Tensor,
        block_tokens: &Tensor,
    ) -> (Tensor, Tensor) {
        let (n, k, r) = (c.block_size - 1, sel.top_k, sel.rank);
        let cands = logits.top_k(k);
        let mut unary = logits.take(&cands);
        if c.output_multiplier != 1.0 {
            unary = unary.scale(c.output_multiplier);
        }
        if let Some(cap) = c.logit_softcap {
            unary = unary.scale(1.0 / cap).act(Act::Tanh).scale(cap);
        }
        let proj = draft.matmul(&self.proj);
        let anchor = self
            .pred
            .gather(block_tokens)
            .slice(0, 0u64, 1u64)
            .expand(0, k);
        let cands_flat = cands.reshape(vec![(n * k).into()]);
        let pred = if n > 1 {
            let prev = self.pred.gather(&cands_flat).slice(0, 0u64, (n - 1) * k);
            join_rows(&anchor, &prev)
        } else {
            anchor
        }
        .reshape(vec![n.into(), k.into(), r.into()]);
        let succ = self
            .succ
            .gather(&cands_flat)
            .reshape(vec![n.into(), k.into(), r.into()]);
        let lhs = pred.mul(
            &proj
                .reshape(vec![n.into(), 1u64.into(), r.into()])
                .expand(1, k),
        );
        let pair = lhs.matmul_rows(&succ);
        let scores = pair.add(
            &unary
                .reshape(vec![n.into(), 1u64.into(), k.into()])
                .expand(1, k),
        );
        (cands_flat, scores)
    }
}

/// Weights for the DSpine adjacent causal injection bridge.
struct BridgeWeights {
    codebook: Tensor,
    gate: Linear,
    gate_bias: Vector,
    pred_gate: Linear,
    prev: Linear,
    reads: Vec<Linear>,
    writes: Vec<Linear>,
}

impl BridgeWeights {
    fn declare(m: &Model, layers: u64, cfg: BridgeConfig) -> Self {
        let bpath = |leaf: &str| vec![PathSeg::Field("bridge".into()), PathSeg::Field(leaf.into())];
        let lpath = |i: u64, leaf: &str| {
            vec![
                PathSeg::Field("layers".into()),
                PathSeg::Index(i as u32),
                PathSeg::Field(leaf.into()),
            ]
        };
        let codebook = m.weight(
            &bpath("codebook"),
            TensorType::new(vec![Vocab.into(), cfg.code_size.into()], DType::F32),
        );
        let rc = || TensorType::new(vec![cfg.rank.into(), cfg.code_size.into()], DType::F32);
        let gate = Linear::of(m.weight(&bpath("gate"), rc()));
        let gate_bias = Vector::of(m.weight(
            &bpath("gate_bias"),
            TensorType::new(vec![cfg.rank.into()], DType::F32),
        ));
        let pred_gate = Linear::of(m.weight(&bpath("pred_gate"), rc()));
        let prev = Linear::of(m.weight(&bpath("prev"), rc()));
        let mut reads = Vec::with_capacity(layers as usize);
        let mut writes = Vec::with_capacity(layers as usize);
        for i in 0..layers {
            reads.push(Linear::of(m.weight(
                &lpath(i, "bridge_read"),
                TensorType::new(vec![cfg.code_size.into(), Hidden.into()], DType::F32),
            )));
            writes.push(Linear::of(m.weight(
                &lpath(i, "bridge_write"),
                TensorType::new(vec![Hidden.into(), cfg.rank.into()], DType::F32),
            )));
        }
        Self {
            codebook,
            gate,
            gate_bias,
            pred_gate,
            prev,
            reads,
            writes,
        }
    }

    fn apply(
        &self,
        layer: usize,
        raw: &Tensor,
        block_tokens: &Tensor,
        c: &Config,
        cfg: BridgeConfig,
    ) -> (Tensor, Tensor, Tensor) {
        let (block, hidden, n) = (c.block_size, c.stack.hidden, c.block_size - 1);
        let normed = raw.rms(cfg.eps);
        let rms = raw
            .mul(&normed)
            .mean(1)
            .reshape(vec![block.into(), 1u64.into()]);
        let code_eps = (cfg.eps * cfg.eps / cfg.code_size as f32).max(1e-24);
        let proj = normed.matmul(&self.reads[layer]).rms(code_eps);
        let anchor = self
            .codebook
            .gather(block_tokens)
            .slice(0, 0u64, 1u64)
            .scale((cfg.code_size as f32).sqrt());
        let codes = join_rows(&anchor, &proj.slice(0, 1u64, n));
        let q = codes.matmul(&self.gate);
        let s = codes.matmul(&self.pred_gate);
        let v = codes.matmul(&self.prev);
        let gate = q
            .slice(0, 1u64, n)
            .add(&s.slice(0, 0u64, n))
            .add_vector(&self.gate_bias)
            .act(Act::Sigmoid);
        let delta = gate.mul(&v.slice(0, 0u64, n)).matmul(&self.writes[layer]);
        let rms_tail = rms.slice(0, 1u64, n).expand(1, hidden);
        let out_tail = raw.slice(0, 1u64, n).add(&rms_tail.mul(&delta));
        let out = join_rows(&raw.slice(0, 0u64, 1u64), &out_tail);
        (out, rms, q)
    }

    fn refine(
        &self,
        m: &Model,
        c: &Config,
        cfg: BridgeConfig,
        (norm, head): (&Norm, &Embed),
        (raw, rms, q): (&Tensor, &Tensor, &Tensor),
        (logits, block_tokens): (&Tensor, &Tensor),
    ) -> (Tensor, Tensor) {
        let (n, k, cs, r, h) = (
            c.block_size - 1,
            cfg.top_k,
            cfg.code_size,
            cfg.rank,
            c.stack.hidden,
        );
        let cands = logits.top_k(k);
        let cands_flat = cands.reshape(vec![(n * k).into()]);
        let anchor = self
            .codebook
            .gather(block_tokens)
            .slice(0, 0u64, 1u64)
            .expand(0, k);
        let e = if n > 1 {
            let prev = self
                .codebook
                .gather(&cands_flat)
                .slice(0, 0u64, (n - 1) * k);
            join_rows(&anchor, &prev)
        } else {
            anchor
        }
        .scale((cs as f32).sqrt())
        .reshape(vec![n.into(), k.into(), cs.into()]);
        let q_tail = q
            .slice(0, 1u64, n)
            .reshape(vec![n.into(), 1u64.into(), r.into()])
            .expand(1, k);
        let gate = q_tail
            .add(&e.matmul(&self.pred_gate))
            .add_vector(&self.gate_bias)
            .act(Act::Sigmoid);
        let val = e.matmul(&self.prev);
        let delta = gate
            .mul(&val)
            .matmul(self.writes.last().expect("at least one layer"));
        let rms_tail = rms
            .slice(0, 1u64, n)
            .reshape(vec![n.into(), 1u64.into(), 1u64.into()])
            .expand(1, k)
            .expand(2, h);
        let raw_tail = raw
            .slice(0, 1u64, n)
            .reshape(vec![n.into(), 1u64.into(), h.into()])
            .expand(1, k);
        let refined = raw_tail.add(&rms_tail.mul(&delta)).rms_norm(norm, c.eps);
        let head_rows = m
            .embed(&cands_flat, head)
            .reshape(vec![n.into(), k.into(), h.into()]);
        let scores = refined.matmul_rows(&head_rows);
        (cands_flat, scores)
    }
}

/// Cached DFlash step graph: block plus taps in, drafts out.
pub fn stepper<N: HeadNorm>(c: &Config) -> Graph {
    assert!(c.block_size >= 2, "dflash block_size must be at least 2");
    assert!(
        !c.target_layers.is_empty(),
        "dflash requires at least one target layer"
    );
    let m = Model::new(c.dims());
    let (w, head) = with_head::<Drafter<N>>(&m, c.tied);
    let convs: Option<Vec<[ConvWeights; 2]>> = c.conv.map(|cv| {
        (0..c.stack.layers)
            .map(|i| {
                [
                    ConvWeights::declare(&m, i, "attention_conv", c.stack.hidden, cv),
                    ConvWeights::declare(&m, i, "mlp_conv", c.stack.hidden, cv),
                ]
            })
            .collect()
    });
    let selector = c.selector.map(|sel| SelectorWeights::declare(&m, sel));
    let bridge = c
        .bridge
        .map(|cfg| BridgeWeights::declare(&m, c.stack.layers, cfg));
    let p = entry_ports(&m, c.block_size);
    let ty = TensorType::new([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let kept: Vec<Cache> = w
        .layers
        .iter()
        .map(|_| [m.kept(ty.clone()), m.kept(ty.clone())])
        .collect();
    let ctx = p.ctx_taps.matmul(&w.fc).rms_norm(&w.hidden_norm, c.eps);
    let mut x = m.embed(&p.block_tokens, &w.embed);
    let mut stash: Option<(Tensor, Tensor, Tensor)> = None;
    for (i, l) in w.layers.iter().enumerate() {
        let [raw] = m.composite(Composite::Layer, || {
            let h = x.rms_norm(&l.ln1, c.eps);
            let attn = match (c.conv, &convs) {
                (Some(cv), Some(cs)) => {
                    let (hc, k_attn) = cs[i][0].prepare(&h, c.block_size, c.stack.hidden, cv);
                    let [a] = m.composite(Composite::Attention, || {
                        [attention(&m, c, l, &ctx, &hc, &p, &kept[i])]
                    });
                    cs[i][0].finish(&a, &k_attn, c.block_size, c.stack.hidden, cv)
                }
                _ => m.composite(Composite::Attention, || {
                    [attention(&m, c, l, &ctx, &h, &p, &kept[i])]
                })[0]
                    .clone(),
            };
            let x = x.add(&attn);
            let h = x.rms_norm(&l.ln2, c.eps);
            let mlp = match (c.conv, &convs) {
                (Some(cv), Some(cs)) => {
                    let (hc, k_mlp) = cs[i][1].prepare(&h, c.block_size, c.stack.hidden, cv);
                    let [y] = m.composite(Swiglu::KIND, || [l.mlp.apply(&m, &hc)]);
                    cs[i][1].finish(&y, &k_mlp, c.block_size, c.stack.hidden, cv)
                }
                _ => m.composite(Swiglu::KIND, || [l.mlp.apply(&m, &h)])[0].clone(),
            };
            [x.add(&mlp)]
        });
        if let (Some(b_cfg), Some(b)) = (c.bridge, &bridge) {
            let (out, rms, q) = b.apply(i, &raw, &p.block_tokens, c, b_cfg);
            x = out;
            stash = Some((raw, rms, q));
        } else {
            x = raw;
        }
    }
    let h = x.rms_norm(&w.norm, c.eps);
    let n = c.block_size - 1;
    let draft = h.slice(0, 1u64, n);
    let head = head.as_ref().unwrap_or(&w.embed);
    let logits = draft.unembed(head);
    if let (Some(b_cfg), Some(b), Some((raw, rms, q))) = (c.bridge, &bridge, stash) {
        let (cands, scores) = b.refine(
            &m,
            c,
            b_cfg,
            (&w.norm, head),
            (&raw, &rms, &q),
            (&logits, &p.block_tokens),
        );
        return m.finish([
            (Role::Candidates, &cands),
            (Role::Taps, &scores),
            (Role::Logits, &logits),
        ]);
    }
    if let (Some(sel_cfg), Some(sel)) = (c.selector, &selector) {
        let (cands, scores) = sel.lattice(c, sel_cfg, &draft, &logits, &p.block_tokens);
        return m.finish([
            (Role::Candidates, &cands),
            (Role::Taps, &scores),
            (Role::Logits, &logits),
        ]);
    }
    let ids = logits.top_k(1u64).reshape(vec![DimExpr::Const(n)]);
    m.finish([(Role::Tokens, &ids), (Role::Logits, &logits)])
}
