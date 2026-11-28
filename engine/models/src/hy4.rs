//! Tencent Hy4-flash: 4-stream iHC, gated MLA with sinks, LatentMoE.

use crate::keys::{At, Keys};
use crate::shared::{
    CalibTaps, NOPE, ROPE, Stack, Steps, Swiglu, cat, declare_parts, each_part, part, rows,
};
use moe::LatentMoe;

mod moe;
pub use moe::{LatentExperts, PlainLatent, RotatedLatent};
use serde_json::json;
use std::marker::PhantomData;
use tessera_trace::DimName::*;
use tessera_trace::hf::{
    ATTENTION_BIAS, EPS, HEADS, HIDDEN, HIDDEN_ACT, LAYERS, TIED, TOP_K, VOCAB,
};
use tessera_trace::{
    Act, CalibTap, Composite, DType, Dim, DimExpr, DimName, Embed, EntryKind, Graph, Kept, Linear,
    Mask, Model, Norm, Pairs, PathSeg, Port, Role, Shape, Size, Tensor, TensorType, Vector,
    Weights,
};
use tessera_trace::{pick, step_ports_with, tokens};

const Q_LATENT: DimName = DimName::named("QLatent");
const LATENT: DimName = DimName::named("Latent");
const SWA_HEADS: DimName = DimName::named("SwaHeads");
const DENSE_FFN: DimName = DimName::named("DenseFfn");
const ROUTED_HIDDEN: DimName = DimName::named("RoutedHidden");
const HC: DimName = DimName::named("Hc");
const HC_MIX: DimName = DimName::named("HcMix");
const HC_SCALE: DimName = DimName::named("HcScale");
const DSA_LAYERS: DimName = DimName::named("DsaLayers");
const SWA_LAYERS: DimName = DimName::named("SwaLayers");
const DENSE_LAYERS: DimName = DimName::named("DenseLayers");
const MOE_LAYERS: DimName = DimName::named("MoeLayers");
const ENGRAM_LAYERS: DimName = DimName::named("EngramLayers");
const ENGRAM_HEADS: DimName = DimName::named("EngramHeads");
const ENGRAM_HEAD_DIM: DimName = DimName::named("EngramHeadDim");
const ENGRAM_DIM: DimName = DimName::named("EngramDim");

/// Epsilon for Engram RMSNorm layers.
const ENGRAM_EPS: f32 = f32::EPSILON;

/// Multi-layer Engram lookup dimensions.
#[derive(Clone, Debug)]
pub struct Engram {
    /// Zero-based layer indices receiving Engram injection.
    pub layers: Vec<u64>,
    pub heads: u64,
    pub head_dim: u64,
}

impl Engram {
    pub fn dim(&self) -> u64 {
        self.heads * self.head_dim
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    /// DSA heads in `heads`, `v_head_dim` in `head_dim`.
    pub stack: Stack,
    pub swa_heads: u64,
    pub swa_window: u64,
    /// True for SWA layers, false for DSA layers.
    pub swa_pattern: Vec<bool>,
    pub dense_layers: u64,
    pub dense_ffn: u64,
    pub nope: u64,
    pub rope: u64,
    pub q_latent: u64,
    pub latent: u64,
    pub rope_theta_dsa: f32,
    pub rope_theta_swa: f32,
    pub hc_mult: u64,
    pub hc_eps: f32,
    pub hc_magnitude: f32,
    pub experts: u64,
    pub top_k: u64,
    pub routed_hidden: u64,
    pub route_scale: f32,
    pub swiglu_limit: Option<f32>,
    pub eps: f32,
    pub engram: Option<Engram>,
    pub tied: bool,
}

impl Config {
    /// Parses a Hugging Face `hy_v4_flash` `config.json`.
    pub fn read(k: &mut Keys, max_seqs: u64, max_context: u64) -> Self {
        let at = At::text_or_top();
        for (key, want) in [
            (HIDDEN_ACT.name, json!("silu")),
            (ATTENTION_BIAS.name, json!(false)),
        ] {
            k.fix(&at, key, want);
        }
        let [vocab, hidden, layers, heads] =
            [VOCAB, HIDDEN, LAYERS, HEADS].map(|key| k.read(&at, key));
        let head_dim: u64 = k.read(&at, "v_head_dim");
        let ffn: u64 = k.read(&at, "moe_intermediate_size");
        let raw_pat: Vec<u64> = k.read(&at, "swa_pattern");
        let swa_pattern: Vec<bool> = raw_pat.iter().map(|&v| v != 0).collect();
        if swa_pattern.len() != layers as usize {
            k.misfit(
                &at.paths("swa_pattern")[0],
                format!("swa_pattern len {} != layers {layers}", swa_pattern.len()),
            );
        }
        let dsa_theta = read_rope_theta(k, &at, "deepseek_sparse_attention", 1e7);
        let swa_theta = read_rope_theta(k, &at, "sliding_attention", 1e4);
        let limit: f32 = k.read_or(&at, "swiglu_limit", 10.0);
        let engram_ids: Vec<u64> = k.read_or(&at, "engram_layer_ids", Vec::new());
        let engram = (!engram_ids.is_empty()).then(|| {
            let max_ng: u64 = k.read_or(&at, "engram_max_ngram_size", 3);
            let per_ng: u64 = k.read_or(&at, "engram_n_head_per_ngram", 8);
            let embed_per_ng: u64 = k.read_or(&at, "engram_n_embed_per_ngram", 768);
            let orders = max_ng.saturating_sub(1).max(1);
            let heads = per_ng * orders;
            let head_dim = (embed_per_ng * orders) / heads.max(1);
            Engram {
                layers: engram_ids
                    .into_iter()
                    .map(|id| id.saturating_sub(1))
                    .collect(),
                heads,
                head_dim,
            }
        });
        Self {
            stack: Stack::sized(
                [vocab, hidden, layers, heads, head_dim, ffn],
                max_seqs,
                max_context,
            ),
            swa_heads: k.read(&at, "swa_num_attention_heads"),
            swa_window: k.read_or(&at, "sliding_window", 512),
            swa_pattern,
            dense_layers: k.read_or(&at, "first_k_dense_replace", 1),
            dense_ffn: k.read(&at, "intermediate_size"),
            nope: k.read(&at, "qk_nope_head_dim"),
            rope: k.read(&at, "qk_rope_head_dim"),
            q_latent: k.read(&at, "q_lora_rank"),
            latent: k.read(&at, "kv_lora_rank"),
            rope_theta_dsa: dsa_theta,
            rope_theta_swa: swa_theta,
            hc_mult: k.read_or(&at, "hc_mult", 4),
            hc_eps: k.read_or(&at, "hc_eps", 1e-6),
            hc_magnitude: k.read_or(&at, "hc_magnitude", 2.0),
            experts: k.read(&at, "n_routed_experts"),
            top_k: k.read(&at, TOP_K),
            routed_hidden: k.read(&at, "routed_expert_hidden_size"),
            route_scale: k.read_or(&at, "routed_scaling_factor", 1.0),
            swiglu_limit: (limit > 0.0).then_some(limit),
            eps: k.read(&at, EPS),
            engram,
            tied: k.read_or(&at, TIED, false),
        }
    }

    /// Hy4-flash at its released sizes.
    pub fn hy4_flash(max_seqs: u64, max_context: u64) -> Self {
        let swa_pattern: Vec<bool> = (0..42).map(|l| !(l == 0 || (l - 1) % 4 == 0)).collect();
        Self {
            stack: Stack::sized([120_832, 3072, 42, 32, 256, 1536], max_seqs, max_context),
            swa_heads: 64,
            swa_window: 512,
            swa_pattern,
            dense_layers: 1,
            dense_ffn: 13_824,
            nope: 192,
            rope: 64,
            q_latent: 1024,
            latent: 512,
            rope_theta_dsa: 1e7,
            rope_theta_swa: 1e4,
            hc_mult: 4,
            hc_eps: 1e-6,
            hc_magnitude: 2.0,
            experts: 512,
            top_k: 16,
            routed_hidden: 1536,
            route_scale: 3.998,
            swiglu_limit: Some(10.0),
            eps: 1e-5,
            engram: Some(Engram {
                layers: vec![1, 11, 21, 31],
                heads: 16,
                head_dim: 96,
            }),
            tied: false,
        }
    }

    /// Small two-layer Hy4-flash for unit and parity tests.
    pub fn hy4_small() -> Self {
        Self {
            stack: Stack::sized([32, 16, 2, 2, 8, 8], 2, 16),
            swa_heads: 4,
            swa_window: 4,
            swa_pattern: vec![false, true],
            dense_ffn: 16,
            nope: 4,
            rope: 4,
            q_latent: 8,
            latent: 8,
            rope_theta_dsa: 1e4,
            rope_theta_swa: 1e3,
            experts: 4,
            top_k: 2,
            routed_hidden: 8,
            route_scale: 2.0,
            engram: None,
            ..Self::hy4_flash(2, 16)
        }
    }

    fn dsa_count(&self) -> u64 {
        self.swa_pattern.iter().filter(|&&s| !s).count() as u64
    }

    fn swa_count(&self) -> u64 {
        self.swa_pattern.iter().filter(|&&s| s).count() as u64
    }

    pub fn engram_slot(&self, layer: usize) -> Option<usize> {
        let eng = self.engram.as_ref()?;
        eng.layers.iter().position(|&l| l as usize == layer)
    }

    pub fn dims(&self) -> Vec<Dim> {
        let moe = self.stack.layers.saturating_sub(self.dense_layers);
        let mut own = vec![
            (SWA_HEADS, self.swa_heads),
            (Q_LATENT, self.q_latent),
            (LATENT, self.latent),
            (NOPE, self.nope),
            (ROPE, self.rope),
            (DENSE_FFN, self.dense_ffn),
            (ROUTED_HIDDEN, self.routed_hidden),
            (HC, self.hc_mult),
            (HC_MIX, self.hc_mult * 2),
            (HC_SCALE, 2),
            (Experts, self.experts),
            (TopK, self.top_k),
            (DSA_LAYERS, self.dsa_count()),
            (SWA_LAYERS, self.swa_count()),
            (DENSE_LAYERS, self.dense_layers),
            (MOE_LAYERS, moe),
        ];
        if let Some(eng) = &self.engram {
            own.extend([
                (ENGRAM_LAYERS, eng.layers.len() as u64),
                (ENGRAM_HEADS, eng.heads),
                (ENGRAM_HEAD_DIM, eng.head_dim),
                (ENGRAM_DIM, eng.dim()),
            ]);
        }
        self.stack.dims(&own)
    }
}

fn read_rope_theta(k: &mut Keys, at: &At, sub: &str, default: f32) -> f32 {
    let key = format!("rope_parameters.{sub}.rope_theta");
    k.read_or(at, key.as_str(), default)
}

/// Non-combined 4-stream iHC pre-mixer and post-injector.
#[derive(Weights)]
struct HyperConn {
    #[shape(HC * Hidden, HC_MIX)]
    fn_w: Linear,
    #[shape(HC_MIX)]
    base: Vector,
    #[shape(HC_SCALE)]
    scale: Vector,
}

impl HyperConn {
    fn pre(&self, hyper: &Tensor, lead: &[DimExpr], c: &Config) -> [Tensor; 2] {
        let axis = lead.len() as u32;
        let flat = hyper.reshape(cat(lead, [HC * Hidden]));
        let mixes = flat.rms(c.eps).matmul(&self.fn_w);
        let swap: Vec<u32> = (0..axis).chain([axis + 1, axis]).collect();
        let scaled = mixes
            .reshape(cat(lead, [HC_SCALE.into(), HC.into()]))
            .permute(&swap)
            .mul_vector(&self.scale)
            .permute(&swap)
            .reshape(cat(lead, [HC_MIX.into()]))
            .add_vector(&self.base);
        let pre = scaled
            .slice(axis, 0u64, HC)
            .act(Act::Sigmoid)
            .shift(c.hc_eps);
        let post = scaled
            .slice(axis, HC, HC)
            .act(Act::Sigmoid)
            .scale(c.hc_magnitude)
            .shift(c.hc_eps);
        [hyper.mix(axis, &pre), post]
    }

    fn post(hyper: &Tensor, y: &Tensor, post: &Tensor, lead: &[DimExpr]) -> Tensor {
        let axis = lead.len() as u32;
        let y_exp = y
            .reshape(cat(lead, [1u64.into(), Hidden.into()]))
            .expand(axis, HC);
        let p_exp = post
            .reshape(cat(lead, [HC.into(), 1u64.into()]))
            .expand(axis + 1, Hidden);
        hyper.add(&y_exp.mul(&p_exp))
    }
}

/// Final iHC collapse before output norm.
#[derive(Weights)]
struct HyperHead {
    #[shape(HC * Hidden, HC)]
    fn_w: Linear,
    #[shape(HC)]
    base: Vector,
    #[shape(1)]
    scale: Vector,
}

impl HyperHead {
    fn apply(&self, hyper: &Tensor, lead: &[DimExpr], c: &Config) -> Tensor {
        let axis = lead.len() as u32;
        let flat = hyper.reshape(cat(lead, [HC * Hidden]));
        let mixes = flat
            .rms(c.eps)
            .matmul(&self.fn_w)
            .reshape(cat(lead, [HC.into(), 1u64.into()]))
            .mul_vector(&self.scale)
            .reshape(cat(lead, [HC.into()]));
        let pre = mixes
            .add_vector(&self.base)
            .act(Act::Sigmoid)
            .shift(c.hc_eps);
        hyper.mix(axis, &pre)
    }
}

/// Engram n-gram gated injection into hyper-streams.
struct EngramBlock {
    head_norms: Tensor,
    val_w: Linear,
    val_b: Vector,
    key_w: Vec<Linear>,
    key_b: Vec<Vector>,
    norm_key: Vec<Norm>,
    norm_query: Vec<Norm>,
    pre_gamma: Vector,
    hc_scales: Vec<Vector>,
}

impl EngramBlock {
    fn declare(m: &Model, p: &mut Vec<PathSeg>, c: &Config) -> Self {
        let hid: DimExpr = Hidden.into();
        let edim: DimExpr = ENGRAM_DIM.into();
        let hc = c.hc_mult as usize;
        p.push(PathSeg::Field("head_norms".into()));
        let head_norms = m.weight(
            p,
            TensorType::new(
                Shape(vec![ENGRAM_HEADS.into(), ENGRAM_HEAD_DIM.into()]),
                DType::F32,
            ),
        );
        p.pop();
        Self {
            head_norms,
            val_w: part(m, p, "val_w", &[edim.clone(), hid.clone()]),
            val_b: part(m, p, "val_b", std::slice::from_ref(&hid)),
            key_w: declare_parts(m, p, "key_w", hc, &[edim, hid.clone()]),
            key_b: declare_parts(m, p, "key_b", hc, std::slice::from_ref(&hid)),
            norm_key: declare_parts(m, p, "norm_key", hc, std::slice::from_ref(&hid)),
            norm_query: declare_parts(m, p, "norm_query", hc, std::slice::from_ref(&hid)),
            pre_gamma: part(m, p, "pre_gamma", std::slice::from_ref(&hid)),
            hc_scales: declare_parts(m, p, "hc_scales", hc, &[hid]),
        }
    }

    fn apply(&self, c: &Config, hyper: &Tensor, raw: &Tensor, lead: &[DimExpr]) -> Tensor {
        let axis = lead.len() as u32;
        let head_shape = cat(lead, [ENGRAM_HEADS.into(), ENGRAM_HEAD_DIM.into()]);
        let emb = raw
            .reshape(head_shape)
            .rms(ENGRAM_EPS)
            .mul(&self.head_norms)
            .reshape(cat(lead, [ENGRAM_DIM.into()]));
        let val = emb.matmul(&self.val_w).add_vector(&self.val_b);
        let main_sig = val.rms(ENGRAM_EPS).mul_vector(&self.pre_gamma);

        let flat = cat(lead, [Hidden.into()]);
        let scalar = cat(lead, [1u64.into()]);
        let scale = (c.stack.hidden as f32).sqrt();
        let mut joined: Option<Tensor> = None;
        for hc in 0..c.hc_mult as usize {
            let key = emb.matmul(&self.key_w[hc]).add_vector(&self.key_b[hc]);
            let key_n = key.rms_norm(&self.norm_key[hc], ENGRAM_EPS);
            let h_hc = hyper.slice(axis, hc as u64, 1u64).reshape(flat.clone());
            let q_n = h_hc.rms_norm(&self.norm_query[hc], ENGRAM_EPS);
            let gate = key_n
                .mul(&q_n)
                .mean(axis)
                .scale(scale)
                .act(Act::SignedSqrt)
                .act(Act::Sigmoid)
                .reshape(scalar.clone())
                .expand(axis, Hidden);
            let branch = main_sig.mul_vector(&self.hc_scales[hc]);
            let out = h_hc.add(&main_sig.mul(&gate)).add(&branch);
            joined = Some(match joined {
                None => out,
                Some(prev) => prev.concat(&out),
            });
        }
        joined
            .expect("hc_mult > 0")
            .reshape(cat(lead, [HC.into(), Hidden.into()]))
    }
}

struct Dsa;
struct Swa;

trait HeadCount {
    const DIM: DimName;
}

impl HeadCount for Dsa {
    const DIM: DimName = Heads;
}

impl HeadCount for Swa {
    const DIM: DimName = SWA_HEADS;
}

/// Gated absorbed MLA block with per-head sinks.
struct GatedMla<H: HeadCount> {
    input_norm: Norm,
    q_a: Linear,
    q_norm: Norm,
    q_b: Linear,
    kv_a: Linear,
    kv_norm: Norm,
    kv_b: Linear,
    linear_gate: Linear,
    o: Linear,
    post_proj_norm: Norm,
    sinks: Vector,
    _heads: PhantomData<H>,
}

impl<H: HeadCount> Weights for GatedMla<H> {
    fn declare(m: &Model, path: &mut Vec<PathSeg>, _shape: &[DimExpr]) -> Self {
        let nh: DimExpr = H::DIM.into();
        let hid: DimExpr = Hidden.into();
        let q_lat: DimExpr = Q_LATENT.into();
        let lat: DimExpr = LATENT.into();
        let qk = nh.clone() * (NOPE + ROPE);
        let kvb = nh.clone() * (NOPE + HeadDim);
        let vo = nh.clone() * HeadDim;
        Self {
            input_norm: part(m, path, "input_norm", std::slice::from_ref(&hid)),
            q_a: part(m, path, "q_a", &[hid.clone(), q_lat.clone()]),
            q_norm: part(m, path, "q_norm", std::slice::from_ref(&q_lat)),
            q_b: part(m, path, "q_b", &[q_lat, qk]),
            kv_a: part(m, path, "kv_a", &[hid.clone(), LATENT + ROPE]),
            kv_norm: part(m, path, "kv_norm", std::slice::from_ref(&lat)),
            kv_b: part(m, path, "kv_b", &[lat, kvb]),
            linear_gate: part(m, path, "linear_gate", &[hid.clone(), vo.clone()]),
            o: part(m, path, "o", &[vo, hid.clone()]),
            post_proj_norm: part(m, path, "post_proj_norm", std::slice::from_ref(&hid)),
            sinks: part(m, path, "sinks", &[nh]),
            _heads: PhantomData,
        }
    }
}

impl<H: HeadCount> GatedMla<H> {
    #[allow(clippy::too_many_arguments)]
    fn apply(
        &self,
        m: &Model,
        c: &Config,
        h_in: &Tensor,
        positions: &Tensor,
        lead: &[DimExpr],
        mask: Mask,
        theta: f32,
        step: Option<(&Kept, &Tensor)>,
        taps: &CalibTaps,
    ) -> Tensor {
        let nh = H::DIM;
        let feat = lead.len() as u32;
        let head_ax = feat + 1;
        let h = h_in.rms_norm(&self.input_norm, c.eps);
        let qr = h.matmul(&self.q_a).rms_norm(&self.q_norm, c.eps);
        taps.note(CalibTap::QResid, &qr);
        let q = qr
            .matmul(&self.q_b)
            .reshape(cat(lead, [nh.into(), NOPE + ROPE]));
        let q_nope = q.slice(head_ax, 0u64, NOPE);
        let q_rope = q
            .slice(head_ax, NOPE, ROPE)
            .rope(positions, theta, Pairs::Adjacent);
        let kv = h.matmul(&self.kv_a);
        let latent = kv.slice(feat, 0u64, LATENT).rms_norm(&self.kv_norm, c.eps);
        taps.note(CalibTap::KvLatent, &latent);
        let shared = kv
            .slice(feat, LATENT, ROPE)
            .rope(positions, theta, Pairs::Adjacent);
        let scale = 1.0 / ((c.nope + c.rope) as f32).sqrt();
        let sinks = self.sinks.stored();
        let out = match step {
            None => {
                let q_cat = q_nope.concat(&q_rope);
                let per_head = latent
                    .matmul(&self.kv_b)
                    .reshape(cat(lead, [nh.into(), NOPE + HeadDim]));
                let sh_exp = shared
                    .reshape(cat(lead, [1u64.into(), ROPE.into()]))
                    .expand(feat, nh);
                let k = per_head.slice(head_ax, 0u64, NOPE).concat(&sh_exp);
                let v = per_head.slice(head_ax, NOPE, HeadDim);
                m.attend_sinks(mask, scale, [&q_cat, &k, &v], None, sinks)
            }
            Some((kv_cache, slots)) => {
                let by = [slots, positions];
                let kv_row = latent
                    .concat(&shared)
                    .reshape(cat(lead, [1u64.into(), LATENT + ROPE]));
                let k_all = kv_row.append(kv_cache, by);
                let set = || {
                    self.kv_b
                        .stored()
                        .reshape(vec![nh.into(), NOPE + HeadDim, LATENT.into()])
                };
                let swap = [1, 0, 2];
                let to_lat = set().slice(1, 0u64, NOPE).permute(&[0, 2, 1]);
                let q_abs = q_nope
                    .permute(&swap)
                    .matmul_rows(&to_lat)
                    .permute(&swap)
                    .concat(&q_rope);
                let raw =
                    m.attend_rows_sinks(mask, scale, [&q_abs, &k_all, &k_all], by, None, sinks);
                let to_val = set().slice(1, NOPE, HeadDim);
                raw.slice(2, 0u64, LATENT)
                    .permute(&swap)
                    .matmul_rows(&to_val)
                    .permute(&swap)
            }
        };
        let gate = h.matmul(&self.linear_gate).act(Act::Sigmoid);
        let o_in = out.reshape(cat(lead, [nh * HeadDim])).mul(&gate);
        taps.note(CalibTap::OIn, &o_in);
        o_in.matmul(&self.o).rms_norm(&self.post_proj_norm, c.eps)
    }
}

/// Layer-0 dense SwiGLU MLP with post-FC2 norm.
#[derive(Weights)]
struct DenseMlp {
    #[shape(Hidden)]
    post_attn_norm: Norm,
    #[shape(Hidden, DENSE_FFN)]
    mlp: Swiglu,
    #[shape(Hidden)]
    post_fc2_norm: Norm,
}

struct Hy4<E: LatentExperts> {
    embed: Embed,
    engram: Vec<EngramBlock>,
    attn_hc: Vec<HyperConn>,
    mlp_hc: Vec<HyperConn>,
    dsa: Vec<GatedMla<Dsa>>,
    swa: Vec<GatedMla<Swa>>,
    dense: Vec<DenseMlp>,
    moe: Vec<LatentMoe<E>>,
    hc_head: HyperHead,
    norm: Norm,
    head: Option<Linear>,
}

impl<E: LatentExperts> Hy4<E> {
    fn declare(m: &Model, c: &Config) -> Self {
        let mut p = Vec::new();
        let hid: DimExpr = Hidden.into();
        let layers = c.stack.layers as usize;
        let eng_n = c.engram.as_ref().map_or(0, |e| e.layers.len());
        let moe_n = c.stack.layers.saturating_sub(c.dense_layers) as usize;
        Self {
            embed: part(m, &mut p, "embed", &[Vocab.into(), hid.clone()]),
            engram: each_part(&mut p, "engram", eng_n, |p| EngramBlock::declare(m, p, c)),
            attn_hc: declare_parts(m, &mut p, "attn_hc", layers, &[]),
            mlp_hc: declare_parts(m, &mut p, "mlp_hc", layers, &[]),
            dsa: declare_parts(m, &mut p, "dsa", c.dsa_count() as usize, &[]),
            swa: declare_parts(m, &mut p, "swa", c.swa_count() as usize, &[]),
            dense: declare_parts(m, &mut p, "dense", c.dense_layers as usize, &[]),
            moe: declare_parts(m, &mut p, "moe", moe_n, &[]),
            hc_head: part(m, &mut p, "hc_head", &[]),
            norm: part(m, &mut p, "norm", std::slice::from_ref(&hid)),
            head: (!c.tied).then(|| part(m, &mut p, "head", &[hid, Vocab.into()])),
        }
    }
}

struct StepState<'a> {
    slots: &'a Tensor,
    taps: &'a [u64],
    kv: Vec<Kept>,
}

fn run<E: LatentExperts>(
    m: &Model,
    c: &Config,
    w: &Hy4<E>,
    [tokens, positions]: [&Tensor; 2],
    ple: Option<&Tensor>,
    step: Option<&StepState>,
    calib: &CalibTaps,
) -> (Tensor, Option<Tensor>) {
    let lead: Vec<DimExpr> = match step {
        Some(_) => vec![Rows.into()],
        None => rows([]),
    };
    let axis = lead.len() as u32;
    let emb = m.embed(tokens, &w.embed);
    let mut hyper = emb
        .reshape(cat(&lead, [1u64.into(), Hidden.into()]))
        .expand(axis, HC);
    let swa_mask = Mask::Back(c.swa_window.saturating_sub(1));
    let (mut di, mut si, mut mi) = (0usize, 0usize, 0usize);
    let mut tapped = Vec::new();

    let mut body = || {
        for l in 0..c.stack.layers as usize {
            let [next] = m.composite(Composite::Layer, || {
                let mut h = hyper.clone();
                if let (Some(slot), Some(ple_t)) = (c.engram_slot(l), ple) {
                    let off = slot as u64 * c.engram.as_ref().expect("engram configured").dim();
                    h = w.engram[slot].apply(c, &h, &ple_t.slice(axis, off, ENGRAM_DIM), &lead);
                }
                let [a_in, a_post] = w.attn_hc[l].pre(&h, &lead, c);
                let cache = step.map(|st| (&st.kv[l], st.slots));
                let [a_out] = m.composite(Composite::Attention, || {
                    if c.swa_pattern[l] {
                        let block = &w.swa[si];
                        si += 1;
                        [block.apply(
                            m,
                            c,
                            &a_in,
                            positions,
                            &lead,
                            swa_mask,
                            c.rope_theta_swa,
                            cache,
                            calib,
                        )]
                    } else {
                        let block = &w.dsa[di];
                        di += 1;
                        [block.apply(
                            m,
                            c,
                            &a_in,
                            positions,
                            &lead,
                            Mask::Causal,
                            c.rope_theta_dsa,
                            cache,
                            calib,
                        )]
                    }
                });
                h = HyperConn::post(&h, &a_out, &a_post, &lead);
                let [m_in, m_post] = w.mlp_hc[l].pre(&h, &lead, c);
                let [m_out] = if (l as u64) < c.dense_layers {
                    let d = &w.dense[l];
                    m.composite(Composite::Mlp, || {
                        let normed = m_in.rms_norm(&d.post_attn_norm, c.eps);
                        [d.mlp.eval(&normed).rms_norm(&d.post_fc2_norm, c.eps)]
                    })
                } else {
                    let moe = &w.moe[mi];
                    mi += 1;
                    m.composite(Composite::Moe, || [moe.apply(m, c, &m_in, calib)])
                };
                [HyperConn::post(&h, &m_out, &m_post, &lead)]
            });
            hyper = next;
            if let Some(st) = step
                && st.taps.contains(&(l as u64 + 1))
            {
                tapped.push(w.hc_head.apply(&hyper, &lead, c));
            }
        }
        let hidden = w.hc_head.apply(&hyper, &lead, c).rms_norm(&w.norm, c.eps);
        match &w.head {
            Some(head) => hidden.matmul(head),
            None => hidden.unembed(&w.embed),
        }
    };
    let [logits] = m.composite(Composite::Model, || [body()]);
    let tapped = tapped.into_iter().reduce(|a, b| a.concat(&b));
    (logits, tapped)
}

fn port(role: Role, shape: impl Into<tessera_trace::Shape>, dtype: DType) -> Port {
    Port {
        role,
        ty: TensorType::new(shape, dtype),
    }
}

/// Whole-sequence forward graph.
pub fn program<E: LatentExperts>(c: &Config) -> Graph {
    let m = Model::new(c.dims());
    let w = Hy4::<E>::declare(&m, c);
    let (tokens, positions, ple) = if c.engram.is_some() {
        let tp = port(Role::Tokens, [Seqs, Context], DType::U32);
        let pp = port(Role::Positions, [Seqs, Context], DType::U32);
        let ep = port(
            Role::Ple,
            vec![Seqs.into(), Context.into(), ENGRAM_LAYERS * ENGRAM_DIM],
            DType::F32,
        );
        let [t, p, ple] = m.entry(EntryKind::Forward, [tp, pp, ep]);
        (t, p, Some(ple))
    } else {
        let [t, p] = tokens(&m);
        (t, p, None)
    };
    let calib = CalibTaps::default();
    let (logits, _) = run(&m, c, &w, [&tokens, &positions], ple.as_ref(), None, &calib);
    calib.finish(&m, &logits)
}

/// Ragged step graph with kept MLA latent caches.
pub fn stepper<E: LatentExperts>(c: &Config) -> Graph {
    let mut dims = c.dims();
    dims.push(Dim {
        name: Rows,
        size: Size::Bounded(c.stack.max_context),
    });
    let m = Model::new(dims);
    let w = Hy4::<E>::declare(&m, c);
    let ple = c.engram.is_some().then(|| {
        let shape = vec![Rows.into(), ENGRAM_LAYERS * ENGRAM_DIM];
        port(Role::Ple, shape, DType::F32)
    });
    let masked = c.stack.steps == Steps::Masked;
    let ([tokens, positions, slots], ple_in, allowed) = step_ports_with(&m, masked, ple);
    let kv_ty = TensorType::new(rows([DimExpr::Const(1), LATENT + ROPE]), DType::F32);
    let kv = (0..c.stack.layers).map(|_| m.kept(kv_ty.clone())).collect();
    let state = StepState {
        slots: &slots,
        taps: &c.stack.taps,
        kv,
    };
    let (logits, taps) = run(
        &m,
        c,
        &w,
        [&tokens, &positions],
        ple_in.as_ref(),
        Some(&state),
        &CalibTaps::default(),
    );
    pick(&m, &logits, allowed.as_ref(), taps.as_ref())
}

crate::preset::traced!("hy4", Config::hy4_small, Config::hy4_flash, [PlainLatent]);
