//! DFlash block-diffusion drafter conditioned on tapped target states.

use crate::decoder::{HeadNorm, Layer, Mlp, Swiglu};
use crate::keys::Keys;
use crate::shared::{Stack, join_rows, with_head};
use serde_json::json;
use tessera_trace::DimName::*;
use tessera_trace::{
    Composite, DType, Dim, DimExpr, Embed, EntryKind, Graph, Kept, Linear, Mask, Model, Norm,
    Pairs, Port, Role, Size, Tensor, TensorType, Weights,
};

/// Target model dimensions a drafter inherits when omitted.
#[derive(Clone, Copy, Debug)]
pub struct TargetSpec {
    pub vocab: u64,
    pub hidden: u64,
    pub layers: u64,
    pub tied: bool,
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
    let attn = m.attend_rows(
        Mask::Causal,
        scale,
        [&q, &k, &v],
        [&p.block_slots, &p.attend_pos],
        None,
    );
    attn.reshape(vec![blk, Heads * HeadDim]).matmul(&l.o)
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
    let p = entry_ports(&m, c.block_size);
    let ty = TensorType::new([Seqs, Context, KvHeads, HeadDim], DType::F32);
    let kept: Vec<Cache> = w
        .layers
        .iter()
        .map(|_| [m.kept(ty.clone()), m.kept(ty.clone())])
        .collect();
    let ctx = p.ctx_taps.matmul(&w.fc).rms_norm(&w.hidden_norm, c.eps);
    let mut x = m.embed(&p.block_tokens, &w.embed);
    for (i, l) in w.layers.iter().enumerate() {
        let [out] = m.composite(Composite::Layer, || {
            let h = x.rms_norm(&l.ln1, c.eps);
            let [attn] = m.composite(Composite::Attention, || {
                [attention(&m, c, l, &ctx, &h, &p, &kept[i])]
            });
            let x = x.add(&attn);
            let h = x.rms_norm(&l.ln2, c.eps);
            let [mlp] = m.composite(Swiglu::KIND, || [l.mlp.apply(&m, &h)]);
            [x.add(&mlp)]
        });
        x = out;
    }
    let h = x.rms_norm(&w.norm, c.eps);
    let draft = h.slice(0, 1u64, c.block_size - 1);
    let head = head.as_ref().unwrap_or(&w.embed);
    let logits = draft.unembed(head);
    let ids = logits
        .top_k(1u64)
        .reshape(vec![DimExpr::Const(c.block_size - 1)]);
    m.finish([(Role::Tokens, &ids), (Role::Logits, &logits)])
}
