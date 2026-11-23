//! DeepSeek-style releases: config, weight names, weights.

use crate::names::Rule;
use crate::{each, pins};
use serde::Deserialize;
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::{Offer, Shards};
use tessera_engine::mla::{Config, Index, Moe, RotatedMoe, program, stepper};
use tessera_engine::{Graph, Pins, Stack, Steps, Weight};

/// The `config.json` keys a DeepSeek-V3 release sets.
#[derive(Deserialize)]
struct Hf {
    vocab_size: u64,
    hidden_size: u64,
    num_hidden_layers: u64,
    num_attention_heads: u64,
    q_lora_rank: u64,
    kv_lora_rank: u64,
    qk_nope_head_dim: u64,
    qk_rope_head_dim: u64,
    v_head_dim: u64,
    intermediate_size: u64,
    moe_intermediate_size: u64,
    n_routed_experts: u64,
    num_experts_per_tok: u64,
    first_k_dense_replace: u64,
    routed_scaling_factor: f32,
    /// DeepSeek sets it here, GLM under `rope_parameters`.
    rope_theta: Option<f32>,
    rope_parameters: Option<Rope>,
    rms_norm_eps: f32,
    /// DeepSeek-V3.2 and GLM keys. GLM indexes every `freq`-th layer.
    index_n_heads: Option<u64>,
    index_head_dim: Option<u64>,
    index_topk: Option<u64>,
    index_topk_freq: Option<u64>,
}

#[derive(Deserialize)]
struct Rope {
    rope_theta: f32,
}

/// The model `json` describes: `seqs` sequences of `context` rows.
pub fn config(json: &str, seqs: u64, context: u64) -> Result<Config, serde_json::Error> {
    let h: Hf = serde_json::from_str(json)?;
    let sizes = [
        h.vocab_size,
        h.hidden_size,
        h.num_hidden_layers,
        h.num_attention_heads,
        h.v_head_dim,
        h.moe_intermediate_size,
    ];
    let index = match (h.index_n_heads, h.index_head_dim, h.index_topk) {
        (Some(heads), Some(dim), Some(keep)) => Some(Index {
            heads,
            dim,
            keep,
            every: h.index_topk_freq.unwrap_or(1),
        }),
        _ => None,
    };
    let theta = h.rope_theta.or(h.rope_parameters.map(|r| r.rope_theta));
    let theta = theta.ok_or_else(|| serde::de::Error::missing_field("rope_theta"))?;
    Ok(Config {
        stack: Stack::sized(sizes, seqs, context),
        dense_layers: h.first_k_dense_replace,
        nope: h.qk_nope_head_dim,
        rope: h.qk_rope_head_dim,
        rope_theta: theta,
        q_latent: h.q_lora_rank,
        latent: h.kv_lora_rank,
        dense_ffn: h.intermediate_size,
        experts: h.n_routed_experts,
        top_k: h.num_experts_per_tok,
        route_scale: h.routed_scaling_factor,
        index,
        eps: h.rms_norm_eps,
    })
}

/// Each weight's HF slice. `rotated` names GLM's exlr8 experts.
pub fn rules(c: &Config, rotated: bool) -> Vec<Rule> {
    let mut rules: Vec<Rule> = Vec::new();
    let mut add = |path: String, stem: String| rules.push((path, stem));
    add("embed".into(), "model.embed_tokens".into());
    add("norm".into(), "model.norm".into());
    add("head".into(), "lm_head".into());
    let attention = [
        ("norm", "input_layernorm"),
        ("q_a", "self_attn.q_a_proj"),
        ("q_norm", "self_attn.q_a_layernorm"),
        ("q_b", "self_attn.q_b_proj"),
        ("kv_a", "self_attn.kv_a_proj_with_mqa"),
        ("kv_norm", "self_attn.kv_a_layernorm"),
        ("kv_b", "self_attn.kv_b_proj"),
        ("o", "self_attn.o_proj"),
    ];
    let indexer = [
        ("wq", "wq_b"),
        ("wk", "wk"),
        ("k_norm", "k_norm"),
        ("k_bias", "k_norm.bias"),
        ("weights", "weights_proj"),
    ];
    let mlp = [
        ("gate", "gate_proj"),
        ("up", "up_proj"),
        ("down", "down_proj"),
    ];
    let mut indexers = 0;
    for l in 0..c.stack.layers {
        let at = format!("model.layers.{l}");
        for (field, stem) in attention {
            add(format!("attn.{l}.{field}"), format!("{at}.{stem}"));
        }
        add(
            format!("mlp_norm.{l}"),
            format!("{at}.post_attention_layernorm"),
        );
        if c.indexes(l) {
            for (field, stem) in indexer {
                let path = format!("indexers.{indexers}.{field}");
                add(path, format!("{at}.self_attn.indexer.{stem}"));
            }
            indexers += 1;
        }
        // Routed layers count from the first one.
        let Some(m) = l.checked_sub(c.dense_layers) else {
            for (field, stem) in mlp {
                add(format!("dense.{l}.{field}"), format!("{at}.mlp.{stem}"));
            }
            continue;
        };
        add(format!("moe.{m}.router"), format!("{at}.mlp.gate"));
        let bias = format!("{at}.mlp.gate.e_score_correction_bias");
        add(format!("moe.{m}.router_bias"), bias);
        for side in ["signs_in", "signs_out"].iter().filter(|_| rotated) {
            add(
                format!("moe.{m}.{side}"),
                format!("{at}.mlp.experts.{side}"),
            );
        }
        for (field, stem) in mlp {
            let shared = format!("{at}.mlp.shared_experts.{stem}");
            add(format!("moe.{m}.shared.{field}"), shared);
            if !rotated {
                let experts = format!("{at}.mlp.experts.{{}}.{stem}");
                add(format!("moe.{m}.{field}"), experts);
                continue;
            }
            // One set per projection, and its scales on each side.
            let set = format!("{at}.mlp.experts.{field}");
            for end in ["su", "sv"] {
                add(format!("moe.{m}.{field}_{end}"), format!("{set}.{end}"));
            }
            add(format!("moe.{m}.{field}"), set);
        }
    }
    rules
}

pub struct Release {
    pub config: Config,
    pub graph: Graph,
    /// Forms packed in the file, when any are.
    pub pins: Pins,
    offer: Box<dyn Offer>,
    rules: Vec<Rule>,
}

impl Release {
    /// Each weight in graph order, read one at a time.
    pub fn each(
        &self,
        shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
        mut take: impl FnMut(u32, Weight) -> Result<(), Box<dyn Error + Send + Sync>>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut at = 0;
        each(
            &self.graph,
            &*self.offer,
            &[&self.rules],
            shares,
            |weight, _| {
                at += 1;
                take(at - 1, weight)
            },
        )
    }
}

/// `dir` as safetensors, or exlr8 with experts at `width`.
pub fn release(
    dir: &Path,
    seqs: u64,
    context: u64,
    width: u32,
) -> Result<Release, Box<dyn Error + Send + Sync>> {
    open(dir, None, [seqs, context], width, None, None)
}

/// The step graph of `dir`. The `user` rules go last.
pub(crate) fn stepping(
    (dir, json, manifest): (&Path, &str, Option<&str>),
    sizes: [u64; 2],
    width: u32,
    user: &[Rule],
    steps: Steps,
    taps: &[u64],
) -> Result<Release, Box<dyn Error + Send + Sync>> {
    open(
        dir,
        Some(json),
        sizes,
        width,
        manifest,
        Some((user, steps, taps)),
    )
}

/// A `step` sets the graph's rules, mask and taps.
fn open(
    dir: &Path,
    merged: Option<&str>,
    [seqs, context]: [u64; 2],
    width: u32,
    manifest: Option<&str>,
    step: Option<(&[Rule], Steps, &[u64])>,
) -> Result<Release, Box<dyn Error + Send + Sync>> {
    let owned;
    let json = match merged {
        Some(s) => s,
        None => {
            owned = std::fs::read_to_string(dir.join("config.json"))?;
            &owned
        }
    };
    let mut config = config(json, seqs, context)?;
    let exlr8 = dir.join("dense/exlr8-vocab.safetensors").is_file();
    let mut rules = rules(&config, exlr8);
    if let Some((user, steps, taps)) = step {
        config.stack.steps = steps;
        config.stack.taps = taps.to_vec();
        rules.extend(user.iter().cloned());
    }
    let graph = match (exlr8, step.is_some()) {
        (true, true) => stepper::<RotatedMoe>(&config),
        (true, false) => program::<RotatedMoe>(&config),
        (false, true) => stepper::<Moe>(&config),
        (false, false) => program::<Moe>(&config),
    };
    let offer: Box<dyn Offer> = match exlr8 {
        true => Box::new(crate::release(dir, manifest)?.width(width)),
        false => Box::new(Shards::open(dir)?),
    };
    let pins = pins(&graph, &*offer, config.stack.layers, &[&rules])?;
    Ok(Release {
        config,
        graph,
        pins,
        offer,
        rules,
    })
}
