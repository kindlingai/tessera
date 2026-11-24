//! DeepSeek-style releases: config, weight names, weights.

use crate::arch::{Arch, Ask, Built, Spot, arch};
use crate::names::Rule;
use crate::{Stream, Weights, is_exlr8, pins, stream_each};
use serde::Deserialize;
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::Offer;
use tessera_engine::mla::{Config, Index, Moe, RotatedMoe, program, stepper};
use tessera_engine::{Graph, Pins, Stack, Steps, Weight};

/// Whether `json` reads as a DeepSeek-style config.
fn probe(json: &str) -> bool {
    config(json, 1, 1).is_ok()
}

fn step(_: &Arch, ask: &Ask) -> Result<Built, Box<dyn Error + Send + Sync>> {
    let step = Some((ask.user, ask.steps, ask.taps));
    let (dir, json) = (ask.dir, Some(ask.json));
    let release = open(dir, json, ask.sizes, None, ask.manifest, step)?;
    let (graph, pins) = (release.graph.clone(), release.pins.clone());
    Ok((graph, Weights::Streamed(Box::new(release)), pins))
}

arch!("mla", [Spot::Probe(probe)], step: step);

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

const ATTENTION: [(&str, &str); 8] = [
    ("norm", "input_layernorm"),
    ("q_a", "self_attn.q_a_proj"),
    ("q_norm", "self_attn.q_a_layernorm"),
    ("q_b", "self_attn.q_b_proj"),
    ("kv_a", "self_attn.kv_a_proj_with_mqa"),
    ("kv_norm", "self_attn.kv_a_layernorm"),
    ("kv_b", "self_attn.kv_b_proj"),
    ("o", "self_attn.o_proj"),
];

const INDEXER: [(&str, &str); 5] = [
    ("wq", "wq_b"),
    ("wk", "wk"),
    ("k_norm", "k_norm"),
    ("k_bias", "k_norm.bias"),
    ("weights", "weights_proj"),
];

const MLP: [(&str, &str); 3] = [
    ("gate", "gate_proj"),
    ("up", "up_proj"),
    ("down", "down_proj"),
];

/// Adds attention, indexer and MLP norm rules for one layer.
pub(crate) fn attn_rules(
    layer: u64,
    attn_prefix: &str,
    mlp_norm: &str,
    indexer_prefix: Option<&str>,
    rules: &mut Vec<Rule>,
) {
    let at = format!("model.layers.{layer}");
    for (field, stem) in ATTENTION {
        rules.push((format!("{attn_prefix}.{field}"), format!("{at}.{stem}")));
    }
    rules.push((mlp_norm.into(), format!("{at}.post_attention_layernorm")));
    if let Some(prefix) = indexer_prefix {
        for (field, stem) in INDEXER {
            rules.push((
                format!("{prefix}.{field}"),
                format!("{at}.self_attn.indexer.{stem}"),
            ));
        }
    }
}

/// Adds routed and shared expert rules for one MoE layer.
pub(crate) fn moe_rules(layer: u64, moe_prefix: &str, rotated: bool, rules: &mut Vec<Rule>) {
    let at = format!("model.layers.{layer}");
    rules.push((format!("{moe_prefix}.router"), format!("{at}.mlp.gate")));
    rules.push((
        format!("{moe_prefix}.router_bias"),
        format!("{at}.mlp.gate.e_score_correction_bias"),
    ));
    for side in ["signs_in", "signs_out"].iter().filter(|_| rotated) {
        rules.push((
            format!("{moe_prefix}.{side}"),
            format!("{at}.mlp.experts.{side}"),
        ));
    }
    for (field, stem) in MLP {
        let shared = format!("{at}.mlp.shared_experts.{stem}");
        rules.push((format!("{moe_prefix}.shared.{field}"), shared));
        if !rotated {
            let experts = format!("{at}.mlp.experts.{{}}.{stem}");
            rules.push((format!("{moe_prefix}.{field}"), experts));
            continue;
        }
        let set = format!("{at}.mlp.experts.{field}");
        for end in ["su", "sv"] {
            rules.push((
                format!("{moe_prefix}.{field}_{end}"),
                format!("{set}.{end}"),
            ));
        }
        rules.push((format!("{moe_prefix}.{field}"), set));
    }
}

/// Each weight's HF slice. `rotated` names GLM's exlr8 experts.
pub fn rules(c: &Config, rotated: bool) -> Vec<Rule> {
    let mut rules: Vec<Rule> = Vec::new();
    rules.push(("embed".into(), "model.embed_tokens".into()));
    rules.push(("norm".into(), "model.norm".into()));
    rules.push(("head".into(), "lm_head".into()));
    let mut indexers = 0;
    for l in 0..c.stack.layers {
        let idx = c.indexes(l).then(|| {
            let p = format!("indexers.{indexers}");
            indexers += 1;
            p
        });
        attn_rules(
            l,
            &format!("attn.{l}"),
            &format!("mlp_norm.{l}"),
            idx.as_deref(),
            &mut rules,
        );
        // Routed layers count from the first one.
        let Some(m) = l.checked_sub(c.dense_layers) else {
            let at = format!("model.layers.{l}");
            for (field, stem) in MLP {
                rules.push((format!("dense.{l}.{field}"), format!("{at}.mlp.{stem}")));
            }
            continue;
        };
        moe_rules(l, &format!("moe.{m}"), rotated, &mut rules);
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
        take: impl FnMut(u32, Weight) -> Result<(), Box<dyn Error + Send + Sync>>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        stream_each(&self.graph, &*self.offer, &self.rules, shares, take)
    }
}

impl Stream for Release {
    fn stream(
        &self,
        shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
        take: &mut dyn FnMut(u32, Weight) -> Result<(), Box<dyn Error + Send + Sync>>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.each(shares, take)
    }
}

/// `dir` as safetensors, or exlr8 with experts at `width`.
pub fn release(
    dir: &Path,
    seqs: u64,
    context: u64,
    width: u32,
) -> Result<Release, Box<dyn Error + Send + Sync>> {
    open(dir, None, [seqs, context], Some(width), None, None)
}

/// A `step` sets the graph's rules, mask and taps.
fn open(
    dir: &Path,
    merged: Option<&str>,
    [seqs, context]: [u64; 2],
    width: Option<u32>,
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
    let exlr8 = is_exlr8(dir);
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
    let offer: Box<dyn Offer> = match (exlr8, width) {
        (true, Some(w)) => Box::new(crate::release(dir, manifest)?.width(w)),
        _ => crate::offer(dir, manifest)?,
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
