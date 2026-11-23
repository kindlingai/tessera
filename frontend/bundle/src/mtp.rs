//! Native MTP speculative bundle: tapped MLA target plus MTP layer.

use crate::mla::{self, attn_rules, moe_rules};
use crate::names::Rule;
use crate::{Bundle, Overrides, WIDTH, load_tapped, pins, release};
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::{Offer, Shards};
use tessera_engine::mla::{Moe, RotatedMoe};
use tessera_engine::mtp_model;
use tessera_engine::{Graph, Pins, Steps, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// Streamed MTP weights, cut per rank on load.
pub struct MtpWeights {
    graph: Graph,
    offer: Box<dyn Offer>,
    rules: Vec<Rule>,
}

impl MtpWeights {
    /// Each weight in graph order, sliced to `shares`.
    pub fn each_share<E: Into<Box<dyn Error + Send + Sync>>>(
        &self,
        shares: impl Fn(u32) -> Option<(u32, usize, usize)>,
        mut take: impl FnMut(u32, &Weight) -> std::result::Result<(), E>,
    ) -> Result<()> {
        let mut at = 0u32;
        crate::each(
            &self.graph,
            &*self.offer,
            &[&self.rules],
            &shares,
            |w, _| {
                take(at, &w).map_err(Into::into)?;
                at += 1;
                Ok(())
            },
        )
    }
}

/// Loaded MLA target bundle paired with its native MTP drafter.
pub struct Mtp {
    pub target: Bundle,
    pub graph: Graph,
    pub weights: MtpWeights,
    pub pins: Pins,
    pub layer: u64,
}

/// Weight-name rules for MTP layer `layer`.
pub fn rules(layer: u64, indexed: bool, rotated: bool) -> Vec<Rule> {
    let at = format!("model.layers.{layer}");
    let mut out = vec![
        ("embed".into(), "model.embed_tokens".into()),
        ("enorm".into(), format!("{at}.enorm")),
        ("hnorm".into(), format!("{at}.hnorm")),
        ("eh_proj".into(), format!("{at}.eh_proj")),
        ("norm".into(), format!("{at}.shared_head.norm")),
        ("head".into(), "lm_head".into()),
    ];
    attn_rules(
        layer,
        "attn",
        "mlp_norm",
        indexed.then_some("indexers.0"),
        &mut out,
    );
    moe_rules(layer, "moe", rotated, &mut out);
    out
}

/// Loads a tapped MLA target and its native MTP layer.
pub fn load(
    dir: &Path,
    [slots, context]: [u64; 2],
    user: &[Rule],
    overrides: &Overrides,
    manifest: Option<&str>,
) -> Result<Mtp> {
    let json = overrides.merged("config.json", &dir.join("config.json"))?;
    let target_cfg = mla::config(&json, slots, context)
        .map_err(|e| format!("--mtp requires an MLA checkpoint: {e}"))?;
    let layer = target_cfg.stack.layers;
    let exlr8 = dir.join("dense/exlr8-vocab.safetensors").is_file();
    let offer: Box<dyn Offer> = match exlr8 {
        true => Box::new(release(dir, manifest)?.width(WIDTH)),
        false => Box::new(Shards::open(dir)?),
    };
    let slices = offer.slices();
    let eh = format!("model.layers.{layer}.eh_proj.weight");
    if !slices.iter().any(|s| s.name == eh) {
        return Err(format!("checkpoint has no MTP layer {layer} (missing {eh})").into());
    }
    let idx_prefix = format!("model.layers.{layer}.self_attn.indexer.");
    let indexed =
        target_cfg.index.is_some() && slices.iter().any(|s| s.name.starts_with(&idx_prefix));
    let mtp_cfg = mtp_model::config(&target_cfg, indexed);
    let graph = match exlr8 {
        true => mtp_model::stepper::<RotatedMoe>(&mtp_cfg),
        false => mtp_model::stepper::<Moe>(&mtp_cfg),
    };
    let mut all_rules = rules(layer, indexed, exlr8);
    all_rules.extend_from_slice(user);
    let mtp_pins = pins(&graph, offer.as_ref(), 1, &[&all_rules])?;
    let tap = [mtp_model::tap_layer(&target_cfg)];
    let target = load_tapped(
        dir,
        [slots, context],
        user,
        Steps::Plain,
        overrides,
        manifest,
        &tap,
    )?;
    let weights = MtpWeights {
        graph: graph.clone(),
        offer,
        rules: all_rules,
    };
    Ok(Mtp {
        target,
        graph,
        weights,
        pins: mtp_pins,
        layer,
    })
}
