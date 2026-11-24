//! Native MTP speculative bundle: tapped MLA target plus MTP layer.

use crate::mla::{self, attn_rules, moe_rules};
use crate::names::Rule;
use crate::repack::{self, Repack, Streamed};
use crate::{Bundle, Overrides, Weights, load_tapped, offer};
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::Offer;
use tessera_engine::mla::{Moe, RotatedMoe};
use tessera_engine::mtp_model;
use tessera_engine::{Graph, PathSeg, Pins, Steps};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// MTP weight forms, quantizing eligible float head to MXFP8.
pub(crate) fn mtp_pins(graph: &Graph, offer: &dyn Offer, rules: &[&[Rule]]) -> Result<Pins> {
    repack::pins(
        graph,
        offer,
        1,
        rules,
        Repack::mxfp8(),
        |path| matches!(path, [PathSeg::Field(f)] if f == "head"),
    )
}

/// Loaded MLA target bundle paired with its native MTP drafter.
pub struct Mtp {
    pub target: Bundle,
    pub graph: Graph,
    pub weights: Weights,
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
    let offer = offer(dir, manifest)?;
    let slices = offer.slices();
    let eh = format!("model.layers.{layer}.eh_proj.weight");
    if !slices.iter().any(|s| s.name == eh) {
        return Err(format!("checkpoint has no MTP layer {layer} (missing {eh})").into());
    }
    let idx_prefix = format!("model.layers.{layer}.self_attn.indexer.");
    let indexed =
        target_cfg.index.is_some() && slices.iter().any(|s| s.name.starts_with(&idx_prefix));
    let signs = format!("model.layers.{layer}.mlp.experts.signs_in");
    let rotated = slices.iter().any(|s| s.name == signs);
    let mtp_cfg = mtp_model::config(&target_cfg, indexed);
    let graph = match rotated {
        true => mtp_model::stepper::<RotatedMoe>(&mtp_cfg),
        false => mtp_model::stepper::<Moe>(&mtp_cfg),
    };
    let mut all_rules = rules(layer, indexed, rotated);
    all_rules.extend_from_slice(user);
    let mtp_pins = mtp_pins(&graph, offer.as_ref(), &[&all_rules])?;
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
    let weights = Weights::Streamed(Box::new(Streamed {
        graph: graph.clone(),
        offer,
        rules: all_rules,
        pins: mtp_pins.clone(),
    }));
    Ok(Mtp {
        target,
        graph,
        weights,
        pins: mtp_pins,
        layer,
    })
}
