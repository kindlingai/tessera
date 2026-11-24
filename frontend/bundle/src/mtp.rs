//! Native MTP speculative bundle: tapped MLA target plus MTP layer.

use crate::dflash::packing;
use crate::mla::{self, attn_rules, moe_rules};
use crate::names::Rule;
use crate::{Bundle, Overrides, Stream, Weights, is_exlr8, load_tapped, offer};
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::{Offer, Stored};
use tessera_engine::form::{Load, named};
use tessera_engine::mla::{Moe, RotatedMoe};
use tessera_engine::mtp_model;
use tessera_engine::{DimExpr, Graph, PathSeg, Pins, Steps, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// Streamed MTP weights, cut per rank on load.
pub(crate) struct MtpWeights {
    pub(crate) graph: Graph,
    pub(crate) offer: Box<dyn Offer>,
    pub(crate) rules: Vec<Rule>,
    pub(crate) pins: Pins,
}

fn head_shape(graph: &Graph, path: &[PathSeg], shape: &[DimExpr]) -> Option<[usize; 2]> {
    if !matches!(path, [PathSeg::Field(f)] if f == "head") || shape.len() != 2 {
        return None;
    }
    let val = |d: &DimExpr| d.value(&graph.dims, &|_| None).map(|n| n as usize);
    Some([val(&shape[0])?, val(&shape[1])?])
}

/// MTP weight forms, quantizing eligible BF16 head to MXFP8.
pub(crate) fn mtp_pins(graph: &Graph, offer: &dyn Offer, rules: &[&[Rule]]) -> Result<Pins> {
    let slices = offer.slices();
    let claims = crate::names::resolve(graph, &slices, rules)?;
    let mxfp8 = named("mxfp8");
    let mut forms: Vec<&'static dyn Load> = Vec::with_capacity(claims.len());
    for (c, w) in claims.iter().zip(&graph.weights) {
        let slice = slices.iter().find(|s| s.name == c.name).expect("resolved");
        let quantize = slice.stored == Stored::Bf16
            && c.parts.is_empty()
            && c.rows.is_none()
            && head_shape(graph, &w.path, &w.ty.shape.0)
                .is_some_and(|[n, k]| n.is_multiple_of(16) && k.is_multiple_of(512));
        if quantize && let Some(form) = mxfp8 {
            forms.push(form);
        } else {
            forms.push(crate::form(slice.stored, !c.parts.is_empty())?);
        }
    }
    Ok(crate::gguf::pins(graph, &forms, 1))
}

impl Stream for MtpWeights {
    fn stream(
        &self,
        shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
        take: &mut dyn FnMut(u32, Weight) -> Result<()>,
    ) -> Result<()> {
        let (g, offer, rules) = (&self.graph, &*self.offer, &self.rules[..]);
        packing(g, offer, rules, &self.pins, head_shape, shares, take)
    }
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
    let exlr8 = is_exlr8(dir);
    let offer = offer(dir, manifest)?;
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
    let weights = Weights::Streamed(Box::new(MtpWeights {
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
