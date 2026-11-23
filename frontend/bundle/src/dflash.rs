//! DFlash speculative bundle: tapped target plus cached drafter.

use crate::names::Rule;
use crate::{Bundle, Overrides, load_tapped, pull, release};
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::{Offer, Shards, Slice};
use tessera_engine::dflash::{Config, TargetSpec, stepper};
use tessera_engine::{Graph, Norm, Pins, Steps, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// Loaded target bundle paired with its DFlash drafter.
pub struct DFlash {
    pub target: Bundle,
    pub graph: Graph,
    pub weights: Vec<Weight>,
    pub pins: Pins,
    pub block: usize,
    pub mask_token: u32,
    pub causal: bool,
    /// 0-based target layer indices from the drafter config.
    pub target_layers: Vec<u64>,
}

/// Drafter slices supplemented by target embed and output head.
pub(crate) struct Overlay<'a> {
    primary: &'a dyn Offer,
    fallback: &'a dyn Offer,
    slices: Vec<Slice>,
    primary_names: Vec<String>,
}

impl<'a> Overlay<'a> {
    pub fn new(primary: &'a dyn Offer, fallback: &'a dyn Offer) -> Self {
        let mut slices = primary.slices();
        let primary_names: Vec<String> = slices.iter().map(|s| s.name.clone()).collect();
        let has_embed = slices.iter().any(|s| is_embed(&s.name));
        let has_head = slices.iter().any(|s| is_head(&s.name));
        for s in fallback.slices() {
            if primary_names.contains(&s.name) {
                continue;
            }
            let take = (!has_embed && is_embed(&s.name)) || (!has_head && is_head(&s.name));
            if take {
                slices.push(s);
            }
        }
        Self {
            primary,
            fallback,
            slices,
            primary_names,
        }
    }

    fn source(&self, name: &str) -> &'a dyn Offer {
        match self.primary_names.iter().any(|n| n == name) {
            true => self.primary,
            false => self.fallback,
        }
    }
}

impl Offer for Overlay<'_> {
    fn slices(&self) -> Vec<Slice> {
        self.slices.clone()
    }

    fn floats(&self, name: &str) -> std::result::Result<Vec<f32>, tessera_checkpoint::Error> {
        self.source(name).floats(name)
    }

    fn packed(&self, name: &str) -> std::result::Result<Vec<u8>, tessera_checkpoint::Error> {
        self.source(name).packed(name)
    }

    fn bf16(&self, name: &str) -> std::result::Result<Vec<u8>, tessera_checkpoint::Error> {
        self.source(name).bf16(name)
    }

    fn packed_rows(
        &self,
        name: &str,
        rows: (usize, usize),
    ) -> std::result::Result<Vec<u8>, tessera_checkpoint::Error> {
        self.source(name).packed_rows(name, rows)
    }

    fn packed_share(
        &self,
        name: &str,
        rows: (usize, usize),
    ) -> std::result::Result<Vec<u8>, tessera_checkpoint::Error> {
        self.source(name).packed_share(name, rows)
    }
}

fn has_index(name: &str) -> bool {
    name.split(['.', '_', '-'])
        .any(|p| p.parse::<usize>().is_ok())
}

fn is_embed(name: &str) -> bool {
    !has_index(name) && (name.contains("embed") || name.contains("embd") || name.contains("wte"))
}

fn is_head(name: &str) -> bool {
    !has_index(name) && (name.contains("lm_head") || name.starts_with("output."))
}

/// Whether the drafter checkpoint carries per-head Q/K norms.
pub(crate) fn has_norms(slices: &[Slice]) -> bool {
    slices
        .iter()
        .any(|s| s.name.contains("q_norm") || s.name.contains("q.norm"))
}

/// Reads the target's shared dimensions from `dir/config.json`.
fn target_spec(dir: &Path, overrides: &Overrides) -> Result<TargetSpec> {
    overrides.config(dir, |k| TargetSpec {
        vocab: k.need(&["vocab_size"]),
        hidden: k.need(&["hidden_size"]),
        layers: k.need(&["num_hidden_layers"]),
        tied: k.or(&["tie_word_embeddings"], false),
    })
}

/// Weight-name rules for DFlash2 convolutions and candidate selector.
pub(crate) fn extra_rules(config: &Config, slices: &[Slice]) -> Vec<Rule> {
    let prefix = match slices.iter().any(|s| s.name.starts_with("model.")) {
        true => "model.",
        false => "",
    };
    let mut rules = Vec::new();
    if config.conv.is_some() {
        for kind in ["attention_conv", "mlp_conv"] {
            for leaf in ["base_kernel", "kernel_projection"] {
                rules.push((
                    format!("layers.*.{kind}.{leaf}"),
                    format!("{prefix}layers.*.{kind}.{leaf}"),
                ));
            }
        }
    }
    if config.selector.is_some() {
        for leaf in [
            "hidden_projection",
            "predecessor_codebook",
            "successor_codebook",
        ] {
            rules.push((
                format!("candidate_selector.{leaf}"),
                format!("{prefix}candidate_selector.{leaf}"),
            ));
        }
    }
    rules
}

/// Loads only the tapped target bundle, for multi-rank follower ranks.
pub fn load_target(
    [target_dir, draft_dir]: [&Path; 2],
    [slots, context]: [u64; 2],
    user: &[Rule],
    overrides: &Overrides,
    manifest: Option<&str>,
) -> Result<Bundle> {
    let spec = target_spec(target_dir, overrides)?;
    let draft_over = overrides.under("draft");
    let config = draft_over.config(draft_dir, |k| Config::read(k, &spec, slots, context))?;
    load_tapped(
        target_dir,
        [slots, context],
        user,
        Steps::Plain,
        overrides,
        manifest,
        &config.tap_layers(),
    )
}

/// Loads a tapped target model and its DFlash drafter.
pub fn load(
    [target_dir, draft_dir]: [&Path; 2],
    [slots, context]: [u64; 2],
    user: &[Rule],
    overrides: &Overrides,
    manifest: Option<&str>,
) -> Result<DFlash> {
    let spec = target_spec(target_dir, overrides)?;
    let draft_over = overrides.under("draft");
    let config = draft_over.config(draft_dir, |k| Config::read(k, &spec, slots, context))?;
    let tap_layers = config.tap_layers();
    let target = load_tapped(
        target_dir,
        [slots, context],
        user,
        Steps::Plain,
        overrides,
        manifest,
        &tap_layers,
    )?;
    let draft_offer = Shards::open(draft_dir)?;
    let draft_slices = draft_offer.slices();
    let graph = match has_norms(&draft_slices) {
        true => stepper::<Norm>(&config),
        false => stepper::<()>(&config),
    };
    let target_offer: Box<dyn Offer> =
        match target_dir.join("dense/exlr8-vocab.safetensors").exists() {
            true => Box::new(release(target_dir, manifest)?),
            false => Box::new(Shards::open(target_dir)?),
        };
    let combined = Overlay::new(&draft_offer, target_offer.as_ref());
    let extra = extra_rules(&config, &draft_slices);
    let (weights, pins) = pull(&graph, &combined, config.stack.layers, &[&extra, user])?;
    Ok(DFlash {
        target,
        graph,
        weights,
        pins,
        block: config.block_size as usize,
        mask_token: config.mask_token,
        causal: config.causal,
        target_layers: config.target_layers,
    })
}
