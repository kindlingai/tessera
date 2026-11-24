//! DFlash speculative bundle: tapped target plus cached drafter.

use crate::names::Rule;
use crate::repack::{self, Repack};
use crate::{Bundle, Overrides, Stream, Weights, load_tapped, offer};
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::{Offer, Shards, Slice};
use tessera_engine::dflash::{Config, DraftQuant, TargetSpec, stepper};
use tessera_engine::{Graph, Norm, PathSeg, Pins, Steps, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// Streamed drafter weights, cut per rank on load.
pub(crate) struct DraftWeights {
    graph: Graph,
    draft: Shards,
    target: Box<dyn Offer>,
    rules: Vec<Rule>,
    pins: Pins,
}

impl Stream for DraftWeights {
    fn stream(
        &self,
        shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
        take: &mut dyn FnMut(u32, Weight) -> Result<()>,
    ) -> Result<()> {
        let combined = Overlay::new(&self.draft, self.target.as_ref());
        repack::stream(
            &self.graph,
            &combined,
            &self.rules,
            &self.pins,
            shares,
            take,
        )
    }
}

fn keep_bf16(path: &[PathSeg], quant: DraftQuant) -> bool {
    if matches!(
        path.last(),
        Some(PathSeg::Field(f)) if f == "embed" || f.ends_with("codebook")
    ) {
        return true;
    }
    if quant == DraftQuant::MxFp8 {
        return matches!(
            path.first(),
            Some(PathSeg::Field(f)) if f == "fc" || f == "candidate_selector"
        );
    }
    false
}

/// Pins eligible float linear weights to MXFP8 when `quant` asks.
pub(crate) fn draft_pins(
    graph: &Graph,
    offer: &dyn Offer,
    layers: u64,
    rules: &[&[Rule]],
    quant: DraftQuant,
) -> Result<Pins> {
    let target = (quant != DraftQuant::Bf16).then(Repack::mxfp8).flatten();
    repack::pins(graph, offer, layers, rules, target, |path| {
        !keep_bf16(path, quant)
    })
}

/// Loaded target bundle paired with its DFlash drafter.
pub struct DFlash {
    pub target: Bundle,
    pub graph: Graph,
    pub weights: Weights,
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
    /// Virtual 2D layer slices carved from 3D stacked bridge weights.
    stacked: HashMap<String, (String, usize, usize)>,
}

fn is_stacked_bridge(s: &Slice) -> bool {
    s.shape.len() == 3
        && (s.name.ends_with("early_read_weights") || s.name.ends_with("feature_write_weights"))
}

impl<'a> Overlay<'a> {
    pub fn new(primary: &'a dyn Offer, fallback: &'a dyn Offer) -> Self {
        let mut slices = Vec::new();
        let mut primary_names = Vec::new();
        let mut stacked = HashMap::new();
        for s in primary.slices() {
            primary_names.push(s.name.clone());
            if is_stacked_bridge(&s) {
                let count = s.shape[0];
                let shape = vec![s.shape[1], s.shape[2]];
                for i in 0..count {
                    let vname = format!("{}.{i}", s.name);
                    stacked.insert(vname.clone(), (s.name.clone(), i, count));
                    slices.push(Slice {
                        name: vname,
                        shape: shape.clone(),
                        stored: s.stored,
                    });
                }
            } else {
                slices.push(s);
            }
        }
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
            stacked,
        }
    }

    fn source(&self, name: &str) -> &'a dyn Offer {
        match self.primary_names.iter().any(|n| n == name) {
            true => self.primary,
            false => self.fallback,
        }
    }
}

fn take_part<T: Copy>(data: &[T], index: usize, count: usize) -> Vec<T> {
    let chunk = data.len() / count;
    data[index * chunk..(index + 1) * chunk].to_vec()
}

impl Offer for Overlay<'_> {
    fn slices(&self) -> Vec<Slice> {
        self.slices.clone()
    }

    fn floats(&self, name: &str) -> std::result::Result<Vec<f32>, tessera_checkpoint::Error> {
        if let Some((parent, i, count)) = self.stacked.get(name) {
            let all = self.primary.floats(parent)?;
            return Ok(take_part(&all, *i, *count));
        }
        self.source(name).floats(name)
    }

    fn packed(&self, name: &str) -> std::result::Result<Vec<u8>, tessera_checkpoint::Error> {
        if let Some((parent, i, count)) = self.stacked.get(name) {
            let all = self.primary.packed(parent)?;
            return Ok(take_part(&all, *i, *count));
        }
        self.source(name).packed(name)
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

    fn packed_axis(
        &self,
        name: &str,
        axis: u32,
        range: (usize, usize),
    ) -> std::result::Result<Vec<u8>, tessera_checkpoint::Error> {
        if self.stacked.contains_key(name) {
            return self.packed(name);
        }
        self.source(name).packed_axis(name, axis, range)
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
    if config.bridge.is_some() {
        let tencent = slices
            .iter()
            .any(|s| s.name.contains("selected_prefix_bridge."));
        if tencent {
            let b = format!("{prefix}selected_prefix_bridge");
            rules.push(("bridge.codebook".into(), format!("{b}.codebook")));
            rules.push(("bridge.gate".into(), format!("{b}.gate")));
            rules.push(("bridge.gate_bias".into(), format!("{b}.gate.bias")));
            rules.push(("bridge.pred_gate".into(), format!("{b}.predecessor_gate")));
            rules.push(("bridge.prev".into(), format!("{b}.prev")));
            let last = config.stack.layers.saturating_sub(1);
            for i in 0..last {
                rules.push((
                    format!("layers.{i}.bridge_read"),
                    format!("{b}.early_read_weights.{i}"),
                ));
            }
            rules.push((
                format!("layers.{last}.bridge_read"),
                format!("{b}.read_weight"),
            ));
            rules.push((
                "layers.*.bridge_write".into(),
                format!("{b}.feature_write_weights.*"),
            ));
        } else {
            rules.push(("bridge.codebook".into(), format!("{prefix}transfer_codes")));
            rules.push((
                "bridge.gate".into(),
                format!("{prefix}injection.receiver_gate"),
            ));
            rules.push(("bridge.gate_bias".into(), format!("{prefix}injection.bias")));
            rules.push((
                "bridge.pred_gate".into(),
                format!("{prefix}injection.predecessor_gate"),
            ));
            rules.push(("bridge.prev".into(), format!("{prefix}injection.message")));
            rules.push((
                "layers.*.bridge_read".into(),
                format!("{prefix}injection.readouts.*"),
            ));
            rules.push((
                "layers.*.bridge_write".into(),
                format!("{prefix}injection.writes.*"),
            ));
        }
    }
    rules
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
    let target_offer = offer(target_dir, manifest)?;
    let mut rules = extra_rules(&config, &draft_slices);
    rules.extend_from_slice(user);
    let draft_pins = {
        let combined = Overlay::new(&draft_offer, target_offer.as_ref());
        draft_pins(
            &graph,
            &combined,
            config.stack.layers,
            &[&rules],
            config.quant,
        )?
    };
    let weights = Weights::Streamed(Box::new(DraftWeights {
        graph: graph.clone(),
        draft: draft_offer,
        target: target_offer,
        rules,
        pins: draft_pins.clone(),
    }));
    Ok(DFlash {
        target,
        graph,
        weights,
        pins: draft_pins,
        block: config.block_size as usize,
        mask_token: config.mask_token,
        causal: config.causal,
        target_layers: config.target_layers,
    })
}
