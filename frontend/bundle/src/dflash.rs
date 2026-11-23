//! DFlash speculative bundle: tapped target plus cached drafter.

use crate::names::Rule;
use crate::{Bundle, Overrides, load_tapped, release};
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::{Offer, Shards, Slice, Stored};
use tessera_engine::dflash::{Config, DraftQuant, TargetSpec, stepper};
use tessera_engine::form::{Load, named};
use tessera_engine::{DimExpr, Graph, Norm, PathSeg, Pins, Steps, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// Streamed drafter weights, cut per rank on load.
pub struct DraftWeights {
    graph: Graph,
    draft: Shards,
    target: Box<dyn Offer>,
    rules: Vec<Rule>,
    pins: Pins,
}

impl DraftWeights {
    /// Each weight in graph order, sliced to `shares`.
    pub fn each_share<E: Into<Box<dyn Error + Send + Sync>>>(
        &self,
        shares: impl Fn(u32) -> Option<(u32, usize, usize)>,
        mut take: impl FnMut(u32, &Weight) -> std::result::Result<(), E>,
    ) -> Result<()> {
        let combined = Overlay::new(&self.draft, self.target.as_ref());
        let mxfp8 = named("mxfp8");
        let mut at = 0u32;
        crate::each(&self.graph, &combined, &[&self.rules], &shares, |w, _| {
            let weight = &self.graph.weights[at as usize];
            let want_mx = self.pins.form(&weight.path).name() == "mxfp8";
            if want_mx
                && let (Some(form), Weight::Packed(src, bytes)) = (mxfp8, &w)
                && src.name() == "bf16"
                && let Some(shape) = fixed_2d(&self.graph, &weight.ty.shape.0)
            {
                let packed = bf16_to_mxfp8(bytes, shape, shares(at), form);
                take(at, &Weight::Packed(form, packed)).map_err(Into::into)?;
            } else {
                take(at, &w).map_err(Into::into)?;
            }
            at += 1;
            Ok(())
        })
    }
}

fn fixed_2d(graph: &Graph, shape: &[DimExpr]) -> Option<[usize; 2]> {
    if shape.len() != 2 {
        return None;
    }
    let val = |d: &DimExpr| d.value(&graph.dims, &|_| None).map(|n| n as usize);
    Some([val(&shape[0])?, val(&shape[1])?])
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

/// Pins eligible BF16 linear weights to MXFP8.
pub(crate) fn draft_pins(
    graph: &Graph,
    offer: &dyn Offer,
    layers: u64,
    rules: &[&[Rule]],
    quant: DraftQuant,
) -> Result<Pins> {
    let slices = offer.slices();
    let claims = crate::names::resolve(graph, &slices, rules)?;
    let mxfp8 = (quant != DraftQuant::Bf16)
        .then(|| named("mxfp8"))
        .flatten();
    let mut forms: Vec<&'static dyn Load> = Vec::with_capacity(claims.len());
    for (c, w) in claims.iter().zip(&graph.weights) {
        let slice = slices.iter().find(|s| s.name == c.name).expect("resolved");
        let eligible = mxfp8.is_some()
            && slice.stored == Stored::Bf16
            && c.parts.is_empty()
            && c.rows.is_none()
            && !keep_bf16(&w.path, quant)
            && fixed_2d(graph, &w.ty.shape.0)
                .is_some_and(|[n, k]| n.is_multiple_of(16) && k.is_multiple_of(512));
        if eligible {
            forms.push(mxfp8.expect("checked"));
        } else {
            forms.push(crate::form(slice.stored, !c.parts.is_empty())?);
        }
    }
    Ok(crate::gguf::pins(graph, &forms, layers))
}

/// Packs a BF16 matrix slice into MXFP8 across CPU threads.
pub(crate) fn bf16_to_mxfp8(
    bytes: &[u8],
    [rows, cols]: [usize; 2],
    share: Option<(u32, usize, usize)>,
    mxfp8: &'static dyn Load,
) -> Vec<u8> {
    let (r0, r_cnt, c0, c_cnt) = match share {
        Some((0, first, count)) => (first, count, 0, cols),
        Some((1, first, count)) => (0, rows, first, count),
        _ => (0, rows, 0, cols),
    };
    let row_out = mxfp8.bytes(c_cnt);
    let mut out = vec![0u8; r_cnt * row_out];
    if r_cnt == 0 || c_cnt == 0 {
        return out;
    }
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(r_cnt);
    let per = r_cnt.div_ceil(threads);
    std::thread::scope(|scope| {
        for (ti, chunk) in out.chunks_mut(per * row_out).enumerate() {
            let start_r = r0 + ti * per;
            let count_r = chunk.len() / row_out;
            scope.spawn(move || {
                let batch = (65_536 / c_cnt.max(1)).clamp(1, count_r);
                let mut buf = vec![0.0f32; batch * c_cnt];
                let mut done = 0;
                while done < count_r {
                    let take = batch.min(count_r - done);
                    let floats = &mut buf[..take * c_cnt];
                    for dr in 0..take {
                        let base = ((start_r + done + dr) * cols + c0) * 2;
                        let src = &bytes[base..base + c_cnt * 2];
                        let dst = &mut floats[dr * c_cnt..(dr + 1) * c_cnt];
                        for (o, &pair) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                            *o = f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16);
                        }
                    }
                    let packed = mxfp8.pack(floats);
                    chunk[done * row_out..(done + take) * row_out].copy_from_slice(&packed);
                    done += take;
                }
            });
        }
    });
    out
}

/// Loaded target bundle paired with its DFlash drafter.
pub struct DFlash {
    pub target: Bundle,
    pub graph: Graph,
    pub weights: DraftWeights,
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

    fn bf16(&self, name: &str) -> std::result::Result<Vec<u8>, tessera_checkpoint::Error> {
        if let Some((parent, i, count)) = self.stacked.get(name) {
            let all = self.primary.bf16(parent)?;
            return Ok(take_part(&all, *i, *count));
        }
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
    let target_offer: Box<dyn Offer> =
        match target_dir.join("dense/exlr8-vocab.safetensors").exists() {
            true => Box::new(release(target_dir, manifest)?),
            false => Box::new(Shards::open(target_dir)?),
        };
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
    let weights = DraftWeights {
        graph: graph.clone(),
        draft: draft_offer,
        target: target_offer,
        rules,
        pins: draft_pins.clone(),
    };
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
