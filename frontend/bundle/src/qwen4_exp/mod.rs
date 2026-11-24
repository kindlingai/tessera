//! Qwen3.8-Flash-Next (`qwen4_exp`) directory loading, rules and PLE config.

use crate::arch::{Arch, Ask, Built, Spot, arch};
use crate::names::Rule;
use crate::{Stream, Weights, numbered, pins};
use serde::Deserialize;
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::Offer;
use tessera_engine::qwen4_exp::{Config, Moe, RotatedMoe, program, stepper};
use tessera_engine::qwen35::Mixing;
use tessera_engine::{Graph, Pins, PleConfig, Steps, Weight};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

const CLASS: &str = "Qwen4ExpForConditionalGeneration";
const LM: &str = "model.language_model";

fn step(_: &Arch, ask: &Ask) -> Result<Built> {
    let release = open(
        ask.dir,
        ask.json,
        ask.sizes,
        ask.manifest,
        Some((ask.user, ask.steps, ask.taps)),
    )?;
    let (graph, pins) = (release.graph.clone(), release.pins.clone());
    Ok((graph, Weights::Streamed(Box::new(release)), pins))
}

arch!(
    "qwen4_exp",
    [Spot::Type("qwen4_exp"), Spot::Class(CLASS)],
    step: step,
    ple: ple,
);

/// Maps every `qwen4_exp` weight path to its HF tensor stem.
pub fn rules(c: &Config, rotated: bool) -> Vec<Rule> {
    let mut rules = Vec::new();
    rules.push(("embed".into(), format!("{LM}.embed_tokens")));
    rules.push(("head".into(), "lm_head".into()));
    for (f, s) in [
        ("norm", "hc_norm"),
        ("down", "input_mix_weight_down"),
        ("up", "input_mix_weight_up"),
    ] {
        rules.push((
            format!("final_hc.{f}"),
            format!("{LM}.hyper_connection_mixer.{s}"),
        ));
    }
    for (i, kind) in c.layers.iter().enumerate() {
        let at = format!("{LM}.layers.{i}");
        if c.ple.is_some_and(|p| p.layer == i as u64) {
            for (f, s) in [
                ("key_proj", "key_proj"),
                ("value_proj", "value_proj"),
                ("conv", "conv1d"),
                ("norm_key", "norm_key"),
                ("norm_query", "norm_query"),
                ("norm_conv", "norm_conv"),
            ] {
                rules.push((format!("layers.{i}.ple.{f}"), format!("{at}.ple.{s}")));
            }
        }
        for (hc_f, hc_s) in [
            ("attn_hc", "attn_hyper_connection"),
            ("mlp_hc", "mlp_hyper_connection"),
        ] {
            for (f, s) in [
                ("norm", "hc_norm"),
                ("down", "input_mix_weight_down"),
                ("up", "input_mix_weight_up"),
                ("inject", "block_inject_weight"),
            ] {
                rules.push((format!("layers.{i}.{hc_f}.{f}"), format!("{at}.{hc_s}.{s}")));
            }
        }
        match kind {
            Mixing::Linear => {
                for (f, s) in [
                    ("qkv", "in_proj_qkv"),
                    ("z", "in_proj_z"),
                    ("b", "in_proj_b"),
                    ("a", "in_proj_a"),
                    ("conv", "conv1d"),
                    ("dt_bias", "dt_bias"),
                    ("a_log", "A_log"),
                    ("norm", "norm"),
                    ("out", "out_proj"),
                ] {
                    rules.push((
                        format!("layers.{i}.linear.{f}"),
                        format!("{at}.linear_attn.{s}"),
                    ));
                }
            }
            Mixing::Full => {
                for (f, s) in [
                    ("q", "q_proj"),
                    ("k", "k_proj"),
                    ("v", "v_proj"),
                    ("q_norm", "q_norm"),
                    ("k_norm", "k_norm"),
                    ("o", "o_proj"),
                ] {
                    rules.push((
                        format!("layers.{i}.attn.{f}"),
                        format!("{at}.self_attn.{s}"),
                    ));
                }
            }
        }
        let mp = format!("layers.{i}.moe");
        rules.push((format!("{mp}.router"), format!("{at}.mlp.gate")));
        rules.push((
            format!("{mp}.shared.scale"),
            format!("{at}.mlp.shared_expert_gate"),
        ));
        for (f, s) in [
            ("gate", "gate_proj"),
            ("up", "up_proj"),
            ("down", "down_proj"),
        ] {
            rules.push((
                format!("{mp}.shared.{f}"),
                format!("{at}.mlp.shared_expert.{s}"),
            ));
            if rotated {
                let set = format!("{at}.mlp.experts.{f}");
                for end in ["su", "sv"] {
                    rules.push((format!("{mp}.{f}_{end}"), format!("{set}.{end}")));
                }
                rules.push((format!("{mp}.{f}"), set));
            } else {
                rules.push((format!("{mp}.{f}"), format!("{at}.mlp.experts.{{}}.{s}")));
            }
        }
        if rotated {
            for side in ["signs_in", "signs_out"] {
                rules.push((format!("{mp}.{side}"), format!("{at}.mlp.experts.{side}")));
            }
        }
    }
    rules
}

pub struct Release {
    pub config: Config,
    pub graph: Graph,
    pub pins: Pins,
    offer: Box<dyn Offer>,
    rules: Vec<Rule>,
}

impl Stream for Release {
    fn stream(
        &self,
        shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
        take: &mut dyn FnMut(u32, Weight) -> Result<()>,
    ) -> Result<()> {
        numbered(&self.graph, &*self.offer, &[&self.rules], shares, take)
    }

    fn i64s(&self, name: &str) -> Result<Vec<i64>> {
        Ok(self.offer.i64s(name)?)
    }
}

pub(crate) fn parse_config(json: &str, [seqs, context]: [u64; 2]) -> Result<Config> {
    let offered = [tessera_engine::keys::Offered::parse("config.json", json)?];
    let mut keys = tessera_engine::keys::Keys::new(&offered);
    let config = Config::read(&mut keys, seqs, context);
    keys.finish()?;
    Ok(config)
}

fn open(
    dir: &Path,
    json: &str,
    sizes: [u64; 2],
    manifest: Option<&str>,
    step: Option<(&[Rule], Steps, &[u64])>,
) -> Result<Release> {
    let mut config = parse_config(json, sizes)?;
    let offer = crate::offer(dir, manifest)?;
    let rotated = crate::mla::has_rotated_experts(&offer.slices());
    let mut rules = rules(&config, rotated);
    if let Some((user, steps, taps)) = step {
        config.stack.steps = steps;
        config.stack.taps = taps.to_vec();
        rules.extend(user.iter().cloned());
    }
    let graph = match (rotated, step.is_some()) {
        (true, true) => stepper::<RotatedMoe>(&config),
        (true, false) => program::<RotatedMoe>(&config),
        (false, true) => stepper::<Moe>(&config),
        (false, false) => program::<Moe>(&config),
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

#[derive(Deserialize)]
struct PleMeta {
    dim: usize,
}

/// The config's host PLE table and hash parameters, if any.
pub(crate) fn ple(ask: &Ask, weights: &Weights) -> Result<Option<PleConfig>> {
    let cfg = parse_config(ask.json, ask.sizes)?;
    let Some(ple) = cfg.ple else {
        return Ok(None);
    };
    let stem = ask.dir.join(format!("ple/exlr8-ple-L{:03}", ple.layer));
    let bin = stem.with_extension("bin");
    if !bin.is_file() {
        return Err(format!("missing PLE table {}", bin.display()).into());
    }
    let heads = ple.heads as usize;
    let meta = stem.with_extension("json");
    let row_dim = match std::fs::read_to_string(&meta) {
        Ok(text) => {
            serde_json::from_str::<PleMeta>(&text)
                .map_err(|e| format!("{}: {e}", meta.display()))?
                .dim
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => ple.dim as usize / heads.max(1),
        Err(e) => return Err(format!("{}: {e}", meta.display()).into()),
    };
    if row_dim * heads != ple.dim as usize {
        let dim = ple.dim;
        return Err(format!("PLE rows of {row_dim} over {heads} heads do not make {dim}").into());
    }
    let Weights::Streamed(release) = weights else {
        return Err("PLE hash parameters need the open release".into());
    };
    let base = format!("{LM}.layers.{}.ple.ple_embedding", ple.layer);
    let read = |field: &str, want: usize| -> Result<Vec<i64>> {
        let v = release.i64s(&format!("{base}.{field}"))?;
        if v.len() != want {
            return Err(format!("{field} has {} values, want {want}", v.len()).into());
        }
        Ok(v)
    };
    Ok(Some(PleConfig {
        path: bin,
        eos_token: cfg.eos_token,
        multipliers: read("layer_multipliers", ple.dilation as usize)?,
        offsets: read("ngram_heads_offsets", heads)?,
        vocab_sizes: read("ngram_heads_vocab_sizes", heads)?,
        row_dim,
        cache_rows: PleConfig::DEFAULT_CACHE_ROWS,
    }))
}
