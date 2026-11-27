//! Qwen3.8-Flash-Next (`qwen4_exp`) directory loading, rules and PLE config.

use crate::arch::{Arch, Ask, Built, Spot, arch};
use crate::names::Rule;
use crate::{Stream, Weights, numbered, pins};
use serde::Deserialize;
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::{Offer, Shards};
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
    rules_for(c, rotated, false)
}

fn rules_for(c: &Config, rotated: bool, stacked: bool) -> Vec<Rule> {
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
            } else if stacked {
                rules.push((format!("{mp}.{f}"), format!("{at}.mlp.experts.{s}")));
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
}

fn parse_config(json: &str, [seqs, context]: [u64; 2]) -> Result<Config> {
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
    let slices = offer.slices();
    let rotated = crate::mla::has_rotated_experts(&slices);
    let stacked = !rotated && crate::mla::has_stacked_experts(&slices);
    let mut rules = rules_for(&config, rotated, stacked);
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

/// Locates the on-disk PLE table from exlr8 or safetensors shards.
fn ple_table(
    dir: &Path,
    ple: tessera_engine::qwen4_exp::Ple,
    base: &str,
) -> Result<(std::path::PathBuf, u64, Option<f32>, usize)> {
    let stem = dir.join(format!("ple/exlr8-ple-L{:03}", ple.layer));
    let bin = stem.with_extension("bin");
    if bin.is_file() {
        let meta = stem.with_extension("json");
        let row_dim = match std::fs::read_to_string(&meta) {
            Ok(text) => {
                serde_json::from_str::<PleMeta>(&text)
                    .map_err(|e| format!("{}: {e}", meta.display()))?
                    .dim
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                ple.dim as usize / tessera_engine::ple_heads()
            }
            Err(e) => return Err(format!("{}: {e}", meta.display()).into()),
        };
        return Ok((bin, 0, None, row_dim));
    }
    let shards = Shards::open(dir)?;
    let table = format!("{base}.ngram_embedding.weight");
    let (path, offset, _) = shards
        .location(&table)
        .ok_or_else(|| format!("missing PLE table {} and {table}", bin.display()))?;
    let shape = shards
        .shape(&table)
        .ok_or_else(|| format!("missing {table}"))?;
    let row_dim = *shape.get(1).ok_or_else(|| format!("{table} is not 2D"))?;
    let dtype = shards.dtype(&table).unwrap_or("");
    let fp8_scale = match dtype {
        "BF16" => None,
        "F8_E4M3" => {
            let scale_name = format!("{base}.ngram_embedding.weight_scale");
            let inv_name = format!("{base}.ngram_embedding.weight_scale_inv");
            let vals = shards
                .floats(&scale_name)
                .or_else(|_| shards.floats(&inv_name))?;
            let &s = vals
                .first()
                .ok_or_else(|| format!("empty scale for {table}"))?;
            Some(s)
        }
        other => return Err(format!("unsupported PLE dtype {other} for {table}").into()),
    };
    Ok((path.to_path_buf(), offset, fp8_scale, row_dim))
}

/// The config's host PLE table and hash parameters, if any.
fn ple(ask: &Ask) -> Result<Option<PleConfig>> {
    let cfg = parse_config(ask.json, ask.sizes)?;
    let Some(ple) = cfg.ple else {
        return Ok(None);
    };
    let base = format!("{LM}.layers.{}.ple.ple_embedding", ple.layer);
    let (path, data_offset, fp8_scale, row_dim) = ple_table(ask.dir, ple, &base)?;
    let offer = crate::offer(ask.dir, ask.manifest)?;
    let read = |field: &str| offer.i64s(&format!("{base}.{field}"));
    let fixed = |field: &str, v: Vec<i64>| format!("{field} has {} values", v.len());
    Ok(Some(PleConfig {
        path,
        data_offset,
        fp8_scale,
        eos_token: cfg.eos_token,
        multipliers: (read("layer_multipliers")?.try_into())
            .map_err(|v| fixed("layer_multipliers", v))?,
        offsets: (read("ngram_heads_offsets")?.try_into())
            .map_err(|v| fixed("ngram_heads_offsets", v))?,
        vocab_sizes: (read("ngram_heads_vocab_sizes")?.try_into())
            .map_err(|v| fixed("ngram_heads_vocab_sizes", v))?,
        row_dim,
        cache_rows: PleConfig::DEFAULT_CACHE_ROWS,
    }))
}
