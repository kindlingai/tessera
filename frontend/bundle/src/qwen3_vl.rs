//! Qwen3-VL-Embedding bundle loader: vision tower and packed text backbone.

use crate::arch::{Spot, arch};
use crate::encoder::{Embeds, Encoder, Shape, Vision};
use crate::names::repo;
use crate::tokenize::hf::Hf;
use crate::{Loaded, Overrides, pull_raw};
use std::error::Error;
use std::path::Path;
use tessera_engine::Linear;
use tessera_engine::qwen3_vl::{Config, text_program, vision_program};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

const DEFAULT_IMAGE_TOKEN: u32 = 151_655;
const DEFAULT_MIN_PIXELS: usize = 4_096;
const DEFAULT_MAX_PIXELS: usize = 1_310_720;
const MAX_MERGED: u64 = 4_096;

arch!(
    "qwen3_vl",
    [Spot::Type("qwen3_vl")],
    rules: include_str!("../names/qwen3_vl.json"),
    embed: Embeds {
        shape: Shape {
            max_tokens: 32_768,
            seqs: 256,
            rows: 16_384,
        },
        open: encoder,
    },
);

/// Reads image pixel bounds from `preprocessor_config.json`, if present.
fn pixel_bounds(dir: &Path) -> (usize, usize) {
    let Ok(text) = std::fs::read_to_string(dir.join("preprocessor_config.json")) else {
        return (DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS);
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return (DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS);
    };
    let min_px = v["min_pixels"]
        .as_u64()
        .or_else(|| v["size"]["shortest_edge"].as_u64())
        .map(|n| n as usize)
        .unwrap_or(DEFAULT_MIN_PIXELS);
    let max_px = v["max_pixels"]
        .as_u64()
        .or_else(|| v["size"]["longest_edge"].as_u64())
        .map(|n| n as usize)
        .unwrap_or(DEFAULT_MAX_PIXELS);
    (min_px, max_px)
}

/// Reads `image_token_id` from `config.json`, falling back to default.
fn image_token_id(dir: &Path, overrides: &Overrides) -> u32 {
    let Ok(text) = overrides.merged("config.json", &dir.join("config.json")) else {
        return DEFAULT_IMAGE_TOKEN;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return DEFAULT_IMAGE_TOKEN;
    };
    v["image_token_id"]
        .as_u64()
        .map(|n| n as u32)
        .unwrap_or(DEFAULT_IMAGE_TOKEN)
}

/// Loads the Qwen3-VL text backbone and vision tower from `dir`.
fn encoder(
    dir: &Path,
    [seqs, max_tokens]: [u64; 2],
    _: bool,
    overrides: &Overrides,
) -> Result<Encoder> {
    let max_rows = max_tokens.max(16_384);
    let cfg = overrides.config(dir, |k| Config::read(k, seqs, max_rows, MAX_MERGED))?;
    let (min_pixels, max_pixels) = pixel_bounds(dir);
    let image_token = image_token_id(dir, overrides);
    let text_graph = text_program::<Linear>(&cfg.text);
    let vis_graph = vision_program::<Linear>(&cfg.vision);
    let rules = repo("qwen3_vl").unwrap_or_default();
    let offer = tessera_checkpoint::open(dir, None)?;
    // Warm cached slice inventory once before pulling weights.
    let _ = offer.slices();
    let tok_path = dir.join("tokenizer.json");
    let (text_res, vis_res, tok_res) = std::thread::scope(|scope| {
        let t_tok = scope.spawn(|| Hf::open(&tok_path));
        let t_vis = scope.spawn(|| {
            pull_raw(
                &vis_graph,
                offer.as_ref(),
                cfg.vision.depth,
                &[&rules],
                &|_| None,
            )
        });
        let text_res = pull_raw(
            &text_graph,
            offer.as_ref(),
            cfg.text.layers,
            &[&rules],
            &|_| None,
        );
        let vis_res = t_vis.join().expect("vision weights");
        let tok_res = t_tok.join().expect("tokenizer");
        (text_res, vis_res, tok_res)
    });
    let (raw_text, pins) = text_res?;
    let (raw_vis, vis_pins) = vis_res?;
    let weights = raw_text.into_iter().map(Loaded::into_weight).collect();
    let vis_weights = raw_vis.into_iter().map(Loaded::into_weight).collect();
    let tokenizer = tok_res?;
    Ok(Encoder {
        graph: text_graph,
        weights,
        pins,
        tokenizer,
        vision: Some(Vision {
            graph: vis_graph,
            weights: vis_weights,
            pins: vis_pins,
            pos_grid: cfg.vision.pos_grid() as usize,
            min_pixels,
            max_pixels,
            image_token,
            deepstack: cfg.vision.deepstack_indexes.len(),
            max_merged: MAX_MERGED as usize,
        }),
        hidden: cfg.text.hidden as usize,
    })
}
