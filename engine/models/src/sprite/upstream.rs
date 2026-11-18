//! tessera-small's `model_config.json`, as the graph's config.

use super::tables::COLOR_FREQS;
use super::{Config, HEAD_DIM, REGISTERS, TEXT};
use serde::Deserialize;

/// What `model_config.json` holds. The graph fixes some of it.
#[derive(Deserialize)]
struct Upstream {
    vocab_size: u64,
    grid_size: u64,
    dim: u64,
    n_heads: u64,
    depth_in: u64,
    depth_core: u64,
    depth_out: u64,
    patch: u64,
    mlp_ratio: f64,
    n_registers: u64,
    text_dim: u64,
    text_len: u64,
    rope_theta: f64,
    conv_ffn: bool,
    color_freqs: u64,
    self_cond: bool,
}

impl Config {
    /// The config `model_config.json` gives, for `seqs` canvases.
    pub fn upstream(json: &str, seqs: u64) -> Result<Config, String> {
        let u: Upstream = serde_json::from_str(json).map_err(|e| e.to_string())?;
        // The graph merges 2x2 patches and reads two Fourier tables.
        let fixed = [
            ("n_registers", u.n_registers == REGISTERS),
            ("text_len", u.text_len == TEXT),
            ("patch", u.patch == 2),
            ("dim / n_heads", u.dim == HEAD_DIM * u.n_heads),
            ("color_freqs", u.color_freqs == COLOR_FREQS),
            ("conv_ffn", u.conv_ffn),
            ("self_cond", u.self_cond),
        ];
        if let Some((name, _)) = fixed.iter().find(|(_, same)| !same) {
            return Err(format!("model_config.json: the graph fixes {name}"));
        }
        Ok(Config {
            grid: u.grid_size,
            palette: u.vocab_size,
            hidden: u.dim,
            heads: u.n_heads,
            ffn: (u.dim as f64 * u.mlp_ratio) as u64,
            text_dim: u.text_dim,
            depth: [u.depth_in, u.depth_core, u.depth_out],
            max_seqs: seqs,
            rope: u.rope_theta,
        })
    }
}
