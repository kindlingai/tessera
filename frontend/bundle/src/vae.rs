//! FLUX.2's VAE decoder from a diffusers directory.

use crate::Overrides;
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::File;
use tessera_engine::vae::{Config, program};
use tessera_engine::{Graph, PathSeg, Tensor, Weight, bind};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub struct Decoder {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    /// Pixels a side at each level, the latent first.
    pub sides: Vec<u64>,
}

/// `[out, in, k, k]` kernels as `[out, k, k, in]`.
fn taps(v: &[f32], shape: &[usize]) -> Vec<f32> {
    let [out, cin, kh, kw] = shape[..] else {
        return v.to_vec();
    };
    let mut t = vec![0.0; v.len()];
    for o in 0..out {
        for c in 0..cin {
            for k in 0..kh * kw {
                t[(o * kh * kw + k) * cin + c] = v[(o * cin + c) * kh * kw + k];
            }
        }
    }
    t
}

/// `dir/vae`, `side` tokens a side, `seqs` per launch.
pub fn decoder(dir: &Path, sizes: [u64; 2], overrides: &Overrides) -> Result<Decoder> {
    let config = config(dir, sizes, overrides)?;
    let graph = program(&config);
    let weights = weights(dir, config.bn_eps, &graph)?;
    let sides = config.sides();
    Ok(Decoder {
        graph,
        weights,
        sides,
    })
}

/// The decoder `dir/vae/config.json` describes.
pub fn config(dir: &Path, [side, seqs]: [u64; 2], overrides: &Overrides) -> Result<Config> {
    const FILE: &str = "vae/config.json";
    overrides.read(FILE, Some(&dir.join(FILE)), |k| Config::read(k, side, seqs))
}

/// The weights `graph` reads from `dir/vae`. `bn_eps` is the config's.
pub fn weights(dir: &Path, bn_eps: f32, graph: &Graph) -> Result<Vec<Weight>> {
    let file = File::open(&dir.join("vae/diffusion_pytorch_model.safetensors"))?;
    let find = |p: &[PathSeg]| {
        let name = PathSeg::dotted(p);
        let v = match name.as_str() {
            "bn.std" => (file.floats("bn.running_var").ok()?.iter())
                .map(|v| (v + bn_eps).sqrt())
                .collect(),
            _ => taps(&file.floats(&name).ok()?, file.shape(&name)?),
        };
        Some(Tensor::f32(vec![v.len()], v))
    };
    let weights = bind(graph, find)?;
    Ok(weights.into_iter().map(Weight::Floats).collect())
}
