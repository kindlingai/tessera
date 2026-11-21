//! Transformer latents to pictures, through FLUX.2's VAE.

use crate::Error;
use crate::api::Image;
use std::path::Path;
use tessera_bundle::Overrides;
use tessera_bundle::vae::decoder;
use tessera_engine::vae::{TAPS, TOKEN};
use tessera_engine::{Model, Recipe, Settings, Tensor};

pub struct Pictures {
    model: Model,
    /// Transformer tokens a side.
    side: usize,
    /// Pixels a side at each level, the latent first.
    sides: Vec<usize>,
}

/// Each pixel's 3x3 neighbours, row-major. Past the edge reads zero.
fn neighbours(side: usize) -> Vec<u32> {
    let mut ids = Vec::with_capacity(side * side * TAPS as usize);
    for r in 0..side as isize {
        for c in 0..side as isize {
            for (dr, dc) in (-1..=1).flat_map(|dr| (-1..=1).map(move |dc| (dr, dc))) {
                let (rr, cc) = (r + dr, c + dc);
                let inside = (0..side as isize).contains(&rr) && (0..side as isize).contains(&cc);
                ids.push(if inside {
                    (rr * side as isize + cc) as u32
                } else {
                    (side * side) as u32
                });
            }
        }
    }
    ids
}

impl Pictures {
    /// The VAE in `dir`, sized for one launch.
    pub fn open(
        dir: &Path,
        settings: &Settings,
        [side, seqs]: [usize; 2],
        config: &Overrides,
    ) -> Result<Self, Error> {
        let d = decoder(dir, [side as u64, seqs as u64], config)?;
        let mut model = Model::open(d.graph, &Recipe::default(), settings)?;
        model.load(&d.weights)?;
        let sides = d.sides.iter().map(|&s| s as usize).collect();
        Ok(Self { model, side, sides })
    }

    /// One picture per block of `side^2` tokens.
    pub fn decode(&mut self, tokens: &[f32]) -> Result<Vec<Image>, Error> {
        let rows = self.side * self.side;
        let n = tokens.len() / (rows * TOKEN as usize);
        let mut inputs = vec![Tensor::f32(vec![n, rows, TOKEN as usize], tokens.to_vec())];
        for &s in &self.sides {
            inputs.push(Tensor::u32(vec![s * s, TAPS as usize], neighbours(s)));
        }
        let rgb = self.model.run(&inputs)?[0].floats()?.to_vec();
        let side = *self.sides.last().ok_or("a decoder with no levels")?;
        let byte = |v: f32| ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
        let picture = |px: &[f32]| Image {
            width: side,
            height: side,
            rgba: px
                .chunks(3)
                .flat_map(|p| [byte(p[0]), byte(p[1]), byte(p[2]), 255])
                .collect(),
        };
        Ok(rgb.chunks(side * side * 3).map(picture).collect())
    }
}
