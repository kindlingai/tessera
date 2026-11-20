//! FLUX.2's transformer as a velocity field over latent rows.

use crate::Error;
use sensai_bundle::Overrides;
use sensai_bundle::flux2::transformer;
use sensai_engine::flux2::{AXES, LATENT, TIME};
use sensai_engine::{Model, Recipe, Settings, Tensor};
use std::path::Path;

pub struct Flow {
    model: Model,
    /// Text rows per canvas, pads included.
    text: usize,
}

/// The timestep's sinusoids, cosines first, matching diffusers.
fn sinusoids(t: f32) -> Vec<f32> {
    let half = TIME as usize / 2;
    let at = |i: usize| t * 1000.0 * (-(10_000f32).ln() * i as f32 / half as f32).exp();
    let cos = (0..half).map(|i| at(i).cos());
    cos.chain((0..half).map(|i| at(i).sin())).collect()
}

/// Each axis's positions: text by index, then the grid.
fn positions(text: usize, side: usize) -> [Vec<u32>; AXES] {
    let mut ids: [Vec<u32>; AXES] = Default::default();
    for l in 0..text {
        [0, 0, 0, l]
            .iter()
            .zip(&mut ids)
            .for_each(|(&v, a)| a.push(v as u32));
    }
    for cell in 0..side * side {
        let at = [0, cell / side, cell % side, 0];
        at.iter().zip(&mut ids).for_each(|(&v, a)| a.push(v as u32));
    }
    ids
}

impl Flow {
    /// The diffusers model in `dir`, sized for one launch.
    pub fn open(
        dir: &Path,
        settings: &Settings,
        [seqs, text, rows]: [usize; 3],
        config: &Overrides,
    ) -> Result<Self, Error> {
        let t = transformer(dir, [seqs as u64, text as u64, rows as u64], config)?;
        let recipe = Recipe {
            pins: t.pins,
            ..Recipe::default()
        };
        let mut model = Model::open(t.graph, &recipe, settings)?;
        model.load(&t.weights)?;
        Ok(Self { model, text })
    }

    /// Velocity for square latents at time `t`.
    pub fn velocity(
        &mut self,
        latents: &[f32],
        text: &[f32],
        side: usize,
        t: f32,
    ) -> Result<Vec<f32>, Error> {
        let rows = side * side;
        let n = latents.len() / (rows * LATENT as usize);
        let width = text.len() / (n * self.text);
        let joint = self.text + rows;
        let mut inputs = vec![
            Tensor::f32(vec![n, rows, LATENT as usize], latents.to_vec()),
            Tensor::f32(vec![n, self.text, width], text.to_vec()),
            Tensor::f32(vec![n, TIME as usize], sinusoids(t).repeat(n)),
        ];
        for axis in positions(self.text, side) {
            inputs.push(Tensor::u32(vec![n, joint], axis.repeat(n)));
        }
        Ok(self.model.run(&inputs)?[0].floats()?.to_vec())
    }
}
