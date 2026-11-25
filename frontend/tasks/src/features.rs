//! Prompts as a decoder's residual rows, as FLUX.2 reads text.

use crate::Error;
use std::path::Path;
use tessera_bundle::Message;
use tessera_bundle::Overrides;
use tessera_bundle::taps::{Tapped, tapped};
use tessera_bundle::tokenize::hf::Hf;
use tessera_engine::progress::{Progress, Unit};
use tessera_engine::{Model, Recipe, Settings, Tensor};

pub struct Features {
    model: Model,
    tapped: Tapped,
    /// Rows per prompt, pads included.
    rows: usize,
}

impl Features {
    /// The model and tokenizer dirs, sized for one launch.
    pub fn open(
        dirs: [&Path; 2],
        settings: &Settings,
        layers: &[u64],
        [seqs, rows]: [usize; 2],
        config: &Overrides,
        progress: &Progress,
    ) -> Result<Self, Error> {
        let reading = progress.step("read model", Unit::Items, None);
        let t = tapped(dirs, [seqs as u64, rows as u64], layers, config)?;
        reading.finish();
        let recipe = Recipe {
            pins: t.pins.clone(),
            ..Recipe::default()
        };
        let graph = t.graph.clone();
        let model = Model::open_loaded(graph, &recipe, settings, &t.weights, progress)?;
        Ok(Self {
            model,
            tapped: t,
            rows,
        })
    }

    pub fn tokenizer(&self) -> &Hf {
        &self.tapped.tokenizer
    }

    /// The prompt as a user turn, without thinking.
    pub fn ids(&self, prompt: &str) -> Result<Vec<u32>, Error> {
        let message = Message::text("user", prompt);
        let text = self.tapped.template.render(&[message], false)?;
        let mut ids = self.tapped.tokenizer.encode_special(&text)?;
        ids.truncate(self.rows);
        Ok(ids)
    }

    /// Each prompt's `rows` rows, padded. Pads read only the prompt.
    pub fn encode(&mut self, prompts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        let (n, rows) = (prompts.len(), self.rows);
        let (mut tokens, mut lists) = (Vec::with_capacity(n * rows), Vec::new());
        for p in prompts {
            let mut ids = self.ids(p)?;
            let len = ids.len().max(1);
            ids.resize(rows, self.tapped.pad);
            tokens.extend(ids);
            for row in 0..rows {
                let last = row.min(len - 1);
                lists.extend((0..rows).map(|j| if j <= last { j as u32 } else { rows as u32 }));
            }
        }
        let positions = (0..n).flat_map(|_| 0..rows as u32).collect();
        let inputs = [
            Tensor::u32(vec![n, rows], tokens),
            Tensor::u32(vec![n, rows], positions),
            Tensor::u32(vec![n, rows, rows], lists),
        ];
        let out = self.model.run(&inputs)?[0].floats()?.to_vec();
        Ok(out.chunks(out.len() / n).map(<[f32]>::to_vec).collect())
    }
}
