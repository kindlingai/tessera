//! Texts to unit-length EmbeddingGemma 2 vectors.

use crate::Error;
use crate::api::{Embed, Embedded, Tokens};
use crate::worker::Worker;
use std::path::{Path, PathBuf};
use tessera_bundle::Overrides;
use tessera_bundle::encoder::encoder;
use tessera_bundle::tokenize::hf::Hf;
use tessera_engine::{Model, Recipe, Settings, Tensor};

/// The launch sizes an embedder plans for.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// Tokens per text, `<bos>` and `<eos>` included.
    pub max_tokens: usize,
    /// The most texts in one launch.
    pub seqs: usize,
    /// The most padded rows in one launch.
    pub rows: usize,
}

/// EmbeddingGemma 2's sizes.
pub const EG2: Shape = Shape {
    max_tokens: 1024,
    seqs: 16,
    rows: 4096,
};

pub struct Embedder {
    model: Model,
    tokenizer: Hf,
    /// Texts of unlike length share launches, padded.
    padded: bool,
    shape: Shape,
}

/// Texts that share a launch: shortest first, or equal unpadded.
fn groups(ids: &[Vec<u32>], padded: bool, shape: Shape) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|&i| ids[i].len());
    let mut out: Vec<Vec<usize>> = Vec::new();
    for i in order {
        let len = ids[i].len();
        let joins = out.last().is_some_and(|g| {
            let fits = match padded {
                true => (g.len() + 1) * len <= shape.rows,
                false => ids[g[0]].len() == len,
            };
            g.len() < shape.seqs && fits
        });
        match out.last_mut() {
            Some(g) if joins => g.push(i),
            _ => out.push(vec![i]),
        }
    }
    out
}

/// Each row scaled to unit length.
fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

impl Embedder {
    /// Loads the exlr8 release in `dir` where `settings` say.
    pub fn open(dir: &Path, settings: &Settings, config: &Overrides) -> Result<Self, Error> {
        Self::shaped(EG2, dir, settings, config)
    }

    /// The same, planned for `shape`.
    pub fn shaped(
        shape: Shape,
        dir: &Path,
        settings: &Settings,
        config: &Overrides,
    ) -> Result<Self, Error> {
        // GPUs lack a fast kernel for listed keys.
        let padded = settings.gpu().is_none();
        let at = [shape.seqs as u64, shape.max_tokens as u64];
        let e = encoder(dir, at, padded, config)?;
        let mut model = Model::open(e.graph, &Recipe::default(), settings)?;
        model.load(&e.weights)?;
        Ok(Self {
            model,
            tokenizer: e.tokenizer,
            padded,
            shape,
        })
    }

    /// The backend and hardware the model runs on.
    pub fn device(&self) -> String {
        self.model.device()
    }

    /// Ids of `text` between `<bos>` and `<eos>`, cut to fit.
    pub fn tokens(&self, text: &str) -> Result<Vec<u32>, Error> {
        let mut ids = self.tokenizer.encode_special(text)?;
        let most = self.shape.max_tokens;
        if ids.len() > most {
            let eos = ids[ids.len() - 1];
            ids.truncate(most - 1);
            ids.push(eos);
        }
        Ok(ids)
    }

    /// One unit vector per text, in order.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        let mut ids = Vec::with_capacity(texts.len());
        for t in texts {
            ids.push(self.tokens(t)?);
        }
        self.embed_ids(&ids)
    }

    /// The same for ids, a few texts per launch.
    pub fn embed_ids(&mut self, ids: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, Error> {
        for (at, t) in ids.iter().enumerate() {
            if t.is_empty() || t.len() > self.shape.max_tokens {
                return Err(format!("text {at} has {} tokens", t.len()).into());
            }
        }
        let mut out = vec![Vec::new(); ids.len()];
        for group in groups(ids, self.padded, self.shape) {
            let got = self.launch(ids, &group)?;
            for (i, v) in group.into_iter().zip(got) {
                out[i] = v;
            }
        }
        Ok(out)
    }

    /// One launch of `group`, padded to its longest text.
    fn launch(&mut self, ids: &[Vec<u32>], group: &[usize]) -> Result<Vec<Vec<f32>>, Error> {
        let (n, len) = (
            group.len(),
            group.iter().map(|&i| ids[i].len()).max().unwrap_or(1),
        );
        let mut tokens = vec![0; n * len];
        for (row, &i) in tokens.chunks_mut(len).zip(group) {
            row[..ids[i].len()].copy_from_slice(&ids[i]);
        }
        let positions = (0..n).flat_map(|_| 0..len as u32).collect();
        let shape = vec![n, len];
        let mut inputs = vec![
            Tensor::u32(shape.clone(), tokens),
            Tensor::u32(shape, positions),
        ];
        if self.padded {
            // Each row lists its own text's keys. `len` lists none.
            let mut lists = Vec::with_capacity(n * len * len);
            for &i in group {
                let own = ids[i].len() as u32;
                let list = (0..len as u32).map(|j| if j < own { j } else { len as u32 });
                let list: Vec<u32> = list.collect();
                (0..len).for_each(|_| lists.extend_from_slice(&list));
            }
            inputs.push(Tensor::u32(vec![n, len, len], lists));
        }
        let got = self.model.run(&inputs)?;
        let floats = got[0].floats()?;
        let per = floats.len() / n;
        let mut out = Vec::with_capacity(n);
        for (&i, block) in group.iter().zip(floats.chunks(per)) {
            if !self.padded {
                out.push(unit(block.to_vec()));
                continue;
            }
            // The text's own rows. The unit scale divides.
            let width = per / len;
            let mut sum = vec![0.0f32; width];
            for row in block.chunks(width).take(ids[i].len()) {
                sum.iter_mut().zip(row).for_each(|(s, v)| *s += v);
            }
            out.push(unit(sum));
        }
        Ok(out)
    }
}

/// An embedder on its own thread.
pub type Threaded = Worker<Embedder>;

impl Worker<Embedder> {
    /// Opens the release in `dir` on a new thread.
    pub fn open(
        shape: Shape,
        dir: PathBuf,
        settings: Settings,
        config: Overrides,
    ) -> Result<Self, Error> {
        Worker::spawn(move || {
            let embedder = Embedder::shaped(shape, &dir, &settings, &config)?;
            eprintln!("{}: embedder loaded", embedder.device());
            Ok(embedder)
        })
    }
}

impl Embed for Threaded {
    fn embed(&self, texts: &[String]) -> Result<Embedded, Error> {
        let texts = texts.to_vec();
        self.run(move |embedder| {
            let mut ids = Vec::with_capacity(texts.len());
            for t in &texts {
                ids.push(embedder.tokens(t)?);
            }
            Ok(Embedded {
                tokens: ids.iter().map(Vec::len).sum(),
                vectors: embedder.embed_ids(&ids)?,
            })
        })?
    }
}

impl Tokens for Threaded {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let text = text.to_string();
        self.run(move |embedder| embedder.tokens(&text))?
    }

    fn decode(&self, ids: &[u32]) -> String {
        let ids = ids.to_vec();
        let decoded = self.run(move |embedder| embedder.tokenizer.decode(&ids));
        decoded.unwrap_or_default()
    }
}
