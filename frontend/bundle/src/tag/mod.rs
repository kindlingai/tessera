// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

//! spaCy's part-of-speech tagger, run as a tessera graph.

mod features;
mod symbols;
#[cfg(test)]
mod tag_test;
mod thinc;

use crate::tokenize::Error;
use crate::tokenize::spacy::{Spacy, Word};
use features::{Features, thinc_hash};
use graph::{Graph, PathSeg};
use interp::{Tensor, bind, run};
use models::tagger::{Config, KEYS, PAD, PIECES, path, program};
use std::path::Path;
use thinc::{Array, Node};

/// Tokens past this need a longer graph.
const MAX_CONTEXT: u64 = 4096;

pub struct Tagger {
    tokenizer: Spacy,
    features: Features,
    graph: Graph,
    weights: Vec<Tensor>,
    tags: Vec<String>,
    /// Per feature: hash seed, rows, and first joined row.
    seeds: Vec<u32>,
    rows: Vec<u64>,
    offsets: Vec<u64>,
}

fn nodes(dir: &Path, part: &str) -> Result<Vec<Node>, Error> {
    thinc::read(&std::fs::read(dir.join(part).join("model"))?)
}

fn param<'a>(n: &'a Node, name: &str) -> Result<&'a Array, Error> {
    n.params
        .get(name)
        .ok_or_else(|| format!("{} without {name}", n.name).into())
}

/// Thinc's `[units, pieces, ..]` rows as `[pieces, units, ..]`.
fn by_piece(a: &Array) -> Vec<f32> {
    let (units, pieces) = (a.shape[0], a.shape[1]);
    let row = a.data.len() / (units * pieces);
    let mut out = Vec::with_capacity(a.data.len());
    for p in 0..pieces {
        for u in 0..units {
            let at = (u * pieces + p) * row;
            out.extend_from_slice(&a.data[at..at + row]);
        }
    }
    out
}

impl Tagger {
    /// Reads a spaCy pipeline folder: tokenizer, tok2vec, tagger and vocab.
    pub fn open(dir: &Path) -> Result<Self, Error> {
        let t2v = nodes(dir, "tok2vec")?;
        let mut tables: Vec<&Node> = t2v.iter().filter(|n| n.name == "hashembed").collect();
        tables.sort_by_key(|n| n.attrs.get("column").copied());
        let maxouts: Vec<&Node> = t2v.iter().filter(|n| n.name == "maxout").collect();
        let norms: Vec<&Node> = t2v.iter().filter(|n| n.name == "layernorm").collect();
        let tagger = nodes(dir, "tagger")?;
        let softmax = tagger
            .iter()
            .find(|n| n.name == "softmax")
            .ok_or("tagger without softmax")?;
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("tagger/cfg"))?)?;
        let tags: Vec<String> = serde_json::from_value(cfg["labels"].clone())?;

        let mut weights: Vec<(Vec<PathSeg>, Vec<f32>)> = Vec::new();
        let (mut seeds, mut rows, mut offsets, mut joined) = (vec![], vec![], vec![], vec![]);
        for t in &tables {
            let e = param(t, "E")?;
            seeds.push(*t.attrs.get("seed").ok_or("hashembed without seed")? as u32);
            offsets.push(rows.iter().sum());
            rows.push(e.shape[0] as u64);
            joined.extend_from_slice(&e.data);
        }
        weights.push((vec![PathSeg::Field("embed".into())], joined));
        for (i, (m, n)) in maxouts.iter().zip(&norms).enumerate() {
            let w = param(m, "W")?;
            if w.shape.get(1) != Some(&(PIECES as usize)) {
                return Err(format!("maxout with {:?} pieces", w.shape.get(1)).into());
            }
            let i = Some(i as u32);
            weights.push((path("maxout", i, "w"), by_piece(w)));
            weights.push((path("maxout", i, "b"), by_piece(param(m, "b")?)));
            weights.push((path("norm", i, "g"), param(n, "G")?.data.clone()));
            weights.push((path("norm", i, "b"), param(n, "b")?.data.clone()));
        }
        weights.push((path("tagger", None, "w"), param(softmax, "W")?.data.clone()));
        weights.push((path("tagger", None, "b"), param(softmax, "b")?.data.clone()));

        let config = Config {
            rows: rows.clone(),
            width: param(norms.first().ok_or("tok2vec without layers")?, "G")?
                .data
                .len() as u64,
            depth: maxouts.len() as u64 - 1,
            tags: tags.len() as u64,
            max_context: MAX_CONTEXT,
        };
        let graph = program(&config);
        let find = |p: &[PathSeg]| {
            let (_, v) = weights.iter().find(|(path, _)| path == p)?;
            let v = v.clone();
            Some(Tensor::f32(vec![v.len()], v))
        };
        let lexeme = thinc::lexeme_norms(&std::fs::read(dir.join("vocab/lookups.bin"))?)?;
        Ok(Self {
            tokenizer: Spacy::open(dir)?,
            features: Features::new(lexeme),
            weights: bind(&graph, find)?,
            graph,
            tags,
            seeds,
            rows,
            offsets,
        })
    }

    /// Each token of `text` with its tag.
    pub fn tag(&self, text: &str) -> Result<Vec<(Word, &str)>, Error> {
        let words = self.tokenizer.words(text)?;
        let n = words.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n as u64 > MAX_CONTEXT {
            return Err(format!("{n} tokens, past {MAX_CONTEXT}").into());
        }
        let mut keys = Vec::with_capacity(n * self.seeds.len() * KEYS as usize);
        for w in &words {
            let feats = self.features.of(&text[w.span.clone()], w);
            for (c, f) in feats.iter().enumerate() {
                for k in thinc_hash(*f, self.seeds[c]) {
                    keys.push((u64::from(k) % self.rows[c] + self.offsets[c]) as u32);
                }
            }
        }
        let pad = PAD as usize;
        let padded = n + 2 * pad;
        // Outside the text, rows read zero: index `n` or `padded`.
        let mut sources = Vec::with_capacity(padded);
        let mut around = Vec::with_capacity(3 * padded);
        for j in 0..padded {
            let inside = (pad..pad + n).contains(&j);
            sources.push(if inside { (j - pad) as u32 } else { n as u32 });
            let prev = if j == 0 { padded } else { j - 1 };
            let next = if j + 1 == padded { padded } else { j + 1 };
            around.extend([prev as u32, j as u32, next as u32]);
        }
        let inputs = [
            Tensor::u32(vec![1, n, keys.len() / n], keys),
            Tensor::u32(vec![1, padded], sources),
            Tensor::u32(vec![1, padded, 3], around),
        ];
        let out = run(&self.graph, &self.weights, 0, &inputs)?;
        let logits = out[0].floats()?;
        let width = self.tags.len();
        let mut tagged = Vec::with_capacity(n);
        for (w, row) in words.into_iter().zip(logits.chunks(width).skip(pad)) {
            let best = (0..width)
                .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                .unwrap_or(0);
            tagged.push((w, self.tags[best].as_str()));
        }
        Ok(tagged)
    }
}
