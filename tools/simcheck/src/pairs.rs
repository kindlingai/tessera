//! Window pairs whose embeddings are too close.

use crate::windows::{Window, overlap};
use anyhow::Result;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// Two files' closest window pair, and how many pairs pass.
pub struct Hit {
    pub a: usize,
    pub b: usize,
    pub score: f32,
    pub count: usize,
}

/// File pairs allowed to hold near-copies, in either order.
#[derive(Deserialize)]
pub struct Allow {
    pairs: Vec<(String, String)>,
}

impl Allow {
    pub fn parse(text: &str) -> Result<Allow> {
        Ok(toml::from_str(text)?)
    }

    /// The index of the pair that allows `a` beside `b`.
    fn find(&self, a: &str, b: &str) -> Option<usize> {
        self.pairs
            .iter()
            .position(|(x, y)| (x == a && y == b) || (x == b && y == a))
    }

    /// Pairs that allowed nothing, given which ones did.
    pub fn stale(&self, used: &[bool]) -> Vec<&(String, String)> {
        self.pairs
            .iter()
            .zip(used)
            .filter(|(_, u)| !**u)
            .map(|(p, _)| p)
            .collect()
    }
}

/// Files in different languages may mirror each other.
fn same_kind(a: &Window, b: &Window) -> bool {
    Path::new(&a.path).extension() == Path::new(&b.path).extension()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Hits at `min` or above, and which allow pairs matched.
pub fn near(
    wins: &[Window],
    embs: &[Vec<f32>],
    window: usize,
    min: f32,
    allow: &Allow,
) -> (Vec<Hit>, Vec<bool>) {
    let found: Vec<(usize, usize, f32)> = (0..wins.len())
        .into_par_iter()
        .flat_map_iter(|i| {
            (i + 1..wins.len())
                .filter(move |&j| same_kind(&wins[i], &wins[j]))
                .filter(move |&j| !overlap(&wins[i], &wins[j], window))
                .map(move |j| (i, j, dot(&embs[i], &embs[j])))
                .filter(move |&(_, _, s)| s >= min)
                .collect::<Vec<_>>()
        })
        .collect();
    let mut by_files: BTreeMap<(&str, &str), Hit> = BTreeMap::new();
    let mut used = vec![false; allow.pairs.len()];
    for (a, b, score) in found {
        let files = (wins[a].path.as_str(), wins[b].path.as_str());
        if let Some(i) = allow.find(files.0, files.1) {
            used[i] = true;
            continue;
        }
        let hit = by_files.entry(files).or_insert(Hit {
            a,
            b,
            score,
            count: 0,
        });
        hit.count += 1;
        if score > hit.score {
            (hit.a, hit.b, hit.score) = (a, b, score);
        }
    }
    (by_files.into_values().collect(), used)
}
