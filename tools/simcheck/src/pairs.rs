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

/// Separate sums in `dot` let the compiler vectorize.
const LANES: usize = 8;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sums = [0.0f32; LANES];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<LANES>(), b.as_chunks::<LANES>());
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    for (x, y) in ca.iter().zip(cb) {
        for k in 0..LANES {
            sums[k] += x[k] * y[k];
        }
    }
    sums.iter().sum::<f32>() + tail
}

/// A small id for each distinct value of `of`.
fn ids<'a>(wins: &'a [Window], of: impl Fn(&'a Window) -> &'a str) -> Vec<usize> {
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    let mut out = Vec::with_capacity(wins.len());
    for w in wins {
        let next = seen.len();
        out.push(*seen.entry(of(w)).or_insert(next));
    }
    out
}

/// Hits at `min` or above, and which allow pairs matched.
pub fn near(
    wins: &[Window],
    embs: &[Vec<f32>],
    (window, min): (usize, f32),
    allow: &Allow,
    focus: Option<&[bool]>,
) -> (Vec<Hit>, Vec<bool>) {
    let file = ids(wins, |w| w.path.as_str());
    // Files in different languages may mirror each other.
    let kind = ids(wins, |w| {
        Path::new(&w.path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
    });
    let wanted = |i: usize| focus.is_none_or(|f| f[i]);
    let found: Vec<(usize, usize, f32)> = (0..wins.len())
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut out = Vec::new();
            for j in i + 1..wins.len() {
                if kind[i] != kind[j] || !(wanted(i) || wanted(j)) {
                    continue;
                }
                if file[i] == file[j] && overlap(&wins[i], &wins[j], window) {
                    continue;
                }
                let s = dot(&embs[i], &embs[j]);
                if s >= min {
                    out.push((i, j, s));
                }
            }
            out
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
