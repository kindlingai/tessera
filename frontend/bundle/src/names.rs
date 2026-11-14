//! Pairs the graph's weights with the slices a checkpoint offers.

use checkpoint::Slice;
use graph::{Graph, PathSeg};
use std::collections::BTreeMap;
use std::fmt;

/// A weight path to a slice name, `*` the layer index.
pub type Rule = (String, String);

/// Why the weights and the slices do not pair off.
#[derive(Debug, Default, PartialEq)]
pub struct Unresolved {
    /// Weights no slice fits.
    pub missing: Vec<String>,
    /// Weights several slices fit, with the slices.
    pub ambiguous: Vec<(String, Vec<String>)>,
    /// Weights a rule sent to a slice of another shape.
    pub misfit: Vec<(String, String)>,
}

impl fmt::Display for Unresolved {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for path in &self.missing {
            writeln!(f, "{path}: no slice fits")?;
        }
        for (path, slices) in &self.ambiguous {
            writeln!(f, "{path}: fits {}", slices.join(", "))?;
        }
        for (path, slice) in &self.misfit {
            writeln!(f, "{path}: {slice} has another shape")?;
        }
        Ok(())
    }
}

impl std::error::Error for Unresolved {}

/// Name parts that say nothing about a weight's role.
const NOISE: &[&str] = &[
    "weight",
    "model",
    "layers",
    "blk",
    "self",
    "proj",
    "mlp",
    "h",
    "transformer",
    "language",
];

/// Spellings of each field's role, as word sets.
const ROLES: &[(&str, &[&[&str]])] = &[
    ("q", &[&["q"], &["attn", "q"]]),
    ("k", &[&["k"], &["attn", "k"]]),
    ("v", &[&["v"], &["attn", "v"]]),
    ("o", &[&["o"], &["attn", "o"], &["attn", "output"]]),
    ("gate", &[&["gate"], &["ffn", "gate"]]),
    ("up", &[&["up"], &["ffn", "up"]]),
    ("down", &[&["down"], &["ffn", "down"]]),
    (
        "ln1",
        &[&["input", "layernorm"], &["attn", "norm"], &["ln1"]],
    ),
    (
        "ln2",
        &[
            &["post", "attention", "layernorm"],
            &["ffn", "norm"],
            &["ln2"],
        ],
    ),
    ("q_norm", &[&["attn", "q", "norm"], &["q", "norm"]]),
    ("k_norm", &[&["attn", "k", "norm"], &["k", "norm"]]),
    (
        "embed",
        &[
            &["embed", "tokens"],
            &["token", "embd"],
            &["embed"],
            &["wte"],
        ],
    ),
    (
        "norm",
        &[
            &["norm"],
            &["output", "norm"],
            &["final", "layernorm"],
            &["ln", "f"],
        ],
    ),
];

/// A name's layer index, if any, and its telling words.
fn words(name: &str) -> (Option<usize>, Vec<&str>) {
    let mut index = None;
    let mut out = Vec::new();
    for part in name.split(['.', '_', '-']) {
        if let Ok(n) = part.parse::<usize>() {
            index = index.or(Some(n));
        } else if !part.is_empty() && !NOISE.contains(&part) {
            out.push(part);
        }
    }
    (index, out)
}

/// A weight path as text, `layers.3.q`.
fn text(path: &[PathSeg]) -> String {
    let seg = |p: &PathSeg| match p {
        PathSeg::Field(f) => f.clone(),
        PathSeg::Index(i) => i.to_string(),
    };
    path.iter().map(seg).collect::<Vec<_>>().join(".")
}

/// A path's layer index and field name.
fn parts(path: &[PathSeg]) -> (Option<usize>, &str) {
    let index = path.iter().find_map(|p| match p {
        PathSeg::Index(i) => Some(*i as usize),
        PathSeg::Field(_) => None,
    });
    let field = path.iter().rev().find_map(|p| match p {
        PathSeg::Field(f) => Some(f.as_str()),
        PathSeg::Index(_) => None,
    });
    (index, field.unwrap_or(""))
}

/// The weight's shape, where every axis is a constant.
fn wanted(g: &Graph, shape: &graph::Shape) -> Option<Vec<usize>> {
    let axis = |e: &graph::DimExpr| e.value(&g.dims, &|_| None).map(|n| n as usize);
    shape.0.iter().map(axis).collect()
}

/// Whether `slice` has the shape, any when unknown.
fn shaped(want: Option<&Vec<usize>>, slice: &Slice) -> bool {
    want.is_none_or(|w| *w == slice.shape)
}

/// Whether `pattern` names `path`, `*` taking the index.
fn fits(pattern: &str, path: &str, index: Option<usize>) -> bool {
    match index {
        Some(i) => pattern.replace('*', &i.to_string()) == path,
        None => pattern == path,
    }
}

/// The slice for every weight, in the graph's order.
pub fn resolve(g: &Graph, slices: &[Slice], rules: &[&[Rule]]) -> Result<Vec<String>, Unresolved> {
    let roles: BTreeMap<&str, &[&[&str]]> = ROLES.iter().copied().collect();
    let parsed: Vec<(Option<usize>, Vec<&str>)> = slices.iter().map(|s| words(&s.name)).collect();
    let mut out = Vec::with_capacity(g.weights.len());
    let mut failed = Unresolved::default();
    for w in &g.weights {
        let (path, (index, field)) = (text(&w.path), parts(&w.path));
        let want = wanted(g, &w.ty.shape);
        // The last rule naming the path decides.
        let ruled = (rules.iter().flat_map(|r| r.iter()))
            .rev()
            .find(|(pattern, _)| fits(pattern, &path, index));
        if let Some((_, slice)) = ruled {
            let name = match index {
                Some(i) => slice.replace('*', &i.to_string()),
                None => slice.clone(),
            };
            match slices.iter().find(|s| s.name == name) {
                Some(s) if shaped(want.as_ref(), s) => out.push(name),
                Some(_) => failed.misfit.push((path, name)),
                None => failed.missing.push(path),
            }
            continue;
        }
        let own = [field];
        let fallback = [own.as_slice()];
        let spellings = roles.get(field).copied().unwrap_or(&fallback);
        let found: Vec<&str> = (slices.iter().zip(&parsed))
            .filter(|(_, (at, w))| *at == index && spellings.contains(&w.as_slice()))
            .filter(|(s, _)| shaped(want.as_ref(), s))
            .map(|(s, _)| s.name.as_str())
            .collect();
        match found.as_slice() {
            [one] => out.push((*one).into()),
            [] => failed.missing.push(path),
            many => failed
                .ambiguous
                .push((path, many.iter().map(|s| (*s).into()).collect())),
        }
    }
    match failed == Unresolved::default() {
        true => Ok(out),
        false => Err(failed),
    }
}

/// Rules from a JSON object: path pattern to slice pattern.
pub fn rules(json: &str) -> Result<Vec<Rule>, serde_json::Error> {
    let map: BTreeMap<String, String> = serde_json::from_str(json)?;
    Ok(map.into_iter().collect())
}
