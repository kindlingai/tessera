//! Pairs the graph's weights with the slices a checkpoint offers.

use checkpoint::Slice;
use graph::{Graph, PathSeg};
use std::collections::BTreeMap;
use std::fmt;

/// A weight path to a slice stem, `*` the layer index.
pub type Rule = (String, String);

/// A slice claimed for a weight, whole or by rows.
#[derive(Clone, Debug, PartialEq)]
pub struct Claim {
    pub name: String,
    /// `(first, count)` rows of a slice stacked over the layers.
    pub rows: Option<(usize, usize)>,
}

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

/// Endings that name one piece of a matrix.
const ENDINGS: &[&str] = &["weight", "codes", "suh", "svh"];

/// Fields holding one part of a matrix.
const PARTS: &[&str] = &["w", "suh", "svh"];

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

/// A slice name taken apart: layer index, telling words, ending.
struct Words<'a> {
    index: Option<usize>,
    words: Vec<&'a str>,
    ending: Option<&'a str>,
}

fn words(name: &str) -> Words<'_> {
    let (stem, ending) = match name.rsplit_once('.') {
        Some((stem, last)) if ENDINGS.contains(&last) => (stem, Some(last)),
        _ => (name, None),
    };
    let mut index = None;
    let mut words = Vec::new();
    for part in stem.split(['.', '_', '-']) {
        if let Ok(n) = part.parse::<usize>() {
            index = index.or(Some(n));
        } else if !part.is_empty() && !NOISE.contains(&part) {
            words.push(part);
        }
    }
    Words {
        index,
        words,
        ending,
    }
}

/// A weight path taken apart: index, role field, matrix part.
struct Role<'a> {
    text: String,
    index: Option<usize>,
    field: &'a str,
    part: Option<&'a str>,
}

fn role(path: &[PathSeg]) -> Role<'_> {
    let seg = |p: &PathSeg| match p {
        PathSeg::Field(f) => f.clone(),
        PathSeg::Index(i) => i.to_string(),
    };
    let index = path.iter().find_map(|p| match p {
        PathSeg::Index(i) => Some(*i as usize),
        PathSeg::Field(_) => None,
    });
    let mut fields = path.iter().rev().filter_map(|p| match p {
        PathSeg::Field(f) => Some(f.as_str()),
        PathSeg::Index(_) => None,
    });
    let last = fields.next().unwrap_or("");
    let (field, part, rest) = match PARTS.contains(&last) {
        true => (
            fields.next().unwrap_or(""),
            Some(last),
            &path[..path.len() - 1],
        ),
        false => (last, None, path),
    };
    let text = rest.iter().map(seg).collect::<Vec<_>>().join(".");
    Role {
        text,
        index,
        field,
        part,
    }
}

/// A matrix part prefers stored codes to their decoded view.
fn preferred<'a>(part: Option<&str>, mut found: Vec<&'a Slice>) -> Vec<&'a Slice> {
    let codes = |s: &&Slice| s.name.ends_with(".codes");
    if part == Some("w") && found.iter().any(codes) {
        found.retain(codes);
    }
    found
}

/// Whether `ending` holds `part`, or a whole tensor.
fn holds(ending: Option<&str>, part: Option<&str>) -> bool {
    match part {
        None => matches!(ending, None | Some("weight")),
        Some("w") => matches!(ending, Some("weight" | "codes")),
        Some(p) => ending == Some(p),
    }
}

/// The weight's shape, where every axis is a constant.
fn wanted(g: &Graph, shape: &graph::Shape) -> Option<Vec<usize>> {
    let axis = |e: &graph::DimExpr| e.value(&g.dims, &|_| None).map(|n| n as usize);
    shape.0.iter().map(axis).collect()
}

/// Axes of more than one element.
fn wide(shape: &[usize]) -> Vec<usize> {
    shape.iter().copied().filter(|n| *n > 1).collect()
}

/// Whether `slice` has the shape, any when unknown.
fn shaped(want: Option<&Vec<usize>>, slice: &Slice) -> bool {
    want.is_none_or(|w| wide(w) == wide(&slice.shape))
}

/// The claim on a slice, by rows when stacked.
fn stacked(want: Option<&Vec<usize>>, slice: &Slice, index: Option<usize>) -> Option<Claim> {
    let name = slice.name.clone();
    if shaped(want, slice) {
        return Some(Claim { name, rows: None });
    }
    let (want, index) = (want?, index?);
    let (first, rest) = want.split_first()?;
    let (rows, tail) = slice.shape.split_first()?;
    let fits = rows % first == 0 && (index + 1) * first <= *rows && wide(rest) == wide(tail);
    fits.then(|| Claim {
        name,
        rows: Some((index * first, *first)),
    })
}

/// Whether `pattern` names `path`, `*` taking the index.
fn fits(pattern: &str, path: &str, index: Option<usize>) -> bool {
    match index {
        Some(i) => pattern.replace('*', &i.to_string()) == path,
        None => pattern == path,
    }
}

/// The slice for every weight, in the graph's order.
pub fn resolve(g: &Graph, slices: &[Slice], rules: &[&[Rule]]) -> Result<Vec<Claim>, Unresolved> {
    let roles: BTreeMap<&str, &[&[&str]]> = ROLES.iter().copied().collect();
    let parsed: Vec<Words> = slices.iter().map(|s| words(&s.name)).collect();
    let mut out = Vec::with_capacity(g.weights.len());
    let mut failed = Unresolved::default();
    for w in &g.weights {
        let r = role(&w.path);
        let want = wanted(g, &w.ty.shape);
        // The last rule naming the path decides.
        let ruled = (rules.iter().flat_map(|r| r.iter()))
            .rev()
            .find(|(pattern, _)| fits(pattern, &r.text, r.index));
        if let Some((_, stem)) = ruled {
            let (stem, layered) = match stem.strip_suffix("[*]") {
                Some(stem) => (stem.to_string(), true),
                None => (stem.clone(), false),
            };
            let stem = match r.index {
                Some(i) => stem.replace('*', &i.to_string()),
                None => stem,
            };
            let named: Vec<&Slice> = (slices.iter().zip(&parsed))
                .filter(|(s, w)| holds(w.ending, r.part) && stem_of(&s.name, w.ending) == stem)
                .map(|(s, _)| s)
                .collect();
            let named = preferred(r.part, named);
            let claim = named.first().and_then(|s| match layered {
                true => stacked(want.as_ref(), s, r.index),
                false => shaped(want.as_ref(), s).then(|| Claim {
                    name: s.name.clone(),
                    rows: None,
                }),
            });
            match (claim, named.first()) {
                (Some(c), _) => out.push(c),
                (None, Some(s)) => failed.misfit.push((r.text, s.name.clone())),
                (None, None) => failed.missing.push(r.text),
            }
            continue;
        }
        let own = [r.field];
        let fallback = [own.as_slice()];
        let spellings = roles.get(r.field).copied().unwrap_or(&fallback);
        let found: Vec<&Slice> = (slices.iter().zip(&parsed))
            .filter(|(_, w)| w.index == r.index && holds(w.ending, r.part))
            .filter(|(s, w)| spellings.contains(&w.words.as_slice()) && shaped(want.as_ref(), s))
            .map(|(s, _)| s)
            .collect();
        let found: Vec<&str> = preferred(r.part, found)
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        match found.as_slice() {
            [one] => out.push(Claim {
                name: (*one).into(),
                rows: None,
            }),
            [] => failed.missing.push(r.text),
            many => failed
                .ambiguous
                .push((r.text, many.iter().map(|s| (*s).into()).collect())),
        }
    }
    match failed == Unresolved::default() {
        true => Ok(out),
        false => Err(failed),
    }
}

/// `name` without its ending.
fn stem_of<'a>(name: &'a str, ending: Option<&str>) -> &'a str {
    match ending {
        Some(e) => &name[..name.len() - e.len() - 1],
        None => name,
    }
}

/// Rules from a JSON object: path pattern to slice stem.
pub fn rules(json: &str) -> Result<Vec<Rule>, serde_json::Error> {
    let map: BTreeMap<String, String> = serde_json::from_str(json)?;
    Ok(map.into_iter().collect())
}

/// The repository's rules for a model, where it needs any.
pub fn repo(model: &str) -> Option<Vec<Rule>> {
    let json = match model {
        "eg2" => include_str!("../names/eg2.json"),
        _ => return None,
    };
    Some(rules(json).expect("the repository's rules parse"))
}
