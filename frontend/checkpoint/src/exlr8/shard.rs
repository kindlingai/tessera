//! How an exlr8 file's names become HF names.

use crate::file::File;
use crate::offer::Offer;
use std::collections::HashMap;

/// `name` past `stem`, where `stem` is a whole first part.
pub(crate) fn past<'a>(name: &'a str, stem: &str) -> Option<&'a str> {
    let end = name.strip_prefix(stem)?;
    (end.is_empty() || end.starts_with('.')).then_some(end)
}

/// The text before a layer index in `f`'s names.
pub(crate) fn layer_prefix(f: &File) -> Option<String> {
    if let Some(prefix) = f.meta("tensor_prefix") {
        return Some(prefix.into());
    }
    // Else the most common `layers.` stem before a number.
    let slices = f.slices();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for s in &slices {
        for (at, stem) in s.name.match_indices("layers.") {
            let end = at + stem.len();
            if s.name[end..].starts_with(|c: char| c.is_ascii_digit()) {
                *counts.entry(&s.name[..end]).or_default() += 1;
                break;
            }
        }
    }
    let most = counts.into_iter().max_by_key(|&(stem, n)| (n, stem));
    most.map(|(stem, _)| stem.into())
}

/// One file, and how its names become HF names.
pub(crate) struct Shard {
    pub(crate) file: String,
    pub(crate) prefix: String,
    /// Stems that take other HF names.
    pub(crate) renames: Vec<(String, String)>,
    /// Widths of matrices stored at several, by stem.
    pub(crate) picked: Vec<(String, u32)>,
    /// Whether every tensor is whole, under its own name.
    pub(crate) plain: bool,
}

impl Shard {
    pub(crate) fn new(file: String, prefix: String) -> Self {
        Self {
            file,
            prefix,
            renames: Vec::new(),
            picked: Vec::new(),
            plain: false,
        }
    }

    pub(crate) fn name(&self, local: &str) -> String {
        for (stem, hf) in &self.renames {
            if let Some(end) = past(local, stem) {
                return format!("{hf}{end}");
            }
        }
        format!("{}{local}", self.prefix)
    }

    /// A stem's matrix, or `None` for an unpicked width.
    pub(crate) fn matrix<'a>(&self, stem: &'a str) -> Option<&'a str> {
        let Some((matrix, width)) = stem.rsplit_once(".k") else {
            return Some(stem);
        };
        let Ok(k) = width.parse::<u32>() else {
            return Some(stem);
        };
        self.picked
            .iter()
            .any(|(m, w)| m == matrix && *w == k)
            .then_some(matrix)
    }
}
