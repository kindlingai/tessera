//! How an exlr8 file's names become HF names.

use crate::exlr8_experts as experts;
use crate::file::{Error, File};
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

/// Names in the `dense/` layout.
pub(crate) struct Dense {
    /// Before a layer's index, from the files' `tensor_prefix`.
    pub(crate) prefix: String,
    /// Prefix added to unrenamed tensors in `VOCAB`.
    pub(crate) vocab_prefix: String,
    /// Vocabulary file stems and their HF names.
    pub(crate) vocab: Vec<(String, String)>,
    /// Single dense file when layers are not split per file.
    pub(crate) single: Option<String>,
    /// Files that hold whole HF names.
    pub(crate) extras: Vec<String>,
    /// Whether the extras hold plain tensors only.
    pub(crate) plain: bool,
    /// The manifest's width for each `(layer, matrix)` stored at several.
    pub(crate) widths: HashMap<(usize, String), u32>,
    /// The manifest's expert widths, by layer and projection.
    pub(crate) experts: experts::Widths,
}

impl Dense {
    /// Layer `layer`'s stored name for `local`, at its picked width.
    pub(crate) fn stored(&self, layer: usize, local: &str) -> String {
        let Some((matrix, part)) = local.rsplit_once('.') else {
            return local.into();
        };
        match self.widths.get(&(layer, matrix.to_string())) {
            Some(k) => format!("{matrix}.k{k}.{part}"),
            None => local.into(),
        }
    }
}

/// Group prefixes for `qwen3_vl` releases.
const QWEN3_VL: &[(&str, &str)] = &[
    ("text", "model.language_model.layers."),
    ("vision", "model.visual.blocks."),
];

/// Group prefixes for `embedding_gemma2` releases.
const EG2: &[(&str, &str)] = &[
    ("text", "language_model.layers."),
    ("vision", "vision_tower.encoder.layers."),
    ("audio", "audio_tower.layers."),
];

/// Multi-group dense layout: vocab, global, and per-group layer files.
pub(crate) struct Groups {
    dir: String,
    groups: &'static [(&'static str, &'static str)],
    vocab_prefix: String,
    vocab: Vec<(String, String)>,
}

impl Groups {
    /// Default root layout when opening lazily before reading files.
    pub(crate) fn fallback() -> Self {
        Self {
            dir: String::new(),
            groups: EG2,
            vocab_prefix: "language_model.".into(),
            vocab: Vec::new(),
        }
    }

    /// Reads a multi-group layout from its vocab shard metadata.
    pub(crate) fn read(f: &File, dir: &str) -> Result<Option<Self>, Error> {
        let profile = f.meta("model_profile").unwrap_or("");
        let (groups, default_prefix, default_renames): (_, &str, &[(&str, &str)]) = match profile {
            "qwen3_vl" => (QWEN3_VL, "model.language_model.", &[("lm_head", "lm_head")]),
            "embedding_gemma2" => (EG2, "language_model.", &[]),
            "" if dir.is_empty() => (EG2, "language_model.", &[]),
            _ => return Ok(None),
        };
        let hf = f.meta("hf_names");
        let full = f
            .slices()
            .iter()
            .any(|s| s.name.starts_with("model.") || s.name == "lm_head.weight");
        let (vocab, vocab_prefix) = match hf {
            Some(json) => {
                let map = serde_json::from_str::<HashMap<String, String>>(json)
                    .map_err(|e| Error::DType(format!("exlr8-vocab hf_names: {e}")))?;
                (map.into_iter().collect(), String::new())
            }
            None if full => (Vec::new(), String::new()),
            None => (
                default_renames
                    .iter()
                    .map(|(a, b)| ((*a).into(), (*b).into()))
                    .collect(),
                default_prefix.into(),
            ),
        };
        Ok(Some(Self {
            dir: dir.into(),
            groups,
            vocab_prefix,
            vocab,
        }))
    }

    /// Stem of the vocabulary shard file.
    pub(crate) fn vocab_file(&self) -> String {
        format!("{}exlr8-vocab", self.dir)
    }

    /// File holding `name`, and its stored name inside it.
    pub(crate) fn place(&self, name: &str, in_vocab: impl Fn(&str) -> bool) -> (String, String) {
        for &(group, prefix) in self.groups {
            if let Some(rest) = name.strip_prefix(prefix)
                && let Some((layer, local)) = rest.split_once('.')
                && let Ok(layer) = layer.parse::<usize>()
            {
                let file = format!("{}exlr8-dense-{group}-{layer:03}", self.dir);
                return (file, local.into());
            }
        }
        for (stem, hf) in &self.vocab {
            if let Some(end) = past(name, hf) {
                return (self.vocab_file(), format!("{stem}{end}"));
            }
        }
        if let Some(local) = name.strip_prefix(&self.vocab_prefix)
            && in_vocab(local)
        {
            return (self.vocab_file(), local.into());
        }
        (format!("{}exlr8-dense-global-000", self.dir), name.into())
    }

    /// Every shard in this layout, probed by `exists`.
    pub(crate) fn shards(&self, exists: impl Fn(&str) -> bool) -> Vec<Shard> {
        let mut out = vec![Shard {
            renames: self.vocab.clone(),
            ..Shard::new(self.vocab_file(), self.vocab_prefix.clone())
        }];
        for idx in 0.. {
            let file = format!("{}exlr8-dense-global-{idx:03}", self.dir);
            if !exists(&file) {
                break;
            }
            out.push(Shard::new(file, String::new()));
        }
        for &(group, prefix) in self.groups {
            for layer in 0.. {
                let file = format!("{}exlr8-dense-{group}-{layer:03}", self.dir);
                if !exists(&file) {
                    break;
                }
                out.push(Shard::new(file, format!("{prefix}{layer}.")));
            }
        }
        out
    }
}
