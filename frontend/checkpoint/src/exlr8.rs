//! exlr8 v2 to v4 releases: trellis, block and fp8 matrices.

use crate::exlr8_experts as experts;
use crate::file::{Error, File};
use crate::offer::{Offer, Slice, Stored, narrow};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tessera_engine::storage::Form;
use tessera_engine::trellis::Codebook;
pub use tessera_engine::trellis::rotated;
use tessera_engine::{E4M3, F16, block, trellis};

/// Record field 0: how the matrix is stored.
const TRELLIS: i64 = 1;
const INT4: i64 = 2;
const MXFP8: i64 = 3;
const FP8_BLOCK: i64 = 4;
const FP8_ROW: i64 = 5;
const BF16: i64 = 6;
const FP32: i64 = 7;
const ROWS: i64 = 9;
const INT2: i64 = 10;

/// A matrix's record: format, shape, trellis width, sharded side.
struct Record {
    format: i64,
    out: usize,
    inp: usize,
    k: u32,
    side: i64,
}

fn record(f: &File, name: &str) -> Result<Record, Error> {
    let bytes = f.bytes(&format!("{name}.record"))?;
    let field = |i: usize| i64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().expect("8 bytes"));
    Ok(Record {
        format: field(0),
        out: field(1) as usize,
        inp: field(2) as usize,
        k: field(3) as u32,
        side: field(4),
    })
}

fn halves(f: &File, name: &str) -> Result<Vec<f32>, Error> {
    Ok(F16::floats(&f.bytes(name)?))
}

fn e4m3(bits: u8) -> f32 {
    f32::from(E4M3(bits))
}

/// Tensor `stem` in a known codebook, and that codebook.
pub(crate) fn coded(f: &File, stem: &str) -> Result<(String, Codebook), Error> {
    for book in Codebook::ALL {
        let tensor = format!("{stem}{}", book.suffix());
        if f.shape(&tensor).is_some() {
            return Ok((tensor, book));
        }
    }
    Err(Error::Missing(format!("{stem} in a known codebook")))
}

/// Row trellis `name`, each value through `each`.
fn rows<T: Clone + Send>(
    f: &File,
    name: &str,
    r: &Record,
    each: impl Fn(&f32) -> T + Sync,
) -> Result<Vec<T>, Error> {
    let (tensor, book) = coded(f, &format!("{name}.rows"))?;
    let side = |s: &str| halves(f, &format!("{name}.{s}"));
    let (suh, svh) = (side("suh")?, side("svh")?);
    let bad = || Error::DType(format!("{name}: unreadable K{} {book:?} rows", r.k));
    trellis::rows(r.k, book, &f.bytes(&tensor)?, [&suh, &svh], each).ok_or_else(bad)
}

/// Trellis matrix `name` packed: a header, then its entries.
fn packed(f: &File, name: &str, r: &Record) -> Result<Vec<u8>, Error> {
    let entries = f.bytes(&coded(f, &format!("{name}.trellis"))?.0)?;
    Ok(trellis::packed(r.inp, r.out, r.side as u32, &entries))
}

/// Trellis matrix `name` as f32 `[out][in]`, before its rotations.
fn decoded(f: &File, name: &str, r: &Record) -> Result<Vec<f32>, Error> {
    let book = coded(f, &format!("{name}.trellis"))?.1;
    let form = trellis::form_for(r.k, book)
        .ok_or_else(|| Error::DType(format!("{name}: K{} {book:?}", r.k)))?;
    let bytes = packed(f, name, r)?;
    Ok(form.unpack_all(&bytes))
}

/// Matrix `name` as f32 `[out][in]`, decoded by its record.
fn matrix(f: &File, name: &str) -> Result<Vec<f32>, Error> {
    let r = record(f, name)?;
    match r.format {
        TRELLIS => {
            let suh = halves(f, &format!("{name}.suh"))?;
            let svh = halves(f, &format!("{name}.svh"))?;
            let t = decoded(f, name, &r)?;
            Ok(trellis::rotated(&t, r.inp, r.out, &suh, &svh))
        }
        other => plain(f, name, &r, other),
    }
}

/// Packed codes of `name`, or `rows` as `(first, count)`.
fn codes_packed(f: &File, name: &str, rows: Option<(usize, usize)>) -> Result<Vec<u8>, Error> {
    let r = record(f, name)?;
    if r.format != TRELLIS {
        let what = format!("{name} is in exlr8 format {}, not trellis", r.format);
        return Err(Error::DType(what));
    }
    let Some((first, count)) = rows else {
        return packed(f, name, &r);
    };
    // Output-side entries are 128 rows: a range is whole entries.
    let entry = r.inp * 128 * r.k as usize / 8;
    if r.side == trellis::INPUT_SIDE as i64 || first % 128 != 0 || count % 128 != 0 {
        return Err(Error::DType(format!(
            "{name}: rows {first}.. are not whole entries"
        )));
    }
    let entries = f.bytes(&coded(f, &format!("{name}.trellis"))?.0)?;
    let at = first / 128 * entry;
    let slice = &entries[at..at + count / 128 * entry];
    Ok(trellis::packed(r.inp, count, r.side as u32, slice))
}

/// Matrix `name` as f32 `[out][in]` before its rotations, if any.
pub fn codes(f: &File, name: &str) -> Result<Vec<f32>, Error> {
    let r = record(f, name)?;
    match r.format {
        TRELLIS => decoded(f, name, &r),
        other => plain(f, name, &r, other),
    }
}

/// A matrix in a format other than trellis, as `[out][in]`.
fn plain(f: &File, name: &str, r: &Record, format: i64) -> Result<Vec<f32>, Error> {
    let inp = r.inp;
    let fp8 = || f.bytes(&format!("{name}.fp8"));
    if let Some(form) = block_form(format) {
        return Ok(form.unpack_all(&blocks(f, name, r, (0, r.out))?));
    }
    match format {
        FP8_ROW => {
            let scale = f.floats(&format!("{name}.scale"))?;
            let row = |(i, c): (usize, &u8)| e4m3(*c) * scale[i / inp];
            Ok(fp8()?.iter().enumerate().map(row).collect())
        }
        // One scale per 128 by 128 block.
        FP8_BLOCK => {
            let scale = f.floats(&format!("{name}.scale"))?;
            let at = |i: usize| scale[i / inp / 128 * inp.div_ceil(128) + i % inp / 128];
            let value = |(i, c): (usize, &u8)| e4m3(*c) * at(i);
            Ok(fp8()?.iter().enumerate().map(value).collect())
        }
        BF16 => f.floats(&format!("{name}.bf16")),
        FP32 => f.floats(&format!("{name}.fp32")),
        ROWS => rows(f, name, r, |x| *x),
        other => Err(Error::DType(format!("{name} is in exlr8 format {other}"))),
    }
}

/// The block form a format's planes join into.
fn block_form(format: i64) -> Option<&'static dyn Form> {
    match format {
        INT4 => Some(&block::INT4_G32),
        INT2 => Some(&block::INT2_G32),
        MXFP8 => Some(&block::MXFP8),
        _ => None,
    }
}

/// Rows `first..first + count` of a block form matrix.
fn blocks(
    f: &File,
    name: &str,
    r: &Record,
    (first, count): (usize, usize),
) -> Result<Vec<u8>, Error> {
    // One scale and 32 codes per 32 inputs.
    let (scale_part, code_part, scale_bytes, code_bytes) = match r.format {
        INT4 => ("s4", "q4", 2, 16),
        INT2 => ("s2", "q2", 2, 8),
        _ => ("e8m0", "fp8", 1, 32),
    };
    let scales = f.bytes(&format!("{name}.{scale_part}"))?;
    let codes = f.bytes(&format!("{name}.{code_part}"))?;
    if !r.inp.is_multiple_of(32) || first + count > r.out {
        return Err(Error::DType(format!("{name}: {count} rows of {}", r.inp)));
    }
    // Input-side int4 shards hold 128 inputs of every output.
    let grouped = r.format == INT4 && r.side == trellis::INPUT_SIDE as i64;
    let groups = r.inp / 32;
    let mut out = Vec::with_capacity(count * groups * (scale_bytes + code_bytes));
    for o in first..first + count {
        for g in 0..groups {
            let at = match grouped {
                true => (g / 4 * r.out + o) * 4 + g % 4,
                false => o * groups + g,
            };
            out.extend_from_slice(&scales[at * scale_bytes..(at + 1) * scale_bytes]);
            out.extend_from_slice(&codes[at * code_bytes..(at + 1) * code_bytes]);
        }
    }
    Ok(out)
}

/// A trellis matrix as a rotated file, rotations after codes.
fn turned(f: &File, name: &str) -> Result<Vec<u8>, Error> {
    let mut bytes = codes_packed(f, name, None)?;
    bytes.extend(f.bytes(&format!("{name}.suh"))?);
    bytes.extend(f.bytes(&format!("{name}.svh"))?);
    Ok(bytes)
}

/// An exlr8 release, read by HF tensor name.
pub struct Release {
    /// A directory, or a URL prefix ending in `/`.
    at: Where,
    files: RefCell<HashMap<String, File>>,
    /// GLM's and later layouts: layer files under `dense/`.
    dense: Option<Dense>,
    /// Fallback trellis width for routed experts without a manifest.
    width: u32,
}

/// The `dense/` layout's vocabulary file. Its presence marks the layout.
const VOCAB: &str = "dense/exlr8-vocab";

/// Names in the `dense/` layout.
struct Dense {
    /// Before a layer's index, from the files' `tensor_prefix`.
    prefix: String,
    /// Prefix added to unrenamed tensors in `VOCAB`.
    vocab_prefix: String,
    /// Vocabulary file stems and their HF names.
    vocab: Vec<(String, String)>,
    /// Single dense file when layers are not split per file.
    single: Option<String>,
    /// Files that hold whole HF names.
    extras: Vec<String>,
    /// The manifest's width for each `(layer, matrix)` stored at several.
    widths: HashMap<(usize, String), u32>,
    /// The manifest's expert widths, by layer and projection.
    experts: experts::Widths,
}

impl Dense {
    /// Layer `layer`'s stored name for `local`, at its picked width.
    fn stored(&self, layer: usize, local: &str) -> String {
        let Some((matrix, part)) = local.rsplit_once('.') else {
            return local.into();
        };
        match self.widths.get(&(layer, matrix.to_string())) {
            Some(k) => format!("{matrix}.k{k}.{part}"),
            None => local.into(),
        }
    }
}

/// `name` past `stem`, where `stem` is a whole first part.
fn past<'a>(name: &'a str, stem: &str) -> Option<&'a str> {
    let end = name.strip_prefix(stem)?;
    (end.is_empty() || end.starts_with('.')).then_some(end)
}

/// The text before a layer index in `f`'s names.
fn layer_prefix(f: &File) -> Option<String> {
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
struct Shard {
    file: String,
    prefix: String,
    /// Stems that take other HF names.
    renames: Vec<(String, String)>,
    /// Widths of matrices stored at several, by stem.
    picked: Vec<(String, u32)>,
}

impl Shard {
    fn new(file: String, prefix: String) -> Self {
        Self {
            file,
            prefix,
            renames: Vec::new(),
            picked: Vec::new(),
        }
    }

    fn name(&self, local: &str) -> String {
        for (stem, hf) in &self.renames {
            if let Some(end) = past(local, stem) {
                return format!("{hf}{end}");
            }
        }
        format!("{}{local}", self.prefix)
    }

    /// A stem's matrix, or `None` for an unpicked width.
    fn matrix<'a>(&self, stem: &'a str) -> Option<&'a str> {
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

enum Where {
    Dir(PathBuf),
    Url(String),
}

/// Whether `dir` holds an exlr8 release directory.
pub fn is_release(dir: &Path) -> bool {
    dir.join("dense/exlr8-vocab.safetensors").is_file()
        || dir.join("exlr8-vocab.safetensors").is_file()
}

/// Opens an exlr8 release, picking `name` or its sole manifest.
pub fn open_manifest(dir: &Path, name: Option<&str>) -> Result<Release, Error> {
    let release = Release::open(dir);
    let all = release.manifests();
    let picked = match (name, all.as_slice()) {
        (Some(name), _) => name.to_string(),
        (None, []) => return Ok(release),
        (None, [one]) => one.clone(),
        (None, many) => {
            return Err(Error::Format(format!(
                "the release has manifests {}: pick one",
                many.join(", ")
            )));
        }
    };
    release.manifest(&picked)
}

impl Release {
    pub fn open(dir: &Path) -> Self {
        Self::new(Where::Dir(dir.to_path_buf()))
    }

    /// Streams `owner/name` on Hugging Face at `revision`.
    pub fn remote(repo: &str, revision: &str) -> Self {
        let url = tessera_frontend_settings::resolve(repo, revision, "");
        Self::new(Where::Url(url))
    }

    fn new(at: Where) -> Self {
        let mut release = Self {
            at,
            files: RefCell::new(HashMap::new()),
            dense: None,
            width: 3,
        };
        release.dense = release.read_dense().ok();
        release
    }

    /// The `dense/` layout's names, if the release has that layout.
    fn read_dense(&self) -> Result<Dense, Error> {
        let glm = [
            ("embed_tokens", "model.embed_tokens"),
            ("norm", "model.norm"),
            ("lm_head", "lm_head"),
        ];
        let (hf, full_names) = self.with(VOCAB, |f| {
            let hf = f.meta("hf_names").map(String::from);
            let full = f
                .slices()
                .iter()
                .any(|s| s.name.starts_with("model.") || s.name == "lm_head.weight");
            Ok((hf, full))
        })?;
        let vocab = match hf {
            Some(json) => serde_json::from_str::<HashMap<String, String>>(&json)
                .map_err(|e| Error::DType(format!("{VOCAB} hf_names: {e}")))?
                .into_iter()
                .collect(),
            None if full_names => Vec::new(),
            None => glm
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
        };
        let single_file = "dense/exlr8-dense";
        let found = self.with(single_file, |f| Ok(layer_prefix(f))).ok();
        let single = found.is_some().then(|| single_file.to_string());
        let first = "dense/exlr8-dense-000";
        let prefix = found
            .flatten()
            .or_else(|| self.with(first, |f| Ok(layer_prefix(f))).ok().flatten())
            .unwrap_or_else(|| "model.layers.".into());
        let mut extras = Vec::new();
        if let Some(s) = &single {
            extras.push(s.clone());
        }
        if let Where::Dir(dir) = &self.at
            && let Ok(listed) = std::fs::read_dir(dir.join("dense"))
        {
            for entry in listed.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if let Some(stem) = name.strip_suffix(".safetensors")
                    && stem.starts_with("exlr8-extra-")
                {
                    extras.push(format!("dense/{stem}"));
                }
            }
        }
        extras.sort();
        Ok(Dense {
            prefix,
            vocab_prefix: if full_names {
                String::new()
            } else {
                "model.".into()
            },
            vocab,
            single,
            extras,
            widths: HashMap::new(),
            experts: experts::Widths::default(),
        })
    }

    /// The manifests a local release holds, by name.
    pub fn manifests(&self) -> Vec<String> {
        let Where::Dir(dir) = &self.at else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for entry in std::fs::read_dir(dir.join("manifests"))
            .into_iter()
            .flatten()
            .flatten()
        {
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(n) = name
                .strip_prefix("exlr8-manifest-")
                .and_then(|n| n.strip_suffix(".json"))
            {
                out.push(n.to_string());
            }
        }
        out.sort();
        out
    }

    /// Matrices and experts at the widths manifest `name` picks.
    pub fn manifest(mut self, name: &str) -> Result<Self, Error> {
        let Where::Dir(dir) = &self.at else {
            return Err(Error::Missing(format!(
                "manifest {name} of a remote release"
            )));
        };
        let path = dir.join(format!("manifests/exlr8-manifest-{name}.json"));
        let bad = |why: String| Error::DType(format!("{}: {why}", path.display()));
        let text = std::fs::read_to_string(&path)?;
        let m: serde_json::Value = serde_json::from_str(&text).map_err(|e| bad(e.to_string()))?;
        let needs = m["requires"]["trellis"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        for need in needs {
            let need = need.as_str().unwrap_or_default();
            let known = need.split_once(' ').is_some_and(|(width, book)| {
                let k = width.strip_prefix('k').and_then(|k| k.parse().ok());
                let found = k.zip(Codebook::named(book));
                found.is_some_and(|(k, book)| trellis::form_for(k, book).is_some())
            });
            if !known {
                return Err(bad(format!("needs trellis {need}: this engine lacks it")));
            }
        }
        let dense = self
            .dense
            .as_mut()
            .ok_or_else(|| bad("no dense/ files".into()))?;
        dense.widths.clear();
        dense.experts.clear();
        let layers = m["dense"].as_object().cloned().unwrap_or_default();
        for (layer, matrices) in layers {
            let layer: usize = layer.parse().map_err(|_| bad(format!("layer {layer}")))?;
            for (matrix, k) in matrices.as_object().cloned().unwrap_or_default() {
                let k = k
                    .as_u64()
                    .ok_or_else(|| bad(format!("{layer}.{matrix} width")))?;
                dense.widths.insert((layer, matrix), k as u32);
            }
        }
        dense.experts.parse(&m["layers"]).map_err(bad)?;
        Ok(self)
    }

    /// Fallback trellis width for routed experts without a manifest.
    pub fn width(mut self, k: u32) -> Self {
        self.width = k;
        self
    }

    /// Trellis width for one layer's expert projection.
    fn expert_width(&self, layer: usize, shared: bool, proj: experts::Proj) -> u32 {
        match &self.dense {
            Some(d) => d.experts.get(layer, shared, proj, self.width),
            None if shared => experts::shared_width(self.width),
            None => self.width,
        }
    }

    /// Trellis forms for one routed layer's projections.
    fn expert_forms(&self, layer: usize) -> Option<experts::Forms> {
        let mut routed = Vec::with_capacity(3);
        let mut shared = Vec::with_capacity(3);
        for p in experts::Proj::ALL {
            let kr = self.expert_width(layer, false, p);
            let ks = self.expert_width(layer, true, p);
            let rbook = self
                .with(&experts::codes_file(layer, kr), |f| {
                    experts::codebook(f, false, p)
                })
                .ok()?;
            routed.push(trellis::form_for(kr, rbook)?);
            let sform = self
                .with(&experts::codes_file(layer, ks), |f| {
                    experts::codebook(f, true, p)
                })
                .ok()
                .and_then(|book| trellis::turned_for(ks, book));
            if let Some(f) = sform {
                shared.push(f);
            }
        }
        Some(experts::Forms {
            routed: routed.try_into().ok()?,
            shared: shared.try_into().ok(),
        })
    }

    /// Packed expert codes, optionally sliced along an axis.
    fn expert_codes(
        &self,
        layer: usize,
        local: &str,
        slice: Option<(u32, usize, usize)>,
    ) -> Result<Vec<u8>, Error> {
        let (shared, proj) = experts::target(local).ok_or_else(|| Error::Missing(local.into()))?;
        let width = self.expert_width(layer, shared, proj);
        let gate_k = self.expert_width(layer, false, experts::Proj::Gate);
        let sizes = self.with(&experts::index_file(layer), |f| {
            experts::sizes(f, gate_k).ok_or_else(|| Error::Missing(local.into()))
        })?;
        let file = experts::codes_file(layer, width);
        let slice = slice.filter(|_| !shared);
        let read = |f: &File| experts::packed(f, local, sizes, width, slice);
        let mut bytes = self.with(&file, read)?;
        // The shared expert is a rotated file: its rotations follow.
        if let Some(stem) = local.strip_suffix(".weight").filter(|_| shared) {
            let index = experts::index_file(layer);
            for end in ["suh", "svh"] {
                let name = format!("{stem}.{end}");
                let v = self.with(&index, |f| experts::floats(f, &name, width))?;
                bytes.extend(v.iter().flat_map(|x| F16::from(*x).0.to_le_bytes()));
            }
        }
        Ok(bytes)
    }

    /// The file holding `name`, and the name inside it.
    fn place(&self, name: &str) -> (String, String) {
        if let Some(d) = &self.dense {
            return self.dense_place(d, name);
        }
        if let Some(rest) = name.strip_prefix("language_model.layers.") {
            let (layer, local) = rest.split_once('.').expect("a layer index");
            let layer: usize = layer.parse().expect("a layer index");
            return (format!("exlr8-dense-text-{layer:03}"), local.into());
        }
        match name {
            "language_model.embed_tokens.weight" => {
                ("exlr8-vocab".into(), "embed_tokens.weight".into())
            }
            _ => ("exlr8-dense-global-000".into(), name.into()),
        }
    }

    fn with<T>(
        &self,
        file: &str,
        read: impl FnOnce(&File) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let mut files = self.files.borrow_mut();
        if !files.contains_key(file) {
            let name = format!("{file}.safetensors");
            let opened = match &self.at {
                Where::Dir(dir) => File::open(&dir.join(name))?,
                Where::Url(base) => File::remote(&format!("{base}{name}"))?,
            };
            files.insert(file.into(), opened);
        }
        read(&files[file])
    }

    /// Packed bytes of `name`, or `range` rows `(first, count)`.
    pub fn packed(&self, name: &str, range: Option<(usize, usize)>) -> Result<Vec<u8>, Error> {
        if let Some((layer, local)) = self.expert(name) {
            return self.expert_codes(layer, local, range.map(|(f, c)| (0, f, c)));
        }
        let (file, local) = self.place(name);
        let matrix = local
            .strip_suffix(".codes")
            .or(local.strip_suffix(".weight"));
        let matrix = matrix.unwrap_or(&local).to_string();
        let whole = local.ends_with(".weight");
        self.with(&file, |f| match record(f, &matrix) {
            Ok(r) if block_form(r.format).is_some() => {
                blocks(f, &matrix, &r, range.unwrap_or((0, r.out)))
            }
            Ok(r) if r.format == BF16 => {
                let tensor = format!("{matrix}.bf16");
                match range {
                    Some((first, count)) => {
                        let stride = 2 * r.inp as u64;
                        f.bytes_at(&tensor, first as u64 * stride, Some(count as u64 * stride))
                    }
                    None => f.bytes(&tensor),
                }
            }
            Ok(r) if r.format == ROWS => {
                let mut all = rows(f, &matrix, &r, narrow)?;
                if let Some((first, count)) = range {
                    all = all
                        .into_iter()
                        .skip(first * r.inp)
                        .take(count * r.inp)
                        .collect();
                }
                Ok(all.into_flattened())
            }
            Ok(r) if r.format == FP8_BLOCK || r.format == FP8_ROW => {
                let floats = plain(f, &matrix, &r, r.format)?;
                let slice = match range {
                    Some((first, count)) => &floats[first * r.inp..(first + count) * r.inp],
                    None => &floats,
                };
                Ok(slice.iter().flat_map(narrow).collect())
            }
            Ok(r) if r.format == TRELLIS && whole => turned(f, &matrix),
            Ok(_) => codes_packed(f, &matrix, range),
            Err(e) => {
                if f.is_bf16(&local) {
                    return match range {
                        Some(r) => f.packed_rows(&local, r),
                        None => f.bytes(&local),
                    };
                }
                Err(e)
            }
        })
    }

    /// Tensor `name` as f32. `M.codes`, `M.suh`, `M.svh`: unrotated parts.
    pub fn read(&self, name: &str) -> Result<Vec<f32>, Error> {
        if let Some((layer, local)) = self.expert(name) {
            let k = experts::target(local)
                .map(|(shared, proj)| self.expert_width(layer, shared, proj))
                .unwrap_or(self.width);
            return self.with(&experts::index_file(layer), |f| {
                experts::floats(f, local, k)
            });
        }
        let (file, local) = self.place(name);
        self.with(&file, |f| {
            let (matrix_name, part) = match local.rsplit_once('.') {
                Some((m, part)) if ["weight", "codes", "suh", "svh"].contains(&part) => (m, part),
                _ => (local.as_str(), "weight"),
            };
            if f.shape(&format!("{matrix_name}.record")).is_none() {
                return f.floats(&local);
            }
            match part {
                "codes" => codes(f, matrix_name),
                "suh" | "svh" => halves(f, &format!("{matrix_name}.{part}")),
                _ => matrix(f, matrix_name),
            }
        })
    }
}

impl Release {
    /// A routed layer and expert name in the `dense/` layout.
    fn expert<'a>(&self, name: &'a str) -> Option<(usize, &'a str)> {
        experts::split(name, &self.dense.as_ref()?.prefix)
    }

    /// The `dense/` file holding `name`, and its stored name.
    fn dense_place(&self, d: &Dense, name: &str) -> (String, String) {
        if d.single.is_none()
            && let Some(rest) = name.strip_prefix(&d.prefix)
            && let Some((layer, local)) = rest.split_once('.')
            && let Ok(layer) = layer.parse::<usize>()
        {
            return (
                format!("dense/exlr8-dense-{layer:03}"),
                d.stored(layer, local),
            );
        }
        for (stem, hf) in &d.vocab {
            if let Some(end) = past(name, hf) {
                return (VOCAB.into(), format!("{stem}{end}"));
            }
        }
        let matrix = name
            .strip_suffix(".weight")
            .or_else(|| name.strip_suffix(".codes"))
            .unwrap_or(name);
        for extra in &d.extras {
            let holds = |f: &File| {
                Ok(f.shape(name).is_some() || f.shape(&format!("{matrix}.record")).is_some())
            };
            if self.with(extra, holds).unwrap_or(false) {
                return (extra.clone(), name.into());
            }
        }
        (VOCAB.into(), name.into())
    }

    /// Every shard, and how its names become HF names.
    fn shards(&self) -> Vec<Shard> {
        let (mut out, layers, prefix) = match &self.dense {
            Some(d) => {
                let vocab = Shard {
                    renames: d.vocab.clone(),
                    ..Shard::new(VOCAB.into(), d.vocab_prefix.clone())
                };
                let mut out = vec![vocab];
                for extra in &d.extras {
                    out.push(Shard::new(extra.clone(), String::new()));
                }
                (out, "dense/exlr8-dense-", d.prefix.as_str())
            }
            None => (
                vec![
                    Shard::new("exlr8-vocab".into(), "language_model.".into()),
                    Shard::new("exlr8-dense-global-000".into(), String::new()),
                ],
                "exlr8-dense-text-",
                "language_model.layers.",
            ),
        };
        for layer in 0.. {
            let file = format!("{layers}{layer:03}");
            if self.with(&file, |_| Ok(())).is_err() {
                break;
            }
            let mut shard = Shard::new(file, format!("{prefix}{layer}."));
            for ((at, matrix), k) in self.dense.iter().flat_map(|d| &d.widths) {
                if *at == layer {
                    shard.picked.push((matrix.clone(), *k));
                }
            }
            out.push(shard);
        }
        out
    }

    /// A shard's slices. GLM's take their most compact forms.
    fn offered(f: &File, shard: &Shard, dense: bool) -> Vec<Slice> {
        let raw = f.slices();
        let shape = |name: &str| raw.iter().find(|s| s.name == name).map(|s| s.shape.clone());
        let mut out = Vec::new();
        for s in &raw {
            let Some(stem) = s.name.strip_suffix(".record") else {
                continue;
            };
            let Some(matrix) = shard.matrix(stem) else {
                continue;
            };
            let name = |part: &str| shard.name(&format!("{matrix}.{part}"));
            let r = record(f, stem);
            let rows = r.as_ref().is_ok_and(|r| r.format == ROWS);
            let side = |s: &str| shape(&format!("{stem}.{s}")).filter(|_| !rows);
            let (Some(suh), Some(svh)) = (side("suh"), side("svh")) else {
                // Other formats decode to the HF weight.
                if let Ok(r) = r {
                    let stored = match (dense, r.format) {
                        (false, _) | (true, FP32) => Stored::F32,
                        (true, format) => block_form(format).map_or(Stored::BF16, Stored::Form),
                    };
                    out.push(Slice {
                        name: name("weight"),
                        shape: vec![r.out, r.inp],
                        stored,
                    });
                }
                continue;
            };
            let (inp, outp) = (suh[0], svh[0]);
            let Ok((tensor, book)) = coded(f, &format!("{stem}.trellis")) else {
                continue;
            };
            let Some(entries) = shape(&tensor) else {
                continue;
            };
            let k = (entries[0] * entries[1] * 8 / (inp * outp)) as u32;
            let Some(form) = trellis::form_for(k, book) else {
                continue;
            };
            // GLM keeps the matrix packed, its rotations beside the codes.
            let whole = match (dense, trellis::turned_for(k, book)) {
                (true, Some(turned)) => Stored::Form(turned),
                _ => Stored::F32,
            };
            out.push(Slice {
                name: name("weight"),
                shape: vec![outp, inp],
                stored: whole,
            });
            out.push(Slice {
                name: name("codes"),
                shape: vec![outp, inp],
                stored: Stored::Form(form),
            });
            out.push(Slice {
                name: name("suh"),
                shape: vec![inp],
                stored: Stored::F16,
            });
            out.push(Slice {
                name: name("svh"),
                shape: vec![outp],
                stored: Stored::F16,
            });
        }
        // Each matrix's stored parts, then codebook-named codes.
        let parts = [
            "record", "checksum", "trellis", "rows", "suh", "svh", "fp8", "scale", "q4", "s4",
            "q2", "s2", "e8m0", "bf16", "fp32",
        ];
        let part = |n: &str| {
            n.rsplit_once('.')
                .is_some_and(|(_, end)| parts.contains(&end))
                || n.contains(".trellis.")
                || n.contains(".rows.")
                || n.starts_with("__pad")
        };
        for s in raw {
            if !part(&s.name) {
                out.push(Slice {
                    name: shard.name(&s.name),
                    ..s
                });
            }
        }
        out
    }
}

impl Offer for Release {
    fn slices(&self) -> Vec<Slice> {
        let mut out = Vec::new();
        for shard in self.shards() {
            let dense = self.dense.is_some();
            let slices = self.with(&shard.file, |f| Ok(Self::offered(f, &shard, dense)));
            out.extend(slices.unwrap_or_default());
        }
        let Some(dense) = &self.dense else {
            return out;
        };
        // Routed layers keep their experts apart.
        for layer in 0.. {
            let index = experts::index_file(layer);
            let has_index = self.with(&index, |_| Ok(())).is_ok();
            let has_dense = dense.single.is_none()
                && self
                    .with(&format!("dense/exlr8-dense-{layer:03}"), |_| Ok(()))
                    .is_ok();
            if !has_index && !has_dense {
                break;
            }
            if !has_index {
                continue;
            }
            let Some(forms) = self.expert_forms(layer) else {
                continue;
            };
            let gate_k = self.expert_width(layer, false, experts::Proj::Gate);
            let found = self.with(&index, |f| {
                Ok(experts::offered(f, &dense.prefix, layer, gate_k, forms))
            });
            out.extend(found.unwrap_or_default());
        }
        out
    }

    fn floats(&self, name: &str) -> Result<Vec<f32>, Error> {
        self.read(name)
    }

    fn packed(&self, name: &str) -> Result<Vec<u8>, Error> {
        Release::packed(self, name, None)
    }

    fn i64s(&self, name: &str) -> Result<Vec<i64>, Error> {
        let (file, local) = self.place(name);
        self.with(&file, |f| f.i64s(&local))
    }

    fn packed_rows(&self, name: &str, rows: (usize, usize)) -> Result<Vec<u8>, Error> {
        Release::packed(self, name, Some(rows))
    }

    /// Routed experts are cut. Other matrices are read whole.
    fn packed_share(&self, name: &str, rows: (usize, usize)) -> Result<Vec<u8>, Error> {
        let routed = self
            .expert(name)
            .is_some_and(|(_, l)| l.starts_with("experts."));
        Release::packed(self, name, routed.then_some(rows))
    }

    /// Routed experts cut along expert or inner axis.
    fn packed_axis(&self, name: &str, axis: u32, range: (usize, usize)) -> Result<Vec<u8>, Error> {
        let Some((layer, local)) = self.expert(name) else {
            return Release::packed(self, name, None);
        };
        let slice = local
            .starts_with("experts.")
            .then_some((axis, range.0, range.1));
        self.expert_codes(layer, local, slice)
    }
}
