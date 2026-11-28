//! exlr8 v2 to v4 releases: trellis, block and fp8 matrices.

mod experts;
mod format;
mod shard;

use crate::file::{Error, File};
use crate::offer::{Offer, Slice, Stored, narrow};
pub use format::codes;
use format::{
    BF16, FP8_BLOCK, FP8_ROW, FP32, ROWS, TRELLIS, block_form, blocks, bounds, coded, codes_packed,
    halves, matrix, plain, record, rows, span, turned,
};
use shard::{Shard, layer_prefix, past};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tessera_engine::trellis::Codebook;
pub use tessera_engine::trellis::rotated;
use tessera_engine::{F16, trellis};

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

/// Every tensor outside the routed experts, by HF name.
const WHOLE: &str = "model-nonexperts";

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
    /// Whether the extras hold plain tensors only.
    plain: bool,
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

enum Where {
    Dir(PathBuf),
    Url(String),
}

/// Whether `dir` holds an exlr8 release directory.
pub fn is_release(dir: &Path) -> bool {
    let file = |stem: &str| dir.join(format!("{stem}.safetensors")).is_file();
    file(VOCAB) || file("exlr8-vocab") || (file(WHOLE) && file(&experts::index_file(0)))
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
        Self::from_url(url)
    }

    /// Streams an exlr8 release from an HTTP base URL.
    pub fn from_url(base: impl Into<String>) -> Self {
        let mut url = base.into();
        if !url.ends_with('/') {
            url.push('/');
        }
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

    /// The names of a release with one whole-tensor file.
    fn read_whole(&self) -> Result<Dense, Error> {
        let index = experts::index_file(0);
        let named = self.with(&index, |f| Ok(f.meta("tensor_prefix").map(String::from)))?;
        let found = self.with(WHOLE, |f| Ok(layer_prefix(f)))?;
        Ok(Dense {
            prefix: named.or(found).unwrap_or_else(|| "model.layers.".into()),
            vocab_prefix: String::new(),
            vocab: Vec::new(),
            single: Some(WHOLE.into()),
            extras: vec![WHOLE.into()],
            plain: true,
            widths: HashMap::new(),
            experts: experts::Widths::default(),
        })
    }

    /// The `dense/` layout's names, if the release has that layout.
    fn read_dense(&self) -> Result<Dense, Error> {
        if self.with(VOCAB, |_| Ok(())).is_err()
            && let Ok(whole) = self.read_whole()
        {
            return Ok(whole);
        }
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
            plain: false,
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
        let rel = format!("manifests/exlr8-manifest-{name}.json");
        let (source, text) = match &self.at {
            Where::Dir(dir) => {
                let path = dir.join(&rel);
                (path.display().to_string(), std::fs::read_to_string(&path)?)
            }
            Where::Url(base) => {
                let url = format!("{base}{rel}");
                let resp = ureq::get(&url)
                    .call()
                    .map_err(|e| Error::Net(format!("{url}: {e}")))?;
                let text = resp
                    .into_string()
                    .map_err(|e| Error::Net(format!("{url}: {e}")))?;
                (url, text)
            }
        };
        let bad = |why: String| Error::DType(format!("{source}: {why}"));
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

    /// Each routed expert's width, where a projection mixes them.
    fn expert_mix(&self, layer: usize, proj: experts::Proj) -> Option<&[u32]> {
        self.dense.as_ref()?.experts.mixed(layer, proj)
    }

    /// Trellis forms for one routed layer's projections.
    fn expert_forms(&self, layer: usize) -> Option<experts::Forms> {
        let mut routed = Vec::with_capacity(3);
        let mut shared = Vec::with_capacity(3);
        for p in experts::Proj::ALL {
            let kr = self.expert_width(layer, false, p);
            let ks = self.expert_width(layer, true, p);
            let book = |k: u32| {
                let file = experts::codes_file(layer, k);
                let book = self.with(&file, |f| experts::codebook(f, false, p)).ok()?;
                trellis::form_for(k, book).map(|_| book)
            };
            let rbook = book(kr)?;
            routed.push(match self.expert_mix(layer, p) {
                // Every width's file must use one codebook.
                Some(each) => {
                    let all = experts::distinct(each);
                    if all.iter().any(|k| book(*k) != Some(rbook)) {
                        return None;
                    }
                    trellis::mixed_for(rbook)?
                }
                None => trellis::form_for(kr, rbook)?,
            });
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
        if let Some(each) = self.expert_mix(layer, proj).filter(|_| !shared) {
            let mut set = experts::Mixed::new(local, sizes, each, slice)?;
            for k in experts::distinct(each) {
                self.with(&experts::codes_file(layer, k), |f| set.fill(f, k))?;
            }
            return Ok(set.bytes);
        }
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

    /// Whether release file `file` opens.
    fn has(&self, file: &str) -> bool {
        self.with(file, |_| Ok(())).is_ok()
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
                let keep = bounds(all.len(), range, r.inp);
                all.truncate(keep.end);
                all.drain(..keep.start);
                Ok(all.into_flattened())
            }
            Ok(r) if r.format == FP8_BLOCK || r.format == FP8_ROW => {
                let floats = plain(f, &matrix, &r, r.format)?;
                Ok(span(&floats, range, r.inp)
                    .iter()
                    .flat_map(narrow)
                    .collect())
            }
            Ok(r) if r.format == TRELLIS && whole => turned(f, &matrix, range),
            Ok(_) => codes_packed(f, &matrix, range),
            Err(e) => {
                if f.is_half(&local) {
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
            let target = experts::target(local);
            let k = target
                .map(|(shared, proj)| self.expert_width(layer, shared, proj))
                .unwrap_or(self.width);
            let mix = target.and_then(|(shared, proj)| match shared {
                true => None,
                false => self.expert_mix(layer, proj),
            });
            return self.with(&experts::index_file(layer), |f| match mix {
                Some(each) => experts::mixed_floats(f, local, each),
                None => experts::floats(f, local, k),
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
                    out.push(Shard {
                        plain: d.plain,
                        ..Shard::new(extra.clone(), String::new())
                    });
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
            if !self.has(&file) {
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

    /// A shard's slices, keeping each matrix in its compact form.
    fn offered(f: &File, shard: &Shard) -> Vec<Slice> {
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
                    let stored = match r.format {
                        FP32 => Stored::F32,
                        format => block_form(format).map_or(Stored::BF16, Stored::Form),
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
            // Keep the matrix packed, its rotations beside the codes.
            let whole = trellis::turned_for(k, book).map_or(Stored::F32, Stored::Form);
            let parts = [
                ("weight", vec![outp, inp], whole),
                ("codes", vec![outp, inp], Stored::Form(form)),
                ("suh", vec![inp], Stored::F16),
                ("svh", vec![outp], Stored::F16),
            ];
            for (part, shape, stored) in parts {
                out.push(Slice {
                    name: name(part),
                    shape,
                    stored,
                });
            }
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
            let kept = shard.plain && !s.name.starts_with("__pad");
            if kept || !part(&s.name) {
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
            let slices = self.with(&shard.file, |f| Ok(Self::offered(f, &shard)));
            out.extend(slices.unwrap_or_default());
        }
        let Some(dense) = &self.dense else {
            return out;
        };
        // Routed layers keep their experts apart.
        let mut layers = dense.experts.layers();
        for s in &out {
            if let Some(n) = s
                .name
                .strip_prefix(&dense.prefix)
                .and_then(|r| r.split('.').next())
                .and_then(|d| d.parse::<usize>().ok())
            {
                layers.push(n);
            }
        }
        for layer in 0.. {
            let has_idx = self.has(&experts::index_file(layer));
            let has_dense =
                dense.single.is_none() && self.has(&format!("dense/exlr8-dense-{layer:03}"));
            if !has_idx && !has_dense {
                break;
            }
            if has_idx {
                layers.push(layer);
            }
        }
        layers.sort_unstable();
        layers.dedup();
        for layer in layers {
            let index = experts::index_file(layer);
            if !self.has(&index) {
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
