//! exlr8 v2 releases: trellis, block and fp8 matrices.

use crate::file::{Error, File};
use crate::offer::{Offer, Slice, Stored};
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

/// Trellis matrix `name` packed: a header, then its entries.
fn packed(f: &File, name: &str, r: &Record) -> Result<Vec<u8>, Error> {
    let entries = f.bytes(&format!("{name}.trellis"))?;
    Ok(trellis::packed(r.inp, r.out, r.side as u32, &entries))
}

/// Trellis matrix `name` as f32 `[out][in]`, before its rotations.
fn decoded(f: &File, name: &str, r: &Record) -> Result<Vec<f32>, Error> {
    let form = trellis::form_for(r.k, Codebook::Mcg)
        .ok_or_else(|| Error::DType(format!("{name}: K{}", r.k)))?;
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
    let entries = f.bytes(&format!("{name}.trellis"))?;
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
        other => Err(Error::DType(format!("{name} is in exlr8 format {other}"))),
    }
}

/// The block form a format's planes join into.
fn block_form(format: i64) -> Option<&'static dyn Form> {
    match format {
        INT4 => Some(&block::INT4_G32),
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
    /// GLM's layout: `dense/` files, layer names without their prefix.
    glm: bool,
    /// The trellis width GLM's routed experts load at.
    width: u32,
}

/// GLM's vocabulary file, which marks its layout.
const GLM_VOCAB: &str = "dense/exlr8-vocab";

/// One file, and how its names become HF names.
struct Shard {
    file: String,
    prefix: String,
    /// Names kept as they are.
    bare: &'static [&'static str],
}

impl Shard {
    fn new(file: String, prefix: String) -> Self {
        Self {
            file,
            prefix,
            bare: &[],
        }
    }

    fn name(&self, local: &str) -> String {
        match self.bare.iter().any(|b| local.starts_with(b)) {
            true => local.into(),
            false => format!("{}{local}", self.prefix),
        }
    }
}

enum Where {
    Dir(PathBuf),
    Url(String),
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
            glm: false,
            width: 3,
        };
        release.glm = release.with(GLM_VOCAB, |_| Ok(())).is_ok();
        release
    }

    /// GLM's routed experts at `k` bits: 2, 3 or 4.
    pub fn width(mut self, k: u32) -> Self {
        self.width = k;
        self
    }

    /// Packed expert codes. `set` keeps a range of routed experts.
    fn expert_codes(
        &self,
        layer: usize,
        local: &str,
        set: Option<(usize, usize)>,
    ) -> Result<Vec<u8>, Error> {
        let k = self.width;
        let sizes = self.with(&experts::index_file(layer), |f| {
            experts::sizes(f, k).ok_or_else(|| Error::Missing(local.into()))
        })?;
        let shared = local.starts_with("shared_experts.");
        let width = if shared { experts::shared_width(k) } else { k };
        let [hidden, _, ffn] = sizes;
        let file = experts::codes_file(layer, width);
        let set = set.filter(|_| !shared);
        let read = |f: &File| experts::packed(f, local, [hidden, ffn], width, set);
        let mut bytes = self.with(&file, read)?;
        // The shared expert is a rotated file: its rotations follow.
        if let Some(stem) = local.strip_suffix(".weight").filter(|_| shared) {
            let index = experts::index_file(layer);
            for end in ["suh", "svh"] {
                let name = format!("{stem}.{end}");
                let v = self.with(&index, |f| experts::floats(f, &name, k))?;
                bytes.extend(v.iter().flat_map(|x| F16::from(*x).0.to_le_bytes()));
            }
        }
        Ok(bytes)
    }

    /// The file holding `name`, and the name inside it.
    fn place(&self, name: &str) -> (String, String) {
        if self.glm {
            return glm_place(name);
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

    /// The packed codes of `M.codes`, or `rows` as `(first, count)`.
    pub fn packed(&self, name: &str, rows: Option<(usize, usize)>) -> Result<Vec<u8>, Error> {
        if let Some((layer, local)) = experts::split(name).filter(|_| self.glm) {
            return self.expert_codes(layer, local, rows);
        }
        let (file, local) = self.place(name);
        let matrix = local
            .strip_suffix(".codes")
            .or(local.strip_suffix(".weight"));
        let matrix = matrix.unwrap_or(&local).to_string();
        let whole = local.ends_with(".weight");
        self.with(&file, |f| match record(f, &matrix)? {
            r if block_form(r.format).is_some() => {
                blocks(f, &matrix, &r, rows.unwrap_or((0, r.out)))
            }
            r if r.format == TRELLIS && whole => turned(f, &matrix),
            _ => codes_packed(f, &matrix, rows),
        })
    }

    /// Tensor `name` as f32. `M.codes`, `M.suh`, `M.svh`: unrotated parts.
    pub fn read(&self, name: &str) -> Result<Vec<f32>, Error> {
        if let Some((layer, local)) = experts::split(name).filter(|_| self.glm) {
            let k = self.width;
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

/// GLM: layer files under `dense/`, and one vocabulary file.
fn glm_place(name: &str) -> (String, String) {
    if let Some(rest) = name.strip_prefix("model.layers.") {
        let (layer, local) = rest.split_once('.').expect("a layer index");
        let layer: usize = layer.parse().expect("a layer index");
        return (format!("dense/exlr8-dense-{layer:03}"), local.into());
    }
    let local = name.strip_prefix("model.").unwrap_or(name);
    (GLM_VOCAB.into(), local.into())
}

impl Release {
    /// Every shard, and how its names become HF names.
    fn shards(&self) -> Vec<Shard> {
        let (mut out, layers, prefix) = match self.glm {
            true => {
                let vocab = Shard {
                    bare: &["lm_head"],
                    ..Shard::new(GLM_VOCAB.into(), "model.".into())
                };
                (vec![vocab], "dense/exlr8-dense-", "model.layers.")
            }
            false => (
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
            out.push(Shard::new(file, format!("{prefix}{layer}.")));
        }
        out
    }

    /// A shard's slices. GLM's take their most compact forms.
    fn offered(f: &File, shard: &Shard, glm: bool) -> Vec<Slice> {
        let raw = f.slices();
        let shape = |name: &str| raw.iter().find(|s| s.name == name).map(|s| s.shape.clone());
        let mut out = Vec::new();
        for s in &raw {
            let Some(matrix) = s.name.strip_suffix(".record") else {
                continue;
            };
            let name = |part: &str| shard.name(&format!("{matrix}.{part}"));
            let (Some(suh), Some(svh)) = (
                shape(&format!("{matrix}.suh")),
                shape(&format!("{matrix}.svh")),
            ) else {
                // Other formats decode to the HF weight.
                if let Ok(r) = record(f, matrix) {
                    let stored = match (glm, r.format) {
                        (false, _) | (true, FP32) => Stored::F32,
                        (true, format) => block_form(format).map_or(Stored::Bf16, Stored::Form),
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
            let Some(entries) = shape(&format!("{matrix}.trellis")) else {
                continue;
            };
            let k = (entries[0] * entries[1] * 8 / (inp * outp)) as u32;
            let Some(form) = trellis::form_for(k, Codebook::Mcg) else {
                continue;
            };
            // GLM keeps the matrix packed, its rotations beside the codes.
            let whole = match (glm, trellis::turned_for(k, Codebook::Mcg)) {
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
        let part = |n: &str| {
            [
                ".record",
                ".checksum",
                ".trellis",
                ".suh",
                ".svh",
                ".fp8",
                ".scale",
                ".q4",
                ".s4",
                ".e8m0",
                ".bf16",
                ".fp32",
            ]
            .iter()
            .any(|p| n.ends_with(p))
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
            let slices = self.with(&shard.file, |f| Ok(Self::offered(f, &shard, self.glm)));
            out.extend(slices.unwrap_or_default());
        }
        if !self.glm {
            return out;
        }
        // GLM's routed layers keep their experts apart.
        for layer in 0.. {
            if self
                .with(&format!("dense/exlr8-dense-{layer:03}"), |_| Ok(()))
                .is_err()
            {
                break;
            }
            let k = self.width;
            let book = |w| self.with(&experts::codes_file(layer, w), experts::codebook);
            let (Ok(routed), Ok(shared)) = (book(k), book(experts::shared_width(k))) else {
                continue;
            };
            let index = experts::index_file(layer);
            let found = self.with(&index, |f| {
                Ok(experts::offered(f, layer, k, [routed, shared]))
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

    fn packed_rows(&self, name: &str, rows: (usize, usize)) -> Result<Vec<u8>, Error> {
        Release::packed(self, name, Some(rows))
    }

    /// Routed experts are cut. Other matrices are read whole.
    fn packed_share(&self, name: &str, rows: (usize, usize)) -> Result<Vec<u8>, Error> {
        let routed = experts::split(name).is_some_and(|(_, l)| l.starts_with("experts."));
        Release::packed(self, name, (routed && self.glm).then_some(rows))
    }
}

/// GLM's routed and shared experts at one trellis width.
mod experts {
    use super::{Codebook, Error, File, Slice, Stored, trellis};

    /// Channels one rotation block scale covers.
    const BLOCK: usize = 128;

    /// A layer's index file: side data for every width.
    pub(crate) fn index_file(layer: usize) -> String {
        format!("experts/exlr8-layer-{layer:03}.index")
    }

    /// A layer's trellis codes at width `k`.
    pub(crate) fn codes_file(layer: usize, k: u32) -> String {
        format!("experts/exlr8-layer-{layer:03}.k{k}")
    }

    /// The shared expert has no K2 codes.
    pub(crate) fn shared_width(k: u32) -> u32 {
        k.max(3)
    }

    /// A layer and its expert name, under the layer's MLP.
    pub(crate) fn split(name: &str) -> Option<(usize, &str)> {
        let rest = name.strip_prefix("model.layers.")?;
        let (layer, local) = rest.split_once('.')?;
        let local = local.strip_prefix("mlp.")?;
        let ours = local.starts_with("experts.") || local.starts_with("shared_experts.");
        ours.then_some((layer.parse().ok()?, local))
    }

    /// Hidden width, expert count and expert width, from the index.
    pub(crate) fn sizes(idx: &File, k: u32) -> Option<[usize; 3]> {
        let hidden = idx.shape("signs.in")?[0];
        let &[experts, ffn] = idx.shape(&format!("k{k}.gate.svh"))? else {
            return None;
        };
        Some([hidden, experts, ffn])
    }

    /// A codes file's codebook, from its tensor names.
    pub(crate) fn codebook(codes: &File) -> Result<Codebook, Error> {
        for book in [Codebook::Mcg, Codebook::Mul1] {
            if codes
                .shape(&format!("trellis.gate{}", book.suffix()))
                .is_some()
            {
                return Ok(book);
            }
        }
        Err(Error::Missing("trellis.gate in a known codebook".into()))
    }

    /// The layer's slices: codes as trellis sets, rotations as floats.
    pub(crate) fn offered(
        idx: &File,
        layer: usize,
        k: u32,
        [routed, shared]: [Codebook; 2],
    ) -> Vec<Slice> {
        let (Some([hidden, experts, ffn]), Some(routed), Some(shared)) = (
            sizes(idx, k),
            trellis::form_for(k, routed),
            trellis::turned_for(shared_width(k), shared),
        ) else {
            return Vec::new();
        };
        let at = format!("model.layers.{layer}.mlp.");
        let slice = |name: String, shape: Vec<usize>, stored| Slice {
            name: format!("{at}{name}"),
            shape,
            stored,
        };
        let mut out = Vec::new();
        for (p, shape) in [
            ("gate", [ffn, hidden]),
            ("up", [ffn, hidden]),
            ("down", [hidden, ffn]),
        ] {
            let [rows, cols] = shape;
            let set = vec![experts, rows, cols];
            out.push(slice(
                format!("experts.{p}.weight"),
                set,
                Stored::Form(routed),
            ));
            let (into, from) = (format!("experts.{p}.su"), format!("experts.{p}.sv"));
            out.push(slice(into, vec![experts, cols], Stored::F32));
            out.push(slice(from, vec![experts, rows], Stored::F32));
            let stem = format!("shared_experts.{p}_proj");
            out.push(slice(
                format!("{stem}.weight"),
                vec![rows, cols],
                Stored::Form(shared),
            ));
        }
        for side in ["signs_in", "signs_out"] {
            out.push(slice(format!("experts.{side}"), vec![hidden], Stored::F32));
        }
        out
    }

    /// Each value repeated over its block of channels.
    fn repeated(values: &[f32]) -> Vec<f32> {
        values.iter().flat_map(|v| [*v; BLOCK]).collect()
    }

    /// The layer's signs, as floats.
    fn signs(idx: &File, which: &str) -> Result<Vec<f32>, Error> {
        let bytes = idx.bytes(&format!("signs.{which}"))?;
        Ok(bytes.iter().map(|b| f32::from(*b as i8)).collect())
    }

    /// A rotation table or vector, from the index file.
    pub(crate) fn floats(idx: &File, local: &str, k: u32) -> Result<Vec<f32>, Error> {
        let half = |name: String| idx.floats(&name);
        let signed = |which: &str, v: Vec<f32>| -> Result<Vec<f32>, Error> {
            let s = signs(idx, which)?;
            Ok(v.iter().zip(s).map(|(a, b)| a * b).collect())
        };
        let ks = shared_width(k);
        match local {
            "experts.signs_in" => signs(idx, "in"),
            "experts.signs_out" => signs(idx, "out"),
            "experts.gate.su" | "experts.up.su" => {
                let p = &local["experts.".len()..local.len() - 3];
                Ok(repeated(&half(format!("k{k}.{p}.su_block"))?))
            }
            "experts.gate.sv" | "experts.up.sv" => {
                let p = &local["experts.".len()..local.len() - 3];
                half(format!("k{k}.{p}.svh"))
            }
            "experts.down.su" => half(format!("k{k}.down.suh")),
            "experts.down.sv" => Ok(repeated(&half(format!("k{k}.down.sv_block"))?)),
            "shared_experts.gate_proj.suh" | "shared_experts.up_proj.suh" => {
                let p = &local["shared_experts.".len()..local.len() - "_proj.suh".len()];
                let block = repeated(&half(format!("shared.k{ks}.{p}.su_block"))?);
                signed("in", block)
            }
            "shared_experts.gate_proj.svh" | "shared_experts.up_proj.svh" => {
                let p = &local["shared_experts.".len()..local.len() - "_proj.svh".len()];
                half(format!("shared.k{ks}.{p}.svh"))
            }
            "shared_experts.down_proj.suh" => half(format!("shared.k{ks}.down.suh")),
            "shared_experts.down_proj.svh" => {
                let block = repeated(&half(format!("shared.k{ks}.down.sv_block"))?);
                signed("out", block)
            }
            _ => Err(Error::Missing(local.into())),
        }
    }

    /// A set or shared matrix as `k`-bit trellis codes.
    pub(crate) fn packed(
        codes: &File,
        local: &str,
        [hidden, ffn]: [usize; 2],
        k: u32,
        set: Option<(usize, usize)>,
    ) -> Result<Vec<u8>, Error> {
        let missing = || Error::Missing(local.into());
        let (tensor, p) = match local.strip_prefix("shared_experts.") {
            Some(rest) => (true, rest.strip_suffix("_proj.weight").ok_or_else(missing)?),
            None => {
                let rest = local.strip_prefix("experts.").ok_or_else(missing)?;
                (false, rest.strip_suffix(".weight").ok_or_else(missing)?)
            }
        };
        let suffix = codebook(codes)?.suffix();
        let name = match tensor {
            true => format!("shared.trellis.{p}{suffix}"),
            false => format!("trellis.{p}{suffix}"),
        };
        let entries = match set {
            None => codes.bytes(&name)?,
            Some((first, count)) => {
                let matrix = (hidden * ffn * k as usize / 8) as u64;
                codes.bytes_at(&name, first as u64 * matrix, Some(count as u64 * matrix))?
            }
        };
        Ok(match p {
            "down" => trellis::packed(ffn, hidden, trellis::INPUT_SIDE, &entries),
            _ => trellis::packed(hidden, ffn, 1, &entries),
        })
    }
}
