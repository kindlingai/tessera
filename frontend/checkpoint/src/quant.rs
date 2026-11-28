//! Composable `quantization_config` tensor-pairing adapter over `Shards`.

use crate::file::{Error, Shards};
use crate::offer::{Offer, Slice, Stored, narrow};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, LazyLock};
use tessera_engine::storage::Form;
use tessera_engine::{E4M3, Filed, block};

/// Minimum elements before FP8 decoding fans out across threads.
const PARALLEL_ELEMS: usize = 65_536;

/// Precomputed f32 table for every OCP E4M3 byte value.
static E4M3_LUT: LazyLock<[f32; 256]> =
    LazyLock::new(|| std::array::from_fn(|b| f32::from(E4M3(b as u8))));

/// Decodes one FP8 row into f32 using block scales.
fn decode_fp8_row(lut: &[f32; 256], codes: &[u8], scales: &[f32], bk: usize, dst: &mut [f32]) {
    for ((out_blk, code_blk), &scale) in dst.chunks_mut(bk).zip(codes.chunks(bk)).zip(scales) {
        for (o, &c) in out_blk.iter_mut().zip(code_blk) {
            *o = lut[usize::from(c)] * scale;
        }
    }
}

/// Encodes one FP8 row to BF16 using block scales.
fn encode_fp8_bf16_row(lut: &[f32; 256], codes: &[u8], scales: &[f32], bk: usize, dst: &mut [u8]) {
    for ((out_blk, code_blk), &scale) in dst.chunks_mut(bk * 2).zip(codes.chunks(bk)).zip(scales) {
        for (pair, &c) in out_blk.as_chunks_mut::<2>().0.iter_mut().zip(code_blk) {
            *pair = narrow(&(lut[usize::from(c)] * scale));
        }
    }
}

/// Loaded FP8 codes and scales for rows.
struct Fp8Slice {
    first: usize,
    count: usize,
    inp: usize,
    block: [usize; 2],
    codes: Vec<u8>,
    scales: Vec<f32>,
}

/// Runs `work` over contiguous row strips, in parallel when large.
fn for_row_strips<T: Send>(
    dst: &mut [T],
    row_len: usize,
    rows: usize,
    work: impl Fn(usize, &mut [T]) + Sync,
) {
    let threads = if rows * row_len >= PARALLEL_ELEMS {
        std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(rows)
    } else {
        1
    };
    if threads <= 1 {
        work(0, dst);
        return;
    }
    let per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        for (ti, chunk) in dst.chunks_mut(per * row_len).enumerate() {
            let start = ti * per;
            let f = &work;
            scope.spawn(move || f(start, chunk));
        }
    });
}

/// Quantization layout parsed from `config.json`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Scheme {
    /// FP8 E4M3 weights paired with float scales.
    Fp8 { block: Option<[usize; 2]> },
    /// OCP MXFP8: E4M3 codes with per-32 E8M0 scales.
    MxFp8,
    /// OCP MXFP4: packed E2M1 nibbles with per-32 E8M0 scales.
    MxFp4,
    /// NVFP4: E2M1 nibbles with per-16 E4M3 and global scales.
    NvFp4,
    /// Mixed per-tensor quantization such as `compressed-tensors` or `modelopt`.
    Mixed,
}

impl Scheme {
    /// Parses `quantization_config` from a Hugging Face `config.json` string.
    pub fn parse(config_json: &str) -> Option<Self> {
        let root: serde_json::Value = serde_json::from_str(config_json).ok()?;
        let qcfg = root.get("quantization_config").or_else(|| {
            root.get("text_config")
                .and_then(|t| t.get("quantization_config"))
        })?;
        let obj = qcfg.as_object()?;
        let method = obj.get("quant_method")?.as_str()?.to_ascii_lowercase();
        match method.as_str() {
            "fp8" => Some(Self::Fp8 {
                block: parse_block(obj.get("weight_block_size")),
            }),
            "mxfp8" => Some(Self::MxFp8),
            "mxfp4" => Some(Self::MxFp4),
            "nvfp4" => Some(Self::NvFp4),
            "modelopt" => match obj.get("quant_algo").and_then(|v| v.as_str()) {
                Some(algo) if algo.eq_ignore_ascii_case("nvfp4") => Some(Self::NvFp4),
                Some(algo) if algo.eq_ignore_ascii_case("mxfp4") => Some(Self::MxFp4),
                Some(algo) if algo.eq_ignore_ascii_case("mxfp8") => Some(Self::MxFp8),
                Some(algo) if algo.to_ascii_lowercase().contains("fp8") => Some(Self::Fp8 {
                    block: parse_block(obj.get("weight_block_size")),
                }),
                _ => Some(Self::Mixed),
            },
            "compressed-tensors" => Some(Self::Mixed),
            _ => None,
        }
    }

    fn allows_fp8(&self) -> bool {
        matches!(self, Self::Fp8 { .. } | Self::Mixed)
    }

    fn allows_mxfp8(&self) -> bool {
        matches!(self, Self::MxFp8 | Self::Mixed)
    }

    fn allows_mxfp4(&self) -> bool {
        matches!(self, Self::MxFp4 | Self::Mixed)
    }

    fn allows_nvfp4(&self) -> bool {
        matches!(self, Self::NvFp4 | Self::Mixed)
    }

    fn fp8_block(&self) -> Option<[usize; 2]> {
        match *self {
            Self::Fp8 { block } => block,
            _ => None,
        }
    }
}

fn parse_block(val: Option<&serde_json::Value>) -> Option<[usize; 2]> {
    let arr = val?.as_array()?;
    if arr.len() != 2 {
        return None;
    }
    let bm = arr[0].as_u64()? as usize;
    let bk = arr[1].as_u64()? as usize;
    (bm > 0 && bk > 0).then_some([bm, bk])
}

/// How one logical quantized weight is assembled from shard tensors.
enum Recipe {
    /// Interleaved block form with optional f32 tensor head.
    Block {
        form: &'static dyn Form,
        shape: Vec<usize>,
        codes: String,
        scales: String,
        /// Optional `(tensor_name, invert)` for a scalar f32 head.
        head: Option<(String, bool)>,
        scale_bytes: usize,
        code_bytes: usize,
    },
    /// E4M3 codes scaled by float block/row/tensor scales.
    Fp8 {
        shape: [usize; 2],
        codes: String,
        scales: String,
        block: [usize; 2],
    },
}

impl Recipe {
    fn shape(&self) -> &[usize] {
        match self {
            Recipe::Block { shape, .. } => shape,
            Recipe::Fp8 { shape, .. } => shape,
        }
    }

    fn stored(&self) -> Stored {
        match self {
            Recipe::Block { form, .. } => Stored::Form(*form),
            Recipe::Fp8 { .. } => Stored::BF16,
        }
    }
}

/// Adapts paired quantized safetensors slices into Tessera storage forms.
pub(crate) struct Quant {
    shards: Shards,
    slices: Vec<Slice>,
    recipes: HashMap<String, Recipe>,
}

/// Wraps `shards` in `Quant` when `dir/config.json` configures quantization.
pub(crate) fn wrap(dir: &Path, shards: Shards) -> Result<Box<dyn Offer>, Error> {
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return Ok(Box::new(shards));
    };
    let Some(scheme) = Scheme::parse(&text) else {
        return Ok(Box::new(shards));
    };
    let quant = Quant::new(shards, scheme);
    if quant.recipes.is_empty() {
        return Ok(Box::new(quant.shards));
    }
    Ok(Box::new(quant))
}

/// Metadata for raw shard tensors indexed by name.
struct RawIndex<'a> {
    shards: &'a Shards,
    names: HashSet<String>,
}

impl<'a> RawIndex<'a> {
    fn new(shards: &'a Shards, raw: &[Slice]) -> Self {
        let names = raw.iter().map(|s| s.name.clone()).collect();
        Self { shards, names }
    }

    fn has(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    fn shape(&self, name: &str) -> Option<&[usize]> {
        self.shards.shape(name)
    }

    fn dtype(&self, name: &str) -> Option<&str> {
        self.shards.dtype(name)
    }
}

fn is_float_dtype(dtype: &str) -> bool {
    matches!(dtype, "F32" | "BF16" | "F16")
}

fn is_e8m0(dtype: &str) -> bool {
    matches!(dtype, "U8" | "F8_E8M0")
}

fn is_fp8_u8(dtype: &str) -> bool {
    matches!(dtype, "F8_E4M3" | "U8")
}

/// Logical shape when `codes` packs `groups * code_bytes` behind `scales`.
fn block_shape(
    codes_shape: &[usize],
    scales_shape: &[usize],
    block: usize,
    code_bytes: usize,
) -> Option<Vec<usize>> {
    if scales_shape.len() < 2 || codes_shape.len() < scales_shape.len() {
        return None;
    }
    let lead = &scales_shape[..scales_shape.len() - 1];
    let groups = *scales_shape.last()?;
    if groups == 0 || !codes_shape.starts_with(lead) {
        return None;
    }
    let tail_bytes: usize = codes_shape[lead.len()..].iter().product();
    if tail_bytes != groups * code_bytes {
        return None;
    }
    let mut out = lead.to_vec();
    out.push(groups * block);
    Some(out)
}

/// Infers `[row_block, col_block]` for a 2D FP8 weight and scale.
fn fp8_block(
    [out, inp]: [usize; 2],
    scale_shape: &[usize],
    hint: Option<[usize; 2]>,
) -> Option<[usize; 2]> {
    if out == 0 || inp == 0 {
        return None;
    }
    match scale_shape {
        [] | [1] | [1, 1] => Some([out, inp]),
        [r] | [r, 1] if *r == out => Some([1, inp]),
        [sm, sk] if *sm > 0 && *sk > 0 => {
            if let Some([bm, bk]) = hint
                && out.div_ceil(bm) == *sm
                && inp.div_ceil(bk) == *sk
            {
                return Some([bm, bk]);
            }
            let bm = out.div_ceil(*sm);
            let bk = inp.div_ceil(*sk);
            (out.div_ceil(bm) == *sm && inp.div_ceil(bk) == *sk).then_some([bm, bk])
        }
        _ => None,
    }
}

/// Optional NVFP4 global scale tensor and whether it is inverted.
fn find_head(idx: &RawIndex<'_>, stem: &str) -> Option<(String, bool)> {
    let candidates = [
        (format!("{stem}.weight_scale_2"), false),
        (format!("{stem}.weight_global_scale"), true),
    ];
    for (name, invert) in candidates {
        if idx.has(&name)
            && idx.dtype(&name).is_some_and(is_float_dtype)
            && idx
                .shape(&name)
                .is_some_and(|s| s.iter().product::<usize>() == 1)
        {
            return Some((name, invert));
        }
    }
    None
}

fn block_pair(
    logical: String,
    form: &'static dyn Form,
    shape: Vec<usize>,
    codes: &str,
    scales: String,
    head: Option<(String, bool)>,
    code_bytes: usize,
) -> (String, Recipe, Vec<String>) {
    let mut used = vec![codes.to_string(), scales.clone()];
    if let Some((h, _)) = &head {
        used.push(h.clone());
    }
    let recipe = Recipe::Block {
        form,
        shape,
        codes: codes.to_string(),
        scales,
        head,
        scale_bytes: 1,
        code_bytes,
    };
    (logical, recipe, used)
}

/// Matches a raw tensor as the codes of a pair.
fn match_pair(
    idx: &RawIndex<'_>,
    raw_name: &str,
    scheme: &Scheme,
) -> Option<(String, Recipe, Vec<String>)> {
    // OpenAI gpt-oss MXFP4 naming: `<stem>_blocks` + `<stem>_scales`.
    if let Some(stem) = raw_name.strip_suffix("_blocks") {
        if !scheme.allows_mxfp4() || idx.dtype(raw_name) != Some("U8") {
            return None;
        }
        let scales = format!("{stem}_scales");
        if !idx.dtype(&scales).is_some_and(is_e8m0) {
            return None;
        }
        let shape = block_shape(idx.shape(raw_name)?, idx.shape(&scales)?, 32, 16)?;
        return Some(block_pair(
            format!("{stem}.weight"),
            &block::MXFP4,
            shape,
            raw_name,
            scales,
            None,
            16,
        ));
    }

    let stem = raw_name
        .strip_suffix(".weight_packed")
        .or_else(|| raw_name.strip_suffix(".weight"))?;
    let codes_dtype = idx.dtype(raw_name)?;
    let codes_shape = idx.shape(raw_name)?;
    let logical = format!("{stem}.weight");

    // FP8 with `weight_scale_inv` or float `weight_scale`.
    if scheme.allows_fp8() && codes_dtype == "F8_E4M3" && codes_shape.len() == 2 {
        let inv = format!("{stem}.weight_scale_inv");
        let plain = format!("{stem}.weight_scale");
        let scale_name = if idx.dtype(&inv).is_some_and(is_float_dtype) {
            Some(inv)
        } else if idx.dtype(&plain).is_some_and(is_float_dtype) {
            Some(plain)
        } else {
            None
        };
        if let Some(scales) = scale_name {
            let shape = [codes_shape[0], codes_shape[1]];
            let block = fp8_block(shape, idx.shape(&scales)?, scheme.fp8_block())?;
            let used = vec![raw_name.to_string(), scales.clone()];
            let recipe = Recipe::Fp8 {
                shape,
                codes: raw_name.to_string(),
                scales,
                block,
            };
            return Some((logical, recipe, used));
        }
    }

    // Block-form scales (`weight_scale` or `scales`).
    let scales = [format!("{stem}.weight_scale"), format!("{stem}.scales")]
        .into_iter()
        .find(|n| idx.has(n))?;
    let scale_dtype = idx.dtype(&scales)?;
    let scale_shape = idx.shape(&scales)?;

    // MXFP8: 1-byte E8M0 scale + 32-byte E4M3 codes.
    if scheme.allows_mxfp8()
        && is_fp8_u8(codes_dtype)
        && is_e8m0(scale_dtype)
        && let Some(shape) = block_shape(codes_shape, scale_shape, 32, 32)
    {
        return Some(block_pair(
            logical,
            &block::MXFP8,
            shape,
            raw_name,
            scales,
            None,
            32,
        ));
    }

    // NVFP4: f32 head, E4M3 scale, and 8-byte E2M1 codes.
    if scheme.allows_nvfp4()
        && codes_dtype == "U8"
        && is_fp8_u8(scale_dtype)
        && let Some(shape) = block_shape(codes_shape, scale_shape, 16, 8)
    {
        let head = find_head(idx, stem);
        // Mixed mode needs E4M3 scales or a global head.
        let confident = *scheme == Scheme::NvFp4 || scale_dtype == "F8_E4M3" || head.is_some();
        if confident {
            return Some(block_pair(
                logical,
                &block::NVFP4,
                shape,
                raw_name,
                scales,
                head,
                8,
            ));
        }
    }

    // MXFP4: 1-byte E8M0 scale + 16-byte E2M1 codes.
    if scheme.allows_mxfp4()
        && codes_dtype == "U8"
        && is_e8m0(scale_dtype)
        && let Some(shape) = block_shape(codes_shape, scale_shape, 32, 16)
    {
        return Some(block_pair(
            logical,
            &block::MXFP4,
            shape,
            raw_name,
            scales,
            None,
            16,
        ));
    }

    None
}

impl Quant {
    /// Builds a tensor-pairing adapter over `shards` for `scheme`.
    pub fn new(shards: Shards, scheme: Scheme) -> Self {
        let raw = shards.slices();
        let idx = RawIndex::new(&shards, &raw);
        let mut consumed = HashSet::new();
        let mut recipes = HashMap::new();
        let mut virtual_slices = Vec::new();
        for s in &raw {
            if consumed.contains(&s.name) {
                continue;
            }
            let Some((logical, recipe, used)) = match_pair(&idx, &s.name, &scheme) else {
                continue;
            };
            if recipes.contains_key(&logical) {
                continue;
            }
            for u in used {
                consumed.insert(u);
            }
            virtual_slices.push(Slice {
                name: logical.clone(),
                shape: recipe.shape().to_vec(),
                stored: recipe.stored(),
            });
            recipes.insert(logical, recipe);
        }
        let mut slices: Vec<Slice> = raw
            .into_iter()
            .filter(|s| !consumed.contains(&s.name))
            .collect();
        slices.extend(virtual_slices);
        Self {
            shards,
            slices,
            recipes,
        }
    }

    fn read_head(&self, head: Option<&(String, bool)>) -> Result<f32, Error> {
        let Some((name, invert)) = head else {
            return Ok(1.0);
        };
        let vals = self.shards.floats(name)?;
        let raw = *vals
            .first()
            .ok_or_else(|| Error::Format(format!("{name}: empty scale")))?;
        if *invert {
            if raw == 0.0 {
                return Err(Error::Format(format!("{name}: zero global scale")));
            }
            Ok(1.0 / raw)
        } else {
            Ok(raw)
        }
    }

    fn assemble_block(
        &self,
        recipe: &Recipe,
        rows: Option<(usize, usize)>,
    ) -> Result<Vec<u8>, Error> {
        let Recipe::Block {
            form,
            shape,
            codes,
            scales,
            head,
            scale_bytes,
            code_bytes,
        } = recipe
        else {
            unreachable!("checked block recipe");
        };
        let dim0 = *shape.first().unwrap_or(&0);
        let (first, count) = rows.unwrap_or((0, dim0));
        if first + count > dim0 {
            return Err(Error::Format(format!(
                "{codes}: rows {first}..{} past {dim0}",
                first + count
            )));
        }
        let head_bytes = form.bytes(0);
        let tail_elems: usize = shape[1..].iter().product();
        let row_groups = tail_elems / form.block();
        let total_groups = count * row_groups;
        let stride = scale_bytes + code_bytes;
        let mut out = vec![0u8; head_bytes + total_groups * stride];
        if head_bytes == 4 {
            let val = self.read_head(head.as_ref())?;
            out[..4].copy_from_slice(&val.to_le_bytes());
        }
        if count == 0 || row_groups == 0 {
            return Ok(out);
        }
        let scale_row = row_groups * scale_bytes;
        let code_row = row_groups * code_bytes;
        let scale_raw = self.shards.bytes_at(
            scales,
            (first * scale_row) as u64,
            Some((count * scale_row) as u64),
        )?;
        let code_raw = self.shards.bytes_at(
            codes,
            (first * code_row) as u64,
            Some((count * code_row) as u64),
        )?;
        for (g, dst) in out[head_bytes..].chunks_exact_mut(stride).enumerate() {
            let sb = g * scale_bytes;
            let cb = g * code_bytes;
            dst[..*scale_bytes].copy_from_slice(&scale_raw[sb..sb + scale_bytes]);
            dst[*scale_bytes..].copy_from_slice(&code_raw[cb..cb + code_bytes]);
        }
        Ok(out)
    }

    fn load_fp8_parts(
        &self,
        recipe: &Recipe,
        rows: Option<(usize, usize)>,
    ) -> Result<Fp8Slice, Error> {
        let Recipe::Fp8 {
            shape: [out_rows, inp],
            codes,
            scales,
            block,
        } = recipe
        else {
            unreachable!("checked fp8 recipe");
        };
        let (first, count) = rows.unwrap_or((0, *out_rows));
        if first + count > *out_rows {
            return Err(Error::Format(format!(
                "{codes}: rows {first}..{} past {out_rows}",
                first + count
            )));
        }
        if count == 0 || *inp == 0 {
            return Ok(Fp8Slice {
                first,
                count: 0,
                inp: *inp,
                block: *block,
                codes: Vec::new(),
                scales: Vec::new(),
            });
        }
        let code_raw =
            self.shards
                .bytes_at(codes, (first * inp) as u64, Some((count * inp) as u64))?;
        let scale_vals = self.shards.floats(scales)?;
        Ok(Fp8Slice {
            first,
            count,
            inp: *inp,
            block: *block,
            codes: code_raw,
            scales: scale_vals,
        })
    }

    fn map_fp8_rows<T: Copy + Default + Send>(
        &self,
        recipe: &Recipe,
        rows: Option<(usize, usize)>,
        elem_stride: usize,
        row_fn: impl Fn(&[f32; 256], &[u8], &[f32], usize, &mut [T]) + Sync,
    ) -> Result<Vec<T>, Error> {
        let slice = self.load_fp8_parts(recipe, rows)?;
        if slice.count == 0 || slice.inp == 0 {
            return Ok(Vec::new());
        }
        let [bm, bk] = slice.block;
        let lut = &*E4M3_LUT;
        let scale_cols = slice.inp.div_ceil(bk);
        let row_len = slice.inp * elem_stride;
        let mut out = vec![T::default(); slice.count * row_len];
        for_row_strips(&mut out, row_len, slice.count, |start, chunk| {
            for (dr, dst_row) in chunk.chunks_exact_mut(row_len).enumerate() {
                let r = start + dr;
                let sr = (slice.first + r) / bm * scale_cols;
                let codes = &slice.codes[r * slice.inp..(r + 1) * slice.inp];
                row_fn(lut, codes, &slice.scales[sr..sr + scale_cols], bk, dst_row);
            }
        });
        Ok(out)
    }

    fn decode_fp8(&self, recipe: &Recipe, rows: Option<(usize, usize)>) -> Result<Vec<f32>, Error> {
        self.map_fp8_rows(recipe, rows, 1, decode_fp8_row)
    }

    fn assemble(&self, recipe: &Recipe, rows: Option<(usize, usize)>) -> Result<Vec<u8>, Error> {
        match recipe {
            Recipe::Block { .. } => self.assemble_block(recipe, rows),
            Recipe::Fp8 { .. } => self.map_fp8_rows(recipe, rows, 2, encode_fp8_bf16_row),
        }
    }
}

impl Offer for Quant {
    fn slices(&self) -> Vec<Slice> {
        self.slices.clone()
    }

    fn floats(&self, name: &str) -> Result<Vec<f32>, Error> {
        let Some(recipe) = self.recipes.get(name) else {
            return self.shards.floats(name);
        };
        match recipe {
            Recipe::Block { form, .. } => Ok(form.unpack_all(&self.assemble_block(recipe, None)?)),
            Recipe::Fp8 { .. } => self.decode_fp8(recipe, None),
        }
    }

    fn packed(&self, name: &str) -> Result<Vec<u8>, Error> {
        match self.recipes.get(name) {
            Some(recipe) => self.assemble(recipe, None),
            None => self.shards.packed(name),
        }
    }

    fn packed_rows(&self, name: &str, rows: (usize, usize)) -> Result<Vec<u8>, Error> {
        match self.recipes.get(name) {
            Some(recipe) => self.assemble(recipe, Some(rows)),
            None => self.shards.packed_rows(name, rows),
        }
    }

    fn packed_share(&self, name: &str, rows: (usize, usize)) -> Result<Vec<u8>, Error> {
        match self.recipes.get(name) {
            Some(recipe) => self.assemble(recipe, Some(rows)),
            None => self.shards.packed_share(name, rows),
        }
    }

    fn packed_axis(&self, name: &str, axis: u32, range: (usize, usize)) -> Result<Vec<u8>, Error> {
        if axis == 0 {
            return self.packed_share(name, range);
        }
        let Some(recipe) = self.recipes.get(name) else {
            return self.shards.packed_axis(name, axis, range);
        };
        let shape = recipe.shape();
        let ax = axis as usize;
        if ax >= shape.len() {
            return self.assemble(recipe, None);
        }
        let all = self.assemble(recipe, None)?;
        let outer: usize = shape[..ax].iter().product();
        let whole = shape[ax];
        let inner: usize = shape[ax + 1..].iter().product();
        let Stored::Form(form) = recipe.stored() else {
            return Ok(all);
        };
        Ok(form
            .share(&all, [outer, whole, inner], [range.0, range.1])
            .unwrap_or(all))
    }

    fn i64s(&self, name: &str) -> Result<Vec<i64>, Error> {
        self.shards.i64s(name)
    }

    fn filed(&self, name: &str) -> Option<Arc<dyn Filed>> {
        if self.recipes.contains_key(name) {
            return None;
        }
        self.shards.filed(name)
    }
}
