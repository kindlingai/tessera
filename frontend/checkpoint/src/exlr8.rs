//! exlr8 v2 releases: trellis and fp8 matrices decoded to f32.

use crate::file::{Error, File};
use graph::{E4M3, F16};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The codebook multiplier.
const MCG: u32 = 0xCBAC_1FED;

/// Record field 0: how the matrix is stored.
const TRELLIS: i64 = 1;
const FP8_ROW: i64 = 5;
const BF16: i64 = 6;
const FP32: i64 = 7;

/// Record field 4: entries hold 128 input channels.
const INPUT_SIDE: i64 = 2;

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
    let bytes = f.bytes(name)?;
    let half = |b: &[u8; 2]| f32::from(F16(u16::from_le_bytes(*b)));
    Ok(bytes.as_chunks::<2>().0.iter().map(half).collect())
}

fn e4m3(bits: u8) -> f32 {
    f32::from(E4M3(bits))
}

/// Tile position `16 * row + col` per stream value.
fn tile_order() -> [usize; 256] {
    let mut perm = [0; 256];
    for t in 0..32 {
        let (r0, c0) = (2 * (t % 4), t / 4);
        let spots = [
            (0, 0),
            (1, 0),
            (8, 0),
            (9, 0),
            (0, 8),
            (1, 8),
            (8, 8),
            (9, 8),
        ];
        for (i, (dr, dc)) in spots.into_iter().enumerate() {
            perm[8 * t + i] = 16 * (r0 + dr) + c0 + dc;
        }
    }
    perm
}

/// A state's weight: the mcg hash, then E4M3 rounding.
fn codebook(state: u32) -> f32 {
    let x = (state & 0xFFFF).wrapping_mul(MCG);
    let x = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60;
    let low = f32::from(F16((x & 0xFFFF) as u16));
    let high = f32::from(F16((x >> 16) as u16));
    let sum = f32::from(F16::from(low + high));
    f32::from(E4M3::from(sum.clamp(-448.0, 448.0)))
}

/// One packed tile, decoded in tile order.
fn tile(raw: &[u8], k: u32, order: &[usize; 256]) -> [f32; 256] {
    // Little-endian words, pairs swapped, bits most significant first.
    let mut words: Vec<u16> = raw
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect();
    for pair in words.as_chunks_mut::<2>().0 {
        pair.swap(0, 1);
    }
    let bit = |i: usize| u32::from(words[i / 16] >> (15 - i % 16) & 1);
    let value = |i: usize| (0..k as usize).fold(0, |v, b| (v << 1) | bit(i * k as usize + b));
    let values: Vec<u32> = (0..256).map(value).collect();
    // A state: the 16 bits ending at its value, wrapping.
    let mut out = [0.0; 256];
    for (i, &at) in order.iter().enumerate() {
        let mut state = 0;
        for j in 0..16usize.div_ceil(k as usize) {
            state |= values[(i + 256 - j) % 256] << (j * k as usize);
        }
        out[at] = codebook(state);
    }
    out
}

/// Entries into `T[in][out]`, 128 channels of a side each.
fn trellis(f: &File, name: &str, r: &Record) -> Result<Vec<f32>, Error> {
    let raw = f.bytes(&format!("{name}.trellis"))?;
    let shape = f.shape(&format!("{name}.trellis")).expect("read above");
    let (entries, bytes) = (shape[0], shape[1]);
    let (step, order) = (32 * r.k as usize, tile_order());
    let mut t = vec![0.0; r.inp * r.out];
    let mut put = |entry: &[u8], at: usize, row0: usize, col0: usize| {
        for (i, v) in tile(&entry[at..at + step], r.k, &order).iter().enumerate() {
            t[(row0 + i / 16) * r.out + col0 + i % 16] = *v;
        }
    };
    for (g, entry) in raw.chunks_exact(bytes).enumerate().take(entries) {
        if r.side == INPUT_SIDE {
            // Rows 128g..: tiles by output block, then row, then column.
            for b in 0..r.out / 128 {
                for row in 0..8 {
                    for col in 0..8 {
                        let at = (64 * b + 8 * row + col) * step;
                        put(entry, at, 128 * g + 16 * row, 128 * b + 16 * col);
                    }
                }
            }
        } else {
            // Columns 128g..: tiles by row, then column.
            for row in 0..r.inp / 16 {
                for col in 0..8 {
                    put(entry, (8 * row + col) * step, 16 * row, 128 * g + 16 * col);
                }
            }
        }
    }
    Ok(t)
}

/// The scaled Walsh-Hadamard transform of 128 values at `stride`.
fn hadamard(x: &mut [f32], start: usize, stride: usize) {
    let mut len = 1;
    while len < 128 {
        for i in (0..128).step_by(2 * len) {
            for j in i..i + len {
                let (a, b) = (x[start + j * stride], x[start + (j + len) * stride]);
                x[start + j * stride] = a + b;
                x[start + (j + len) * stride] = a - b;
            }
        }
        len *= 2;
    }
    let scale = 1.0 / 128f32.sqrt();
    for j in 0..128 {
        x[start + j * stride] *= scale;
    }
}

/// `diag(suh) H T H diag(svh)`, transposed to `[out][in]`.
fn rotate(mut t: Vec<f32>, r: &Record, suh: &[f32], svh: &[f32]) -> Vec<f32> {
    let (inp, out) = (r.inp, r.out);
    for block in 0..inp / 128 {
        for col in 0..out {
            hadamard(&mut t, 128 * block * out + col, out);
        }
    }
    for (row, s) in suh.iter().enumerate() {
        t[row * out..(row + 1) * out]
            .iter_mut()
            .for_each(|v| *v *= s);
    }
    for row in 0..inp {
        for block in 0..out / 128 {
            hadamard(&mut t, row * out + 128 * block, 1);
        }
    }
    let mut w = vec![0.0; inp * out];
    for row in 0..inp {
        for (col, s) in svh.iter().enumerate() {
            w[col * inp + row] = t[row * out + col] * s;
        }
    }
    w
}

/// Matrix `name` as f32 `[out][in]`, decoded by its record.
pub fn matrix(f: &File, name: &str) -> Result<Vec<f32>, Error> {
    let r = record(f, name)?;
    match r.format {
        TRELLIS => {
            let suh = halves(f, &format!("{name}.suh"))?;
            let svh = halves(f, &format!("{name}.svh"))?;
            Ok(rotate(trellis(f, name, &r)?, &r, &suh, &svh))
        }
        FP8_ROW => {
            let codes = f.bytes(&format!("{name}.fp8"))?;
            let scale = f.floats(&format!("{name}.scale"))?;
            let row = |(i, c): (usize, &u8)| e4m3(*c) * scale[i / r.inp];
            Ok(codes.iter().enumerate().map(row).collect())
        }
        BF16 => f.floats(&format!("{name}.bf16")),
        FP32 => f.floats(&format!("{name}.fp32")),
        other => Err(Error::DType(format!("{name} is in exlr8 format {other}"))),
    }
}

/// An EmbeddingGemma 2 release, read by HF tensor name.
pub struct Release {
    /// A directory, or a URL prefix ending in `/`.
    at: Where,
    files: RefCell<HashMap<String, File>>,
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
        let url = format!("https://huggingface.co/{repo}/resolve/{revision}/");
        Self::new(Where::Url(url))
    }

    fn new(at: Where) -> Self {
        Self {
            at,
            files: RefCell::new(HashMap::new()),
        }
    }

    /// The file holding `name`, and the name inside it.
    fn place(name: &str) -> (String, String) {
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

    /// Tensor `name` as f32: matrices decoded, vectors as stored.
    pub fn read(&self, name: &str) -> Result<Vec<f32>, Error> {
        let (file, local) = Self::place(name);
        self.with(&file, |f| {
            let matrix_name = local.strip_suffix(".weight").unwrap_or(&local);
            match f.shape(&format!("{matrix_name}.record")) {
                Some(_) => matrix(f, matrix_name),
                None => f.floats(&local),
            }
        })
    }
}
