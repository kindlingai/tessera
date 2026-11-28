//! exlr8 matrix formats: records and their decoders.

use crate::file::{Error, File};
use std::ops::Range;
use tessera_engine::storage::Form;
use tessera_engine::trellis::{self, Codebook};
use tessera_engine::{E4M3, F16, block};

/// Record field 0: how the matrix is stored.
pub(crate) const TRELLIS: i64 = 1;
const INT4: i64 = 2;
const MXFP8: i64 = 3;
pub(crate) const FP8_BLOCK: i64 = 4;
pub(crate) const FP8_ROW: i64 = 5;
pub(crate) const BF16: i64 = 6;
pub(crate) const FP32: i64 = 7;
pub(crate) const ROWS: i64 = 9;
const INT2: i64 = 10;

/// A matrix's record: format, shape, trellis width, sharded side.
pub(crate) struct Record {
    pub format: i64,
    pub out: usize,
    pub inp: usize,
    k: u32,
    side: i64,
}

pub(crate) fn record(f: &File, name: &str) -> Result<Record, Error> {
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

pub(crate) fn halves(f: &File, name: &str) -> Result<Vec<f32>, Error> {
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
pub(crate) fn rows<T: Clone + Send>(
    f: &File,
    name: &str,
    r: &Record,
    each: impl Fn(&f32) -> T + Sync,
) -> Result<Vec<T>, Error> {
    let (tensor, book) = coded(f, &format!("{name}.rows"))?;
    let (suh, svh) = sides(f, name)?;
    let bad = || Error::DType(format!("{name}: unreadable K{} {book:?} rows", r.k));
    trellis::rows(r.k, book, &f.bytes(&tensor)?, [&suh, &svh], each).ok_or_else(bad)
}

/// Trellis matrix `name` packed: a header, then its entries.
fn packed(f: &File, name: &str, r: &Record) -> Result<Vec<u8>, Error> {
    let tensor = coded(f, &format!("{name}.trellis"))?.0;
    let len = f.byte_len(&tensor)? as usize;
    let mut out = trellis::alloc_packed(r.inp, r.out, r.side as u32, len);
    f.read_into(&tensor, 0, &mut out[trellis::HEAD..])?;
    Ok(out)
}

/// Trellis matrix `name` as f32 `[out][in]`, before its rotations.
fn decoded(f: &File, name: &str, r: &Record) -> Result<Vec<f32>, Error> {
    let book = coded(f, &format!("{name}.trellis"))?.1;
    let form = trellis::form_for(r.k, book)
        .ok_or_else(|| Error::DType(format!("{name}: K{} {book:?}", r.k)))?;
    let bytes = packed(f, name, r)?;
    Ok(form.unpack_all(&bytes))
}

/// Rotation vectors of matrix `name`: input side, output side.
fn sides(f: &File, name: &str) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let side = |s: &str| halves(f, &format!("{name}.{s}"));
    Ok((side("suh")?, side("svh")?))
}

/// Matrix `name` as f32 `[out][in]`, decoded by its record.
pub(crate) fn matrix(f: &File, name: &str) -> Result<Vec<f32>, Error> {
    let r = record(f, name)?;
    let codes = unrotated(f, name, &r)?;
    if r.format != TRELLIS {
        return Ok(codes);
    }
    let (suh, svh) = sides(f, name)?;
    Ok(trellis::rotated(&codes, r.inp, r.out, &suh, &svh))
}

/// Packed codes of `name`, or `rows` as `(first, count)`.
pub(crate) fn codes_packed(
    f: &File,
    name: &str,
    rows: Option<(usize, usize)>,
) -> Result<Vec<u8>, Error> {
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
    let tensor = coded(f, &format!("{name}.trellis"))?.0;
    let at = (first / 128 * entry) as u64;
    let take = count / 128 * entry;
    let mut out = trellis::alloc_packed(r.inp, count, r.side as u32, take);
    f.read_into(&tensor, at, &mut out[trellis::HEAD..])?;
    Ok(out)
}

/// Matrix `name` as f32 `[out][in]` before its rotations, if any.
pub fn codes(f: &File, name: &str) -> Result<Vec<f32>, Error> {
    unrotated(f, name, &record(f, name)?)
}

fn unrotated(f: &File, name: &str, r: &Record) -> Result<Vec<f32>, Error> {
    match r.format {
        TRELLIS => decoded(f, name, r),
        other => plain(f, name, r, other),
    }
}

/// A matrix in a format other than trellis, as `[out][in]`.
pub(crate) fn plain(f: &File, name: &str, r: &Record, format: i64) -> Result<Vec<f32>, Error> {
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
pub(crate) fn block_form(format: i64) -> Option<&'static dyn Form> {
    match format {
        INT4 => Some(&block::INT4_G32),
        INT2 => Some(&block::INT2_G32),
        MXFP8 => Some(&block::MXFP8),
        _ => None,
    }
}

/// Rows `first..first + count` of a block form matrix.
pub(crate) fn blocks(
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
pub(crate) fn turned(f: &File, name: &str, rows: Option<(usize, usize)>) -> Result<Vec<u8>, Error> {
    let mut bytes = codes_packed(f, name, rows)?;
    bytes.extend(f.bytes(&format!("{name}.suh"))?);
    let svh = f.bytes(&format!("{name}.svh"))?;
    bytes.extend_from_slice(span(&svh, rows, 2));
    Ok(bytes)
}

/// Item range of rows `first..first + count`, each `width` long.
pub(crate) fn bounds(len: usize, rows: Option<(usize, usize)>, width: usize) -> Range<usize> {
    let Some((first, count)) = rows else {
        return 0..len;
    };
    let end = ((first + count) * width).min(len);
    (first * width).min(end)..end
}

/// Rows `first..first + count` of `all`, each `width` long.
pub(crate) fn span<T>(all: &[T], rows: Option<(usize, usize)>, width: usize) -> &[T] {
    &all[bounds(all.len(), rows, width)]
}
