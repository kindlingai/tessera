//! Parallel packing helpers for exlr8 block and fp8-row matrices.

use crate::exlr8::{Record, coded, rows};
use crate::file::{Error, File};
use crate::offer::narrow;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use tessera_engine::storage::Form;
use tessera_engine::{E4M3, block, trellis};

/// Record field 0: how the matrix is stored.
pub(crate) const TRELLIS: i64 = 1;
pub(crate) const INT4: i64 = 2;
pub(crate) const MXFP8: i64 = 3;
pub(crate) const FP8_BLOCK: i64 = 4;
pub(crate) const FP8_ROW: i64 = 5;
pub(crate) const BF16: i64 = 6;
pub(crate) const FP32: i64 = 7;
pub(crate) const ROWS: i64 = 9;
pub(crate) const INT2: i64 = 10;

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
    let (scale_part, code_part, scale_bytes, code_bytes) = match r.format {
        INT4 => ("s4", "q4", 2, 16),
        INT2 => ("s2", "q2", 2, 8),
        _ => ("e8m0", "fp8", 1, 32),
    };
    if !r.inp.is_multiple_of(32) || first + count > r.out {
        return Err(Error::DType(format!("{name}: {count} rows of {}", r.inp)));
    }
    if count == 0 || r.inp == 0 {
        return Ok(Vec::new());
    }
    let grouped = r.format == INT4 && r.side == trellis::INPUT_SIDE as i64;
    let groups = r.inp / 32;
    let (scales, codes, base_first) = if grouped {
        (
            f.bytes(&format!("{name}.{scale_part}"))?,
            f.bytes(&format!("{name}.{code_part}"))?,
            first,
        )
    } else {
        (
            f.bytes_at(
                &format!("{name}.{scale_part}"),
                (first * groups * scale_bytes) as u64,
                Some((count * groups * scale_bytes) as u64),
            )?,
            f.bytes_at(
                &format!("{name}.{code_part}"),
                (first * groups * code_bytes) as u64,
                Some((count * groups * code_bytes) as u64),
            )?,
            0,
        )
    };
    Ok(pack_blocks(
        &scales,
        &codes,
        r.out,
        groups,
        scale_bytes,
        code_bytes,
        grouped,
        (base_first, count),
    ))
}

/// A matrix in a format other than trellis, as `[out][in]`.
pub(crate) fn plain(f: &File, name: &str, r: &Record, format: i64) -> Result<Vec<f32>, Error> {
    let inp = r.inp;
    let fp8 = || f.bytes(&format!("{name}.fp8"));
    if let Some(form) = block_form(format) {
        return Ok(form.unpack_all(&blocks(f, name, r, (0, r.out))?));
    }
    let lut = e4m3_lut();
    match format {
        FP8_ROW => {
            let scale = f.floats(&format!("{name}.scale"))?;
            let row = |(i, c): (usize, &u8)| lut[*c as usize] * scale[i / inp];
            Ok(fp8()?.iter().enumerate().map(row).collect())
        }
        FP8_BLOCK => {
            let scale = f.floats(&format!("{name}.scale"))?;
            let at = |i: usize| scale[i / inp / 128 * inp.div_ceil(128) + i % inp / 128];
            let value = |(i, c): (usize, &u8)| lut[*c as usize] * at(i);
            Ok(fp8()?.iter().enumerate().map(value).collect())
        }
        BF16 => f.floats(&format!("{name}.bf16")),
        FP32 => f.floats(&format!("{name}.fp32")),
        ROWS => rows(f, name, r, |x| *x),
        other => Err(Error::DType(format!("{name} is in exlr8 format {other}"))),
    }
}

/// Lookup table for all 256 E4M3 byte values.
pub(crate) fn e4m3_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| std::array::from_fn(|b| f32::from(E4M3(b as u8))))
}

/// Packs rows `first..first + count` of block scales and codes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pack_blocks(
    scales: &[u8],
    codes: &[u8],
    out_total: usize,
    groups: usize,
    scale_bytes: usize,
    code_bytes: usize,
    grouped: bool,
    (first, count): (usize, usize),
) -> Vec<u8> {
    let blk = scale_bytes + code_bytes;
    let row_bytes = groups * blk;
    let mut out = vec![0u8; count * row_bytes];
    let pack = |row_off: usize, dst: &mut [u8]| {
        for (ri, dst_row) in dst.chunks_exact_mut(row_bytes).enumerate() {
            let o = first + row_off + ri;
            for (g, slot) in dst_row.chunks_exact_mut(blk).enumerate() {
                let at = match grouped {
                    true => (g / 4 * out_total + o) * 4 + g % 4,
                    false => o * groups + g,
                };
                let (s_dst, c_dst) = slot.split_at_mut(scale_bytes);
                s_dst.copy_from_slice(&scales[at * scale_bytes..(at + 1) * scale_bytes]);
                c_dst.copy_from_slice(&codes[at * code_bytes..(at + 1) * code_bytes]);
            }
        }
    };
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(count);
    if threads <= 1 || count * groups < 4_096 {
        pack(0, &mut out);
    } else {
        let per = count.div_ceil(threads);
        std::thread::scope(|scope| {
            let mut rest = out.as_mut_slice();
            let mut row_off = 0usize;
            while !rest.is_empty() {
                let take = per.min(rest.len() / row_bytes);
                let (head, tail) = rest.split_at_mut(take * row_bytes);
                rest = tail;
                let start = row_off;
                row_off += take;
                scope.spawn(move || pack(start, head));
            }
        });
    }
    out
}

/// Rows `first..first + count` of an fp8-row matrix, as bf16.
pub(crate) fn fp8_row_bf16(
    f: &File,
    name: &str,
    out_total: usize,
    inp: usize,
    (first, count): (usize, usize),
) -> Result<Vec<u8>, Error> {
    if first + count > out_total {
        return Err(Error::DType(format!("{name}: {count} rows of {out_total}")));
    }
    if count == 0 || inp == 0 {
        return Ok(Vec::new());
    }
    let codes = f.bytes_at(
        &format!("{name}.fp8"),
        (first * inp) as u64,
        Some((count * inp) as u64),
    )?;
    let scales = f.bytes_at(
        &format!("{name}.scale"),
        (first * 4) as u64,
        Some((count * 4) as u64),
    )?;
    let lut = e4m3_lut();
    let mut out = vec![0u8; count * inp * 2];
    let convert = |c_chunk: &[u8], s_chunk: &[u8], dst: &mut [u8]| {
        let rows = c_chunk.chunks_exact(inp);
        let scs = s_chunk.as_chunks::<4>().0;
        for ((row, sc), dst_row) in rows.zip(scs).zip(dst.chunks_exact_mut(inp * 2)) {
            let scale = f32::from_le_bytes(*sc);
            for (&b, slot) in row.iter().zip(dst_row.as_chunks_mut::<2>().0) {
                *slot = narrow(&(lut[b as usize] * scale));
            }
        }
    };
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(count);
    if threads <= 1 || count * inp < 65_536 {
        convert(&codes, &scales, &mut out);
    } else {
        let per = count.div_ceil(threads);
        std::thread::scope(|scope| {
            let (mut c_rest, mut s_rest, mut o_rest) =
                (codes.as_slice(), scales.as_slice(), out.as_mut_slice());
            while !s_rest.is_empty() {
                let take = per.min(s_rest.len() / 4);
                let (c_head, c_tail) = c_rest.split_at(take * inp);
                let (s_head, s_tail) = s_rest.split_at(take * 4);
                let (o_head, o_tail) = o_rest.split_at_mut(take * inp * 2);
                (c_rest, s_rest, o_rest) = (c_tail, s_tail, o_tail);
                scope.spawn(move || convert(c_head, s_head, o_head));
            }
        });
    }
    Ok(out)
}

/// Packs a rotated trellis matrix in one allocation.
pub(crate) fn pack_turned(
    f: &File,
    name: &str,
    out_total: usize,
    inp: usize,
    k: u32,
    side: i64,
    rows: Option<(usize, usize)>,
) -> Result<Vec<u8>, Error> {
    let tensor = coded(f, &format!("{name}.trellis"))?.0;
    let suh_name = format!("{name}.suh");
    let svh_name = format!("{name}.svh");
    let suh_len = f.byte_len(&suh_name)? as usize;
    let (at, take, out_rows, svh_at, svh_len) = match rows {
        None => (
            0u64,
            f.byte_len(&tensor)? as usize,
            out_total,
            0u64,
            f.byte_len(&svh_name)? as usize,
        ),
        Some((first, count)) => {
            let entry = inp * 128 * k as usize / 8;
            if side == trellis::INPUT_SIDE as i64 || first % 128 != 0 || count % 128 != 0 {
                return Err(Error::DType(format!(
                    "{name}: rows {first}.. are not whole entries"
                )));
            }
            (
                (first / 128 * entry) as u64,
                count / 128 * entry,
                count,
                (2 * first) as u64,
                2 * count,
            )
        }
    };
    let mut out = trellis::alloc_packed(inp, out_rows, side as u32, take + suh_len + svh_len);
    let codes_end = trellis::HEAD + take;
    let suh_end = codes_end + suh_len;
    f.read_into(&tensor, at, &mut out[trellis::HEAD..codes_end])?;
    f.read_into(&suh_name, 0, &mut out[codes_end..suh_end])?;
    f.read_into(&svh_name, svh_at, &mut out[suh_end..])?;
    Ok(out)
}

/// Opens a batch of local safetensors shards in parallel.
pub(crate) fn open_local(dir: &Path, missing: &[&str]) -> Vec<(String, Arc<File>)> {
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(missing.len());
        for &file in missing {
            handles.push(scope.spawn(move || {
                let path = dir.join(format!("{file}.safetensors"));
                File::open(&path).ok().map(|f| (file.into(), Arc::new(f)))
            }));
        }
        let mut out = Vec::with_capacity(handles.len());
        for h in handles {
            if let Some(pair) = h.join().expect("shard open") {
                out.push(pair);
            }
        }
        out
    })
}
