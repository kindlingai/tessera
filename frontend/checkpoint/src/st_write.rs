//! Canonical safetensors writer with sorted JSON headers.

use crate::Error;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

/// One tensor slice to write into a safetensors file.
pub(crate) struct StTensor<'a> {
    pub name: &'a str,
    pub dtype: &'a str,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

/// Encodes `s` as an ASCII JSON string.
fn json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    use std::fmt::Write as _;
                    let _ = write!(out, "\\u{u:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Formats sorted compact header JSON padded to 8 bytes.
fn build_header(meta: &BTreeMap<String, String>, tensors: &[&StTensor<'_>]) -> Vec<u8> {
    let mut out = String::from("{");
    if !meta.is_empty() {
        out.push_str("\"__metadata__\":{");
        for (i, (k, v)) in meta.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            json_str(&mut out, k);
            out.push(':');
            json_str(&mut out, v);
        }
        out.push('}');
    }
    let mut off = 0usize;
    for t in tensors {
        if out.len() > 1 {
            out.push(',');
        }
        let end = off + t.data.len();
        json_str(&mut out, t.name);
        out.push_str(":{\"dtype\":");
        json_str(&mut out, t.dtype);
        out.push_str(",\"shape\":[");
        for (i, d) in t.shape.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            use std::fmt::Write as _;
            let _ = write!(out, "{d}");
        }
        use std::fmt::Write as _;
        let _ = write!(out, "],\"data_offsets\":[{off},{end}]}}");
        off = end;
    }
    out.push('}');
    let mut bytes = out.into_bytes();
    let pad = (8 - bytes.len() % 8) % 8;
    bytes.resize(bytes.len() + pad, b' ');
    bytes
}

/// Validates and orders `tensors` by name for writing.
fn order_tensors<'a>(tensors: &'a [StTensor<'a>]) -> Result<Vec<&'a StTensor<'a>>, Error> {
    let mut ordered: Vec<&StTensor<'_>> = tensors.iter().collect();
    ordered.sort_by(|a, b| a.name.cmp(b.name));
    for i in 0..ordered.len() {
        let name = ordered[i].name;
        if name.is_empty() || name == "__metadata__" {
            return Err(Error::Format(format!("invalid tensor name `{name}`")));
        }
        if i > 0 && ordered[i - 1].name == name {
            return Err(Error::Format(format!("duplicate tensor name `{name}`")));
        }
    }
    Ok(ordered)
}

/// Writes canonical safetensors bytes into `writer`.
pub(crate) fn write_to<W: Write>(
    mut writer: W,
    meta: &BTreeMap<String, String>,
    tensors: &[StTensor<'_>],
) -> Result<(), Error> {
    let ordered = order_tensors(tensors)?;
    let header = build_header(meta, &ordered);
    writer.write_all(&(header.len() as u64).to_le_bytes())?;
    writer.write_all(&header)?;
    for t in ordered {
        writer.write_all(t.data)?;
    }
    writer.flush()?;
    Ok(())
}

/// Writes canonical safetensors atomically via a temporary file.
pub(crate) fn write_file(
    path: &Path,
    meta: &BTreeMap<String, String>,
    tensors: &[StTensor<'_>],
) -> Result<(), Error> {
    if let Some(dir) = path.parent()
        && !dir.as_os_str().is_empty()
    {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("safetensors.tmp");
    let file = fs::File::create(&tmp)?;
    write_to(BufWriter::with_capacity(64 << 10, file), meta, tensors)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Primitive numeric scalar with no padding bytes.
pub(crate) trait LeScalar: Copy {
    #[cfg(not(target_endian = "little"))]
    fn push_le(self, out: &mut Vec<u8>);
}

macro_rules! impl_le_scalar {
    ($($t:ty),+ $(,)?) => {
        $(impl LeScalar for $t {
            #[cfg(not(target_endian = "little"))]
            fn push_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        })+
    };
}

impl_le_scalar!(u16, i16, i32, i64, f32);

/// Borrows little-endian bytes on little-endian hosts.
pub(crate) fn as_le_bytes<T: LeScalar>(values: &[T]) -> Cow<'_, [u8]> {
    #[cfg(target_endian = "little")]
    {
        let byte_len = std::mem::size_of_val(values);
        // SAFETY: `T` is a padding-free little-endian scalar.
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len) };
        Cow::Borrowed(bytes)
    }
    #[cfg(not(target_endian = "little"))]
    {
        let mut out = Vec::with_capacity(std::mem::size_of_val(values));
        for &v in values {
            v.push_le(&mut out);
        }
        Cow::Owned(out)
    }
}
