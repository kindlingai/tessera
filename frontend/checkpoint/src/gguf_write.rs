//! GGUF version 3 bytes, the inverse of `Gguf::open`.

use crate::{GgufType, Meta};

#[derive(Clone)]
pub struct GgufTensor {
    pub name: String,
    pub ty: GgufType,
    /// Fastest first.
    pub dims: Vec<u64>,
    pub bytes: Vec<u8>,
}

fn text(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn meta_size(v: &Meta) -> usize {
    match v {
        Meta::U8(_) | Meta::I8(_) | Meta::Bool(_) => 1,
        Meta::U16(_) | Meta::I16(_) => 2,
        Meta::U32(_) | Meta::I32(_) | Meta::F32(_) => 4,
        Meta::U64(_) | Meta::I64(_) | Meta::F64(_) => 8,
        Meta::Str(s) => 8 + s.len(),
        Meta::Array(items) => 12 + items.iter().map(meta_size).sum::<usize>(),
    }
}

/// Writes `v` into `out` and returns its GGUF type id.
fn value(out: &mut Vec<u8>, v: &Meta) -> u32 {
    match v {
        Meta::U8(x) => {
            out.push(*x);
            0
        }
        Meta::I8(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            1
        }
        Meta::U16(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            2
        }
        Meta::I16(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            3
        }
        Meta::U32(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            4
        }
        Meta::I32(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            5
        }
        Meta::F32(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            6
        }
        Meta::Bool(x) => {
            out.push(u8::from(*x));
            7
        }
        Meta::Str(s) => {
            text(out, s);
            8
        }
        Meta::Array(items) => {
            let elem_at = out.len();
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            let mut elem = 8u32;
            for item in items {
                elem = value(out, item);
            }
            out[elem_at..elem_at + 4].copy_from_slice(&elem.to_le_bytes());
            9
        }
        Meta::U64(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            10
        }
        Meta::I64(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            11
        }
        Meta::F64(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            12
        }
    }
}

fn pad(out: &mut Vec<u8>, alignment: u64) {
    out.resize((out.len() as u64).next_multiple_of(alignment) as usize, 0);
}

/// A whole file. Tensor data follows `general.alignment`, default 32.
pub fn write_gguf(meta: &[(String, Meta)], tensors: &[GgufTensor]) -> Vec<u8> {
    let alignment = match meta.iter().find(|(key, _)| key == "general.alignment") {
        Some((_, Meta::U32(n))) => u64::from(*n),
        Some((_, Meta::U64(n))) => *n,
        _ => 32,
    }
    .max(1);
    let header_len: usize = 24
        + meta
            .iter()
            .map(|(k, v)| 8 + k.len() + 4 + meta_size(v))
            .sum::<usize>()
        + tensors
            .iter()
            .map(|t| 8 + t.name.len() + 4 + 8 * t.dims.len() + 4 + 8)
            .sum::<usize>();
    let aligned_header = (header_len as u64).next_multiple_of(alignment) as usize;
    let mut data_len = 0u64;
    for t in tensors {
        data_len = data_len.next_multiple_of(alignment) + t.bytes.len() as u64;
    }
    let mut out = Vec::with_capacity(aligned_header + data_len as usize);
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    for (key, v) in meta {
        text(&mut out, key);
        let ty_at = out.len();
        out.extend_from_slice(&0u32.to_le_bytes());
        let id = value(&mut out, v);
        out[ty_at..ty_at + 4].copy_from_slice(&id.to_le_bytes());
    }
    let mut offset = 0u64;
    for t in tensors {
        offset = offset.next_multiple_of(alignment);
        text(&mut out, &t.name);
        out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.ty.id().to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        offset += t.bytes.len() as u64;
    }
    pad(&mut out, alignment);
    let data_start = out.len();
    for t in tensors {
        let rel = ((out.len() - data_start) as u64).next_multiple_of(alignment) as usize;
        out.resize(data_start + rel, 0);
        out.extend_from_slice(&t.bytes);
    }
    out
}
