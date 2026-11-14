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
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

/// Writes `v` and returns its GGUF type id.
fn value(out: &mut Vec<u8>, v: &Meta) -> u32 {
    let (id, bytes) = match v {
        Meta::U8(x) => (0, x.to_le_bytes().to_vec()),
        Meta::I8(x) => (1, x.to_le_bytes().to_vec()),
        Meta::U16(x) => (2, x.to_le_bytes().to_vec()),
        Meta::I16(x) => (3, x.to_le_bytes().to_vec()),
        Meta::U32(x) => (4, x.to_le_bytes().to_vec()),
        Meta::I32(x) => (5, x.to_le_bytes().to_vec()),
        Meta::F32(x) => (6, x.to_le_bytes().to_vec()),
        Meta::Bool(x) => (7, vec![u8::from(*x)]),
        Meta::Str(s) => {
            text(out, s);
            return 8;
        }
        Meta::Array(items) => {
            let mut body = Vec::new();
            let mut elem = 8;
            for item in items {
                elem = value(&mut body, item);
            }
            out.extend(elem.to_le_bytes());
            out.extend((items.len() as u64).to_le_bytes());
            out.extend(body);
            return 9;
        }
        Meta::U64(x) => (10, x.to_le_bytes().to_vec()),
        Meta::I64(x) => (11, x.to_le_bytes().to_vec()),
        Meta::F64(x) => (12, x.to_le_bytes().to_vec()),
    };
    out.extend(bytes);
    id
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
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((meta.len() as u64).to_le_bytes());
    for (key, v) in meta {
        text(&mut out, key);
        let mut bytes = Vec::new();
        out.extend(value(&mut bytes, v).to_le_bytes());
        out.extend(bytes);
    }
    let mut data = Vec::new();
    for t in tensors {
        pad(&mut data, alignment);
        text(&mut out, &t.name);
        out.extend((t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            out.extend(d.to_le_bytes());
        }
        out.extend(t.ty.id().to_le_bytes());
        out.extend((data.len() as u64).to_le_bytes());
        data.extend(&t.bytes);
    }
    pad(&mut out, alignment);
    out.extend(data);
    out
}
