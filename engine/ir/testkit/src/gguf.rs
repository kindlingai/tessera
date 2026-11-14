//! Byte pieces of a GGUF file, version 3.

/// A length-prefixed string.
pub fn text(out: &mut Vec<u8>, s: &str) {
    out.extend((s.len() as u64).to_le_bytes());
    out.extend(s.as_bytes());
}

/// The magic, the version, and the table and field counts.
pub fn header(out: &mut Vec<u8>, tensors: usize, fields: usize) {
    out.extend(b"GGUF");
    out.extend(3u32.to_le_bytes());
    out.extend((tensors as u64).to_le_bytes());
    out.extend((fields as u64).to_le_bytes());
}

/// A metadata field: key, value type and value bytes.
pub fn field(out: &mut Vec<u8>, key: &str, ty: u32, bytes: &[u8]) {
    text(out, key);
    out.extend(ty.to_le_bytes());
    out.extend(bytes);
}

/// A tensor table entry. `dims` run fastest first.
pub fn info(out: &mut Vec<u8>, name: &str, dims: &[u64], ty: u32, offset: u64) {
    text(out, name);
    out.extend((dims.len() as u32).to_le_bytes());
    for d in dims {
        out.extend(d.to_le_bytes());
    }
    out.extend(ty.to_le_bytes());
    out.extend(offset.to_le_bytes());
}

/// Zeros up to the next multiple of `alignment`.
pub fn pad(out: &mut Vec<u8>, alignment: usize) {
    out.resize(out.len().next_multiple_of(alignment.max(1)), 0);
}
