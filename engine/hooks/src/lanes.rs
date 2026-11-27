//! Page-table wire patch codec and `Lanes` re-export.

use tessera_engine_comm::{Reader, Writer};
pub use tessera_kv::Lanes;

/// Writes dirty table patches to `w`.
pub fn encode_patches(w: &mut Writer, patches: &[(u32, &[u32])]) {
    w.u32(patches.len() as u32);
    for &(offset, values) in patches {
        w.u32(offset).u32s(values);
    }
}

/// Reads dirty table patches from `r`.
pub fn decode_patches(r: &mut Reader) -> Option<Vec<(u32, Vec<u32>)>> {
    let count = r.u32()? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push((r.u32()?, r.u32s()?));
    }
    Some(out)
}
