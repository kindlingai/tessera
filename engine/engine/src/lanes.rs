//! Page-table wire patch codec and `Lanes` re-export.

use tessera_engine_comm::{Reader, Writer};
pub(crate) use tessera_kv::{
    EVICT_COLD, EVICT_HOT, EVICT_MIDDLE, EVICT_PINNED, EVICT_SHARED, Lanes,
};

/// Writes dirty table patches to `w`.
pub(crate) fn encode_patches(w: &mut Writer, patches: &[(u32, &[u32])]) {
    w.u32(patches.len() as u32);
    for &(offset, values) in patches {
        w.u32(offset).u32s(values);
    }
}

/// Reads dirty table patches from `r`.
pub(crate) fn decode_patches(r: &mut Reader) -> Option<Vec<(u32, Vec<u32>)>> {
    let count = r.u32()? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push((r.u32()?, r.u32s()?));
    }
    Some(out)
}
