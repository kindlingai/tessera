//! What a recipe module links: memory hand-off, the table format.

pub mod site;

#[cfg(test)]
mod lib_test;

/// The version of the JSON exchange with the host.
pub const ABI: u32 = 5;

/// Room for `len` bytes, kept until the module ends.
pub fn alloc(len: i32) -> i32 {
    let room = vec![0u8; len.max(0) as usize].leak();
    room.as_mut_ptr() as usize as i32
}

/// # Safety
///
/// `at` and `len` must come from `alloc`.
pub unsafe fn input(at: i32, len: i32) -> &'static [u8] {
    let at = at as usize as *const u8;
    // SAFETY: the caller passes back what `alloc` returned.
    unsafe { std::slice::from_raw_parts(at, len.max(0) as usize) }
}

/// `bytes` for the host, as `at << 32 | len`.
pub fn output(bytes: Vec<u8>) -> i64 {
    let bytes = bytes.leak();
    let at = bytes.as_ptr() as usize as u64;
    ((at << 32) | bytes.len() as u64) as i64
}

/// Named tables of floats, as one buffer the host decodes.
pub fn encode(tables: &[(String, Vec<f32>)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend((tables.len() as u32).to_le_bytes());
    for (name, values) in tables {
        out.extend((name.len() as u32).to_le_bytes());
        out.extend(name.as_bytes());
        out.extend((values.len() as u32).to_le_bytes());
        out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    }
    out
}

/// The tables `encode` wrote, or `None` if bad.
pub fn decode(bytes: &[u8]) -> Option<Vec<(String, Vec<f32>)>> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let part = bytes.get(at..at.checked_add(n)?)?;
        at += n;
        Some(part)
    };
    let word = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let count = word(take(4)?);
    let mut tables = Vec::new();
    for _ in 0..count {
        let len = word(take(4)?);
        let name = String::from_utf8(take(len)?.to_vec()).ok()?;
        let n = word(take(4)?);
        let raw = take(n.checked_mul(4)?)?;
        let values = raw
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        tables.push((name, values.collect()));
    }
    Some(tables)
}
