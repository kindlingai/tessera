//! Reads one file from an Actions artifact zip.

const END: u32 = 0x0605_4b50;
const ENTRY: u32 = 0x0201_4b50;
const LOCAL: u32 = 0x0403_4b50;
const STORED: u16 = 0;
const DEFLATED: u16 = 8;

fn u16_at(data: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(data.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?))
}

/// The bytes of the entry `name`.
// Entries use data descriptors, so sizes come from the directory.
pub fn file(data: &[u8], name: &str) -> Option<Vec<u8>> {
    let end = (0..data.len().saturating_sub(21))
        .rev()
        .find(|&at| u32_at(data, at) == Some(END))?;
    let count = u16_at(data, end + 10)?;
    let mut at = u32_at(data, end + 16)? as usize;
    for _ in 0..count {
        if u32_at(data, at)? != ENTRY {
            return None;
        }
        let method = u16_at(data, at + 10)?;
        let packed = u32_at(data, at + 20)? as usize;
        let name_len = u16_at(data, at + 28)? as usize;
        let skip = name_len + (u16_at(data, at + 30)? + u16_at(data, at + 32)?) as usize;
        let local = u32_at(data, at + 42)? as usize;
        if data.get(at + 46..at + 46 + name_len)? == name.as_bytes() {
            if u32_at(data, local)? != LOCAL {
                return None;
            }
            let start =
                local + 30 + (u16_at(data, local + 26)? + u16_at(data, local + 28)?) as usize;
            let packed = data.get(start..start + packed)?;
            return match method {
                STORED => Some(packed.to_vec()),
                DEFLATED => miniz_oxide::inflate::decompress_to_vec(packed).ok(),
                _ => None,
            };
        }
        at += 46 + skip;
    }
    None
}
