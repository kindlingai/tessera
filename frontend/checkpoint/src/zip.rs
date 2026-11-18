//! Stored zip entries, found through the central directory.

use crate::Error;
use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};

const MAX_ENTRIES: u64 = 1 << 20;
const MAX_DIRECTORY: u64 = 64 << 20;
const END: u32 = 0x0605_4b50;
const END64: u32 = 0x0606_4b50;
const LOCATOR: u32 = 0x0706_4b50;
const CENTRAL: u32 = 0x0201_4b50;
const LOCAL: u32 = 0x0403_4b50;
const WIDE: u32 = u32::MAX;

/// An entry's bytes: `len` of them from `start`.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub start: u64,
    pub len: u64,
}

fn bad(what: impl std::fmt::Display) -> Error {
    Error::Format(format!("zip: {what}"))
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"))
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

/// `len` bytes at `at`, inside a file of `size` bytes.
pub(crate) fn read_at(file: &fs::File, at: u64, len: u64, size: u64) -> Result<Vec<u8>, Error> {
    if at.checked_add(len).is_none_or(|end| end > size) {
        return Err(bad("a range runs past the end"));
    }
    let mut out = vec![0u8; len as usize];
    let mut file = file;
    file.seek(SeekFrom::Start(at))?;
    file.read_exact(&mut out)?;
    Ok(out)
}

/// The central directory's offset, length and entry count.
fn directory(file: &fs::File, size: u64) -> Result<(u64, u64, u64), Error> {
    let tail_len = size.min(22 + 65_535);
    let tail = read_at(file, size - tail_len, tail_len, size)?;
    let found = (0..tail.len().saturating_sub(21))
        .rev()
        .find(|&i| u32_at(&tail, i) == END);
    let i = found.ok_or_else(|| bad("no end of central directory"))?;
    let end_at = size - tail_len + i as u64;
    let (mut count, mut len, mut at) = (
        u16_at(&tail, i + 10) as u64,
        u32_at(&tail, i + 12) as u64,
        u32_at(&tail, i + 16) as u64,
    );
    if count == 0xffff || len == WIDE as u64 || at == WIDE as u64 {
        let locator = read_at(
            file,
            end_at.checked_sub(20).ok_or_else(|| bad("no zip64"))?,
            20,
            size,
        )?;
        if u32_at(&locator, 0) != LOCATOR {
            return Err(bad("no zip64 locator"));
        }
        let end = read_at(file, u64_at(&locator, 8), 56, size)?;
        if u32_at(&end, 0) != END64 {
            return Err(bad("no zip64 end record"));
        }
        (count, len, at) = (u64_at(&end, 32), u64_at(&end, 40), u64_at(&end, 48));
    }
    if count > MAX_ENTRIES || len > MAX_DIRECTORY {
        return Err(bad("the directory is too large"));
    }
    Ok((at, len, count))
}

/// Sizes and offset, widened by a zip64 extra field.
fn widen(extra: &[u8], mut fields: [u64; 3]) -> Result<[u64; 3], Error> {
    let mut at = 0;
    while at + 4 <= extra.len() {
        let (id, len) = (u16_at(extra, at), u16_at(extra, at + 2) as usize);
        let body = extra
            .get(at + 4..at + 4 + len)
            .ok_or_else(|| bad("an extra field runs over"))?;
        if id == 1 {
            let mut read = 0;
            for f in fields.iter_mut().filter(|f| **f == WIDE as u64) {
                let word = body
                    .get(read..read + 8)
                    .ok_or_else(|| bad("a short zip64 field"))?;
                *f = u64::from_le_bytes(word.try_into().expect("8 bytes"));
                read += 8;
            }
        }
        at += 4 + len;
    }
    Ok(fields)
}

/// A name that stays inside the archive.
fn plain(name: &str) -> bool {
    !name.starts_with('/')
        && !name.contains('\\')
        && name.split('/').all(|part| !part.is_empty() && part != "..")
}

/// Every entry by name. Entries must be stored, not compressed.
pub fn entries(file: &fs::File) -> Result<BTreeMap<String, Entry>, Error> {
    let size = file.metadata()?.len();
    let (at, len, count) = directory(file, size)?;
    let dir = read_at(file, at, len, size)?;
    let mut out = BTreeMap::new();
    let mut i = 0;
    for _ in 0..count {
        let head = dir
            .get(i..i + 46)
            .ok_or_else(|| bad("the directory ends early"))?;
        if u32_at(head, 0) != CENTRAL {
            return Err(bad("a directory entry has the wrong signature"));
        }
        let (flags, method) = (u16_at(head, 8), u16_at(head, 10));
        let lens = [28, 30, 32].map(|at| u16_at(head, at) as usize);
        let fields = [24, 20, 42].map(|at| u32_at(head, at) as u64);
        let tail = dir.get(i + 46..i + 46 + lens.iter().sum::<usize>());
        let tail = tail.ok_or_else(|| bad("the directory ends early"))?;
        let name = std::str::from_utf8(&tail[..lens[0]]).map_err(bad)?;
        let [stored, packed, local] = widen(&tail[lens[0]..lens[0] + lens[1]], fields)?;
        if flags & 1 != 0 || method != 0 || stored != packed {
            return Err(bad(format_args!("{name} is encrypted or compressed")));
        }
        if !plain(name) {
            return Err(bad(format_args!("{name:?} leaves the archive")));
        }
        let header = read_at(file, local, 30, size)?;
        if u32_at(&header, 0) != LOCAL {
            return Err(bad(format_args!("{name} has no local header")));
        }
        let skip = 30 + u16_at(&header, 26) as u64 + u16_at(&header, 28) as u64;
        let start = local + skip;
        if start.checked_add(stored).is_none_or(|end| end > size) {
            return Err(bad(format_args!("{name} runs past the end")));
        }
        let entry = Entry { start, len: stored };
        if out.insert(name.to_string(), entry).is_some() {
            return Err(bad(format_args!("{name} appears twice")));
        }
        i += 46 + lens.iter().sum::<usize>();
    }
    Ok(out)
}
