//! Strings and comments in Rust source, and `Cargo.lock` packages.

use std::collections::BTreeSet;

/// A package name and version.
pub type Pin = (String, String);

/// Each package of a `Cargo.lock`.
pub fn packages(lock: &str) -> BTreeSet<Pin> {
    let quoted = |line: &str, key| {
        let value = line
            .strip_prefix(key)?
            .strip_prefix('"')?
            .strip_suffix('"')?;
        Some(value.to_string())
    };
    let mut out = BTreeSet::new();
    let mut name = None;
    for line in lock.lines() {
        if let Some(found) = quoted(line, "name = ") {
            name = Some(found);
        } else if let Some(version) = quoted(line, "version = ")
            && let Some(name) = name.take()
        {
            out.insert((name, version));
        }
    }
    out
}

/// True when the text ends outside strings and comments.
pub fn in_code(lines: &[&str]) -> bool {
    let text = lines.concat();
    let bytes = text.as_bytes();
    let mut at = 0;
    let mut blocks = 0usize;
    while at < bytes.len() {
        let rest = &bytes[at..];
        if blocks > 0 || rest.starts_with(b"/*") {
            let (open, close) = (rest.starts_with(b"/*"), rest.starts_with(b"*/"));
            blocks = blocks + usize::from(open) - usize::from(close);
            at += if open || close { 2 } else { 1 };
        } else if rest.starts_with(b"//") {
            at += rest.iter().position(|c| *c == b'\n').unwrap_or(rest.len());
        } else if bytes[at] == b'"' {
            let Some(end) = string_end(bytes, at) else {
                return false;
            };
            at = end;
        } else if bytes[at] == b'\'' {
            at += char_len(&text[at + 1..]);
        } else {
            at += 1;
        }
    }
    blocks == 0
}

/// The index after the string that opens at `at`.
fn string_end(bytes: &[u8], at: usize) -> Option<usize> {
    let hashes = bytes[..at].iter().rev().take_while(|c| **c == b'#').count();
    let raw = at > hashes && bytes[at - hashes - 1] == b'r';
    let mut next = at + 1;
    while next < bytes.len() {
        match bytes[next] {
            b'\\' if !raw => next += 1,
            b'"' if !raw => return Some(next + 1),
            b'"' => {
                let tail = &bytes[next + 1..];
                if tail.len() >= hashes && tail[..hashes].iter().all(|c| *c == b'#') {
                    return Some(next + 1 + hashes);
                }
            }
            _ => {}
        }
        next += 1;
    }
    None
}

/// The length of a char literal, or one.
fn char_len(after: &str) -> usize {
    let mut chars = after.chars();
    match (chars.next(), chars.next()) {
        (Some('\\'), _) => match after.get(2..).and_then(|rest| rest.find('\'')) {
            Some(end) => end + 4,
            None => 1,
        },
        (Some(c), Some('\'')) => c.len_utf8() + 2,
        _ => 1,
    }
}
