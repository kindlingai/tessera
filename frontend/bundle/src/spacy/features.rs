// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

//! A token's spaCy features, and their hash table rows.

use super::symbols::{BASE_NORMS, SYMBOLS};
use crate::tokenize::spacy::{Word, is_space};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// MurmurHash64A with seed 1 over raw bytes.
fn hash_bytes(key: &[u8]) -> u64 {
    const M: u64 = 0xc6a4_a793_5bd1_e995;
    let mut h = 1 ^ (key.len() as u64).wrapping_mul(M);
    let (chunks, tail) = key.as_chunks::<8>();
    for c in chunks {
        let mut k = u64::from_le_bytes(*c);
        k = k.wrapping_mul(M);
        k ^= k >> 47;
        h ^= k.wrapping_mul(M);
        h = h.wrapping_mul(M);
    }
    if !tail.is_empty() {
        for (i, b) in tail.iter().enumerate() {
            h ^= u64::from(*b) << (8 * i);
        }
        h = h.wrapping_mul(M);
    }
    h ^= h >> 47;
    h = h.wrapping_mul(M);
    h ^ (h >> 47)
}

/// MurmurHash64A with seed 1: spaCy's `hash_string`.
pub(super) fn hash_string(s: &str) -> u64 {
    hash_bytes(s.as_bytes())
}

/// Thinc's `ops.hash`: MurmurHash3's x64 128-bit mix on one u64.
pub(super) fn thinc_hash(val: u64, seed: u32) -> [u32; 4] {
    let fmix = |mut h: u64| {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^ (h >> 33)
    };
    let mut h1 = val.wrapping_mul(0x87c3_7b91_1142_53d5).rotate_left(31);
    h1 = h1.wrapping_mul(0x4cf5_ad43_2745_937f) ^ u64::from(seed) ^ 8;
    let mut h2 = u64::from(seed) ^ 8;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix(h1);
    h2 = fmix(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    [h1 as u32, (h1 >> 32) as u32, h2 as u32, (h2 >> 32) as u32]
}

/// Non-cryptographic hasher backed by `hash_bytes` for static tables.
#[derive(Default)]
struct WordHash(u64);

impl Hasher for WordHash {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        self.0 = self.0.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ hash_bytes(bytes);
    }

    fn write_u8(&mut self, i: u8) {
        self.0 = self.0.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ u64::from(i);
    }
}

type Table<V> = HashMap<&'static str, V, BuildHasherDefault<WordHash>>;

/// Computes `word_shape` into a stack buffer when ASCII.
pub(super) fn with_shape<R>(text: &str, f: impl FnOnce(&str) -> R) -> R {
    if text.is_ascii() {
        if text.len() >= 100 {
            return f("LONG");
        }
        let mut buf = [0u8; 100];
        let (mut len, mut last, mut run) = (0usize, 0u8, 0u8);
        for b in text.bytes() {
            let s = match b {
                b'A'..=b'Z' => b'X',
                b'a'..=b'z' => b'x',
                b'0'..=b'9' => b'd',
                other => other,
            };
            if len > 0 && s == last {
                run += 1;
            } else {
                (last, run) = (s, 0);
            }
            if run < 4 {
                buf[len] = s;
                len += 1;
            }
        }
        let shaped = std::str::from_utf8(&buf[..len]).expect("ascii");
        return f(shaped);
    }
    let (mut out, mut last, mut run, mut count) = (String::with_capacity(text.len()), None, 0, 0);
    for c in text.chars() {
        count += 1;
        if count >= 100 {
            return f("LONG");
        }
        let s = match c {
            c if c.is_alphabetic() && c.is_uppercase() => 'X',
            c if c.is_alphabetic() => 'x',
            c if c.is_ascii_digit() => 'd',
            c => c,
        };
        if Some(s) == last {
            run += 1;
        } else {
            (last, run) = (Some(s), 0);
        }
        if run < 4 {
            out.push(s);
        }
    }
    f(&out)
}

/// Strings to ids, and texts to norms.
pub struct Features {
    symbols: Table<u64>,
    base: Table<&'static str>,
    /// The pipeline's `lexeme_norm` table, by hash of the lower case.
    lexeme: HashMap<u64, String>,
}

impl Features {
    pub fn new(lexeme: HashMap<u64, String>) -> Self {
        Self {
            symbols: SYMBOLS
                .iter()
                .enumerate()
                .map(|(i, s)| (*s, i as u64))
                .collect(),
            base: BASE_NORMS.into_iter().collect(),
            lexeme,
        }
    }

    /// A string's id: its symbol number, else its hash.
    pub fn id(&self, s: &str) -> u64 {
        self.symbols
            .get(s)
            .copied()
            .unwrap_or_else(|| hash_string(s))
    }

    fn lexeme_or<'a>(&'a self, lower: &'a str) -> &'a str {
        if self.lexeme.is_empty() {
            return lower;
        }
        self.lexeme
            .get(&hash_string(lower))
            .map_or(lower, String::as_str)
    }

    /// Passes the normalized token text to `f` without extra clones.
    pub(super) fn with_norm<R>(&self, text: &str, word: &Word, f: impl FnOnce(&str) -> R) -> R {
        if let Some(n) = &word.norm {
            return f(n);
        }
        if let Some(&n) = self.base.get(text) {
            return f(n);
        }
        if text.is_ascii() {
            if !text.bytes().any(|b| b.is_ascii_uppercase()) {
                return f(self.lexeme_or(text));
            }
            if text.len() <= 64 {
                let mut buf = [0u8; 64];
                for (dst, b) in buf.iter_mut().zip(text.bytes()) {
                    *dst = b.to_ascii_lowercase();
                }
                let lower = std::str::from_utf8(&buf[..text.len()]).expect("ascii");
                return f(self.lexeme_or(lower));
            }
        }
        let lower = text.to_lowercase();
        f(self.lexeme_or(&lower))
    }

    /// NORM, PREFIX, SUFFIX, SHAPE, SPACY and IS_SPACE, as ids.
    pub fn of(&self, text: &str, word: &Word) -> [u64; 6] {
        let prefix_end = text.chars().next().map_or(0, char::len_utf8);
        let suffix_start = text.char_indices().rev().nth(2).map_or(0, |(i, _)| i);
        let blank = text.chars().all(is_space);
        [
            self.with_norm(text, word, |n| self.id(n)),
            self.id(&text[..prefix_end]),
            self.id(&text[suffix_start..]),
            with_shape(text, |s| self.id(s)),
            u64::from(word.space_after),
            u64::from(blank),
        ]
    }
}
