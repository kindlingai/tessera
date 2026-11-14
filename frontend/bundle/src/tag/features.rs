// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

//! A token's spaCy features, and their hash table rows.

use super::symbols::{BASE_NORMS, SYMBOLS};
use crate::tokenize::spacy::{Word, is_space};
use std::collections::HashMap;

/// MurmurHash64A with seed 1: spaCy's `hash_string`.
pub fn hash_string(s: &str) -> u64 {
    const M: u64 = 0xc6a4_a793_5bd1_e995;
    let key = s.as_bytes();
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

/// Thinc's `ops.hash`: MurmurHash3's x64 128-bit mix on one u64.
pub fn thinc_hash(val: u64, seed: u32) -> [u32; 4] {
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

/// spaCy's `word_shape`: case, digits and runs capped at four.
pub fn shape(text: &str) -> String {
    if text.chars().count() >= 100 {
        return "LONG".into();
    }
    let (mut out, mut last, mut run) = (String::new(), None, 0);
    for c in text.chars() {
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
    out
}

/// Strings to ids, and texts to norms.
pub struct Features {
    symbols: HashMap<&'static str, u64>,
    base: HashMap<&'static str, &'static str>,
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

    /// The first of: special case, base, table, lower case.
    pub fn norm(&self, text: &str, word: &Word) -> String {
        if let Some(n) = &word.norm {
            return n.clone();
        }
        if let Some(n) = self.base.get(text) {
            return (*n).into();
        }
        let lower = text.to_lowercase();
        self.lexeme
            .get(&hash_string(&lower))
            .cloned()
            .unwrap_or(lower)
    }

    /// NORM, PREFIX, SUFFIX, SHAPE, SPACY and IS_SPACE, as ids.
    pub fn of(&self, text: &str, word: &Word) -> [u64; 6] {
        let chars: Vec<char> = text.chars().collect();
        let prefix: String = chars.iter().take(1).collect();
        let suffix: String = chars[chars.len().saturating_sub(3)..].iter().collect();
        let blank = chars.iter().all(|&c| is_space(c));
        [
            self.id(&self.norm(text, word)),
            self.id(&prefix),
            self.id(&suffix),
            self.id(&shape(text)),
            u64::from(word.space_after),
            u64::from(blank),
        ]
    }
}
