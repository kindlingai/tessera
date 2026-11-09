//! Byte-level corpus and tokenizer.

use std::collections::HashMap;

/// Tiny shakespeare slice (karpathy/char-rnn, public domain).
pub const DEFAULT_TEXT: &str = include_str!("../data/tiny.txt");

pub struct Corpus {
    bytes: Vec<u8>,
    vocab: usize,
    stoi: HashMap<u8, u32>,
}

impl Corpus {
    pub fn from_text(text: &str) -> Self {
        let bytes = text.as_bytes().to_vec();
        let mut vocab: Vec<u8> = bytes.clone();
        vocab.sort_unstable();
        vocab.dedup();
        let stoi = vocab
            .iter()
            .enumerate()
            .map(|(i, &b)| (b, i as u32))
            .collect();
        Self {
            bytes,
            vocab: vocab.len(),
            stoi,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    /// Deterministic windows of block tokens.
    pub fn batch(&self, start: usize, batch: usize, block: usize) -> Vec<u32> {
        let n = self.bytes.len();
        (0..batch)
            .flat_map(|b| {
                let off = (start + b * block * 7) % (n - block - 1);
                self.bytes[off..off + block].iter().map(|c| self.stoi[c])
            })
            .collect()
    }
}
