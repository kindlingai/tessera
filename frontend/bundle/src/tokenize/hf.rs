//! Hugging Face `tokenizer.json` files, through the `tokenizers` crate.

use super::Error;
use std::path::Path;

pub struct Hf(tokenizers::Tokenizer);

impl Hf {
    pub fn open(path: &Path) -> Result<Self, Error> {
        Ok(Self(tokenizers::Tokenizer::from_file(path)?))
    }

    /// A `tokenizer.json` held in memory.
    pub fn from_json(bytes: &[u8]) -> Result<Self, Error> {
        Ok(Self(tokenizers::Tokenizer::from_bytes(bytes)?))
    }

    /// Vocabulary ids of `text` alone, without special tokens.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        Ok(self.0.encode(text, false)?.get_ids().to_vec())
    }

    /// Ids of `text` with the special tokens the file adds.
    pub fn encode_special(&self, text: &str) -> Result<Vec<u32>, Error> {
        Ok(self.0.encode(text, true)?.get_ids().to_vec())
    }

    /// The id of one token, such as `<|endoftext|>`.
    pub fn id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }

    /// The text of `ids`, minus special tokens. Empty if unknown.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.0.decode(ids, true).unwrap_or_default()
    }

    /// The text of `ids`, keeping special tokens.
    pub fn decode_raw(&self, ids: &[u32]) -> String {
        self.0.decode(ids, false).unwrap_or_default()
    }

    /// Each id whose text passes `keep`. Special tokens are empty.
    pub fn matching(&self, keep: impl Fn(&str) -> bool) -> Vec<u32> {
        let mut out = Vec::new();
        for id in 0..self.0.get_vocab_size(true) as u32 {
            if keep(&self.decode(&[id])) {
                out.push(id);
            }
        }
        out
    }

    /// The tokenizer as `tokenizer.json` text.
    pub fn json(&self) -> Result<String, Error> {
        self.0.to_string(false)
    }

    /// Ids made only of ASCII digits.
    pub fn digits(&self) -> Vec<u32> {
        self.matching(|t| !t.is_empty() && t.bytes().all(|b| b.is_ascii_digit()))
    }
}
