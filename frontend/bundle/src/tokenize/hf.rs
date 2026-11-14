//! Hugging Face `tokenizer.json` files, through the `tokenizers` crate.

use super::{Error, Token, Tokenizer};
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

    /// The text of `ids`, minus special tokens. Empty if unknown.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.0.decode(ids, true).unwrap_or_default()
    }
}

impl Tokenizer for Hf {
    fn tokenize(&self, text: &str) -> Result<Vec<Token>, Error> {
        let encoding = self.0.encode(text, false)?;
        let mut out = Vec::new();
        for (&id, &(start, end)) in encoding.get_ids().iter().zip(encoding.get_offsets()) {
            out.push(Token {
                span: start..end,
                id: Some(id),
            });
        }
        Ok(out)
    }
}
