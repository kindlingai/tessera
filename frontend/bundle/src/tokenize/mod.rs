//! Text to tokens: Hugging Face vocabularies and spaCy's rules.

pub mod hf;
pub mod spacy;

use std::ops::Range;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// A token's byte range, and its vocabulary id if any.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    pub span: Range<usize>,
    pub id: Option<u32>,
}

pub trait Tokenizer {
    /// The tokens of `text`, in order.
    fn tokenize(&self, text: &str) -> Result<Vec<Token>, Error>;
}
