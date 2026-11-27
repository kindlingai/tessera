//! Text to tokens: Hugging Face vocabularies and spaCy's rules.

pub mod hf;
pub mod spacy;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
