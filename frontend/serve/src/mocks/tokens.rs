//! Tokens for `/tokenize`, `/detokenize` and prompts.

use tessera_tasks::Error;
use tessera_tasks::api::Tokens;

/// One id per character: its code point.
pub(crate) struct MockTokens;

impl Tokens for MockTokens {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        Ok(text.chars().map(u32::from).collect())
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter().filter_map(|&id| char::from_u32(id)).collect()
    }
}
