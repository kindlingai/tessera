//! Grammars through llguidance.

use crate::Error;
use llguidance::api::TopLevelGrammar;
use llguidance::{ParserFactory, token_bytes_from_tokenizer_json};
use sensai_bundle::tokenize::hf::Hf;
use std::sync::Arc;
use toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};

/// The token tree and the tokenizer, as llguidance reads them.
struct Env {
    trie: TokTrie,
    hf: Hf,
}

impl TokenizerEnv for Env {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, s: &[u8]) -> Vec<TokenId> {
        // Forced bytes may end inside a character.
        let valid = std::str::from_utf8(s).map_or_else(|e| e.valid_up_to(), |_| s.len());
        let (text, rest) = s.split_at(valid);
        let text = std::str::from_utf8(text).unwrap_or_default();
        let mut ids = self.hf.encode(text).unwrap_or_default();
        ids.extend(self.trie.greedy_tokenize(rest));
        ids
    }
}

/// One model's tokens. Build it once, then share it.
pub struct Vocab {
    factory: ParserFactory,
    stops: Arc<[u32]>,
    rows: usize,
}

impl Vocab {
    /// `rows` may exceed the tokenizer's ids. Masks clear the rest.
    pub fn new(tokenizer_json: &[u8], stops: &[u32], rows: usize) -> Result<Self, Error> {
        let json: serde_json::Value = serde_json::from_slice(tokenizer_json)?;
        let words = token_bytes_from_tokenizer_json(&json)?;
        let &eos = stops.first().ok_or("a grammar needs a stop id")?;
        if words.len() > rows || eos as usize >= words.len() {
            return Err(format!("{} tokens do not fit {rows} rows", words.len()).into());
        }
        let info = TokRxInfo::new(words.len() as u32, eos);
        let env = Env {
            trie: TokTrie::from(&info, &words),
            hf: Hf::from_json(tokenizer_json)?,
        };
        let env: TokEnv = Arc::new(env);
        Ok(Self {
            factory: ParserFactory::new_simple(&env)?,
            stops: stops.into(),
            rows,
        })
    }

    /// The `u32` words of one mask.
    pub fn words(&self) -> usize {
        self.rows.div_ceil(32)
    }

    /// Ids for `bytes`, even when they end inside a character.
    pub fn tokenize(&self, bytes: &[u8]) -> Vec<u32> {
        self.factory.tok_env().tokenize_bytes(bytes)
    }

    /// Answers that match a JSON schema.
    pub fn schema(&self, schema: serde_json::Value) -> Result<Matcher, Error> {
        self.matcher(TopLevelGrammar::from_json_schema(schema))
    }

    /// Answers that are any JSON object.
    pub fn object(&self) -> Result<Matcher, Error> {
        self.schema(serde_json::json!({ "type": "object" }))
    }

    /// Answers that match a regular expression.
    pub fn regex(&self, rx: &str) -> Result<Matcher, Error> {
        self.matcher(TopLevelGrammar::from_regex(rx))
    }

    fn matcher(&self, grammar: TopLevelGrammar) -> Result<Matcher, Error> {
        let parser = self.factory.create_parser(grammar)?;
        Ok(Matcher {
            inner: llguidance::Matcher::new(Ok(parser)),
            stops: self.stops.clone(),
            words: self.words(),
        })
    }
}

/// One answer's place in its grammar.
pub struct Matcher {
    inner: llguidance::Matcher,
    stops: Arc<[u32]>,
    words: usize,
}

impl Matcher {
    /// One bit per row, set where the next token fits.
    pub fn mask(&mut self, out: &mut [u32]) -> Result<(), Error> {
        if out.len() != self.words {
            return Err(format!("a mask holds {} words, not {}", self.words, out.len()).into());
        }
        let allowed = self.inner.compute_mask_or_eos()?;
        let bits = allowed.as_slice();
        out.fill(0);
        let n = bits.len().min(out.len());
        out[..n].copy_from_slice(&bits[..n]);
        // Bits past the token tree's end stay clear.
        let tokens = allowed.len();
        if let Some(word) = out.get_mut(tokens / 32) {
            *word &= (1u32 << (tokens % 32)) - 1;
        }
        out.iter_mut().skip(tokens / 32 + 1).for_each(|w| *w = 0);
        // The end token stands for every stop id.
        if allowed.is_allowed(self.stops[0]) {
            for &s in self.stops.iter() {
                out[s as usize / 32] |= 1 << (s % 32);
            }
        }
        Ok(())
    }

    /// Moves past `id`. A stop id ends the grammar.
    pub fn accept(&mut self, id: u32) -> Result<(), Error> {
        let id = if self.stops.contains(&id) {
            self.stops[0]
        } else {
            id
        };
        Ok(self.inner.consume_token(id)?)
    }

    /// Tokens the grammar forces next, already accepted.
    pub fn forced(&mut self) -> Vec<u32> {
        self.inner.consume_ff_tokens()
    }

    /// Takes back the last `n` accepted tokens.
    pub fn rollback(&mut self, n: usize) -> Result<(), Error> {
        Ok(self.inner.rollback(n)?)
    }

    /// Whether the answer may end here.
    pub fn accepting(&mut self) -> bool {
        self.inner.is_accepting().unwrap_or(false)
    }

    /// Whether the grammar ended or failed. No token fits after.
    pub fn stopped(&self) -> bool {
        self.inner.is_stopped()
    }
}

impl sensai_engine::Guide for Matcher {
    fn allowed(&mut self) -> Result<sensai_engine::Allowed, sensai_engine::Fault> {
        let mut bits = vec![0; self.words];
        Matcher::mask(self, &mut bits)?;
        Ok(sensai_engine::Allowed::from_bits(bits))
    }

    fn forced(&mut self) -> Vec<u32> {
        Matcher::forced(self)
    }

    fn accept(&mut self, id: u32) -> Result<(), sensai_engine::Fault> {
        Matcher::accept(self, id)
    }

    fn done(&self) -> bool {
        self.stopped()
    }
}
