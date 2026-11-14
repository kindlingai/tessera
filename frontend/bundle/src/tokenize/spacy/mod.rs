// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

//! spaCy's rule-based tokenizer, read from a pipeline's `tokenizer` file.

#[cfg(test)]
mod spacy_test;
mod special;

use super::{Error, Token, Tokenizer};
use fancy_regex::Regex;
use serde::Deserialize;
use special::Phrases;
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

/// spaCy's attribute ids for a token's text and its norm.
const ORTH: u64 = 65;
const NORM: u64 = 67;

#[derive(Deserialize)]
struct Saved {
    prefix_search: Option<String>,
    suffix_search: Option<String>,
    infix_finditer: Option<String>,
    token_match: Option<String>,
    url_match: Option<String>,
    exceptions: HashMap<String, Vec<HashMap<u64, String>>>,
    faster_heuristics: bool,
}

/// A token with spaCy's norm and trailing-space flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Word {
    pub span: Range<usize>,
    /// Set when a special case gives one.
    pub norm: Option<String>,
    /// One space follows it and belongs to no token.
    pub space_after: bool,
}

/// One token of a special case.
struct Piece {
    orth: String,
    norm: Option<String>,
}

/// Prefixes, the rest, then suffixes as they came off.
type Affixes = (Vec<Range<usize>>, Range<usize>, Vec<Range<usize>>);

pub struct Spacy {
    prefix: Option<Regex>,
    suffix: Option<Regex>,
    infix: Option<Regex>,
    token_match: Option<Regex>,
    url_match: Option<Regex>,
    /// Text that always splits into these pieces.
    specials: HashMap<String, Vec<Piece>>,
    phrases: Phrases,
}

/// Python's `str.isspace`, which also counts four separator controls.
pub(crate) fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn plain(span: Range<usize>) -> Word {
    Word {
        span,
        norm: None,
        space_after: false,
    }
}

/// The length of the first match, or 0. Python's `search`.
fn match_len(re: &Option<Regex>, s: &str) -> Result<usize, Error> {
    let Some(re) = re else { return Ok(0) };
    Ok(re.find(s)?.map_or(0, |m| m.end() - m.start()))
}

/// Whether a match starts at 0. Python's `match`.
fn matches_start(re: &Option<Regex>, s: &str) -> Result<bool, Error> {
    let Some(re) = re else { return Ok(false) };
    Ok(re.find(s)?.is_some_and(|m| m.start() == 0))
}

impl Spacy {
    /// Reads `dir/tokenizer`, as a spaCy pipeline saves it.
    pub fn open(dir: &Path) -> Result<Self, Error> {
        Self::from_saved(&std::fs::read(dir.join("tokenizer"))?)
    }

    /// Reads the msgpack bytes of a saved `tokenizer` file.
    pub fn from_saved(bytes: &[u8]) -> Result<Self, Error> {
        let saved: Saved = rmp_serde::from_slice(bytes)?;
        let compile = |p: &Option<String>| -> Result<Option<Regex>, Error> {
            Ok(p.as_deref().map(Regex::new).transpose()?)
        };
        let mut specials = HashMap::new();
        for (text, pieces) in saved.exceptions {
            let mut joined = String::new();
            let mut kept = Vec::new();
            for p in pieces {
                let orth = p.get(&ORTH).cloned().unwrap_or_default();
                joined.push_str(&orth);
                let norm = p.get(&NORM).cloned();
                kept.push(Piece { orth, norm });
            }
            if joined != text {
                return Err(format!("special case {text:?} joins to {joined:?}").into());
            }
            specials.insert(text, kept);
        }
        let mut t = Self {
            prefix: compile(&saved.prefix_search)?,
            suffix: compile(&saved.suffix_search)?,
            infix: compile(&saved.infix_finditer)?,
            token_match: compile(&saved.token_match)?,
            url_match: compile(&saved.url_match)?,
            specials,
            phrases: Phrases::default(),
        };
        t.phrases = Phrases::new(&t, saved.faster_heuristics)?;
        Ok(t)
    }

    /// spaCy's tokens for `text`, with norms and space flags.
    pub fn words(&self, text: &str) -> Result<Vec<Word>, Error> {
        let mut words = self.split(text, true)?;
        self.phrases.apply(self, text, &mut words);
        Ok(words)
    }

    /// Space and text runs. A token owns one following space.
    fn split(&self, text: &str, specials: bool) -> Result<Vec<Word>, Error> {
        let mut out = Vec::new();
        let Some(first) = text.chars().next() else {
            return Ok(out);
        };
        let (mut start, mut in_space) = (0, is_space(first));
        for (i, c) in text.char_indices() {
            if is_space(c) == in_space {
                continue;
            }
            if start < i {
                self.chunk(text, start..i, specials, &mut out)?;
            }
            if c == ' ' {
                if let Some(last) = out.last_mut() {
                    last.space_after = true;
                }
                start = i + 1;
            } else {
                start = i;
            }
            in_space = !in_space;
        }
        if start < text.len() {
            self.chunk(text, start..text.len(), specials, &mut out)?;
            if let Some(last) = out.last_mut() {
                last.space_after = text.ends_with(' ') && !in_space;
            }
        }
        Ok(out)
    }

    fn chunk(
        &self,
        text: &str,
        span: Range<usize>,
        specials: bool,
        out: &mut Vec<Word>,
    ) -> Result<(), Error> {
        if specials && self.special(text, span.clone(), out) {
            return Ok(());
        }
        let (prefixes, rest, suffixes) = self.affixes(text, span, specials)?;
        out.extend(prefixes.into_iter().map(plain));
        if !rest.is_empty() {
            self.infixes(text, rest, specials, out)?;
        }
        out.extend(suffixes.into_iter().rev().map(plain));
        Ok(())
    }

    /// If `span` holds a special case, pushes its pieces.
    fn special(&self, text: &str, span: Range<usize>, out: &mut Vec<Word>) -> bool {
        let Some(pieces) = self.specials.get(&text[span.clone()]) else {
            return false;
        };
        let mut at = span.start;
        for p in pieces {
            out.push(Word {
                span: at..at + p.orth.len(),
                norm: p.norm.clone(),
                space_after: false,
            });
            at += p.orth.len();
        }
        true
    }

    /// Strips affixes until none match or a special case remains.
    fn affixes(&self, text: &str, span: Range<usize>, specials: bool) -> Result<Affixes, Error> {
        let is_special = |r: Range<usize>| specials && self.specials.contains_key(&text[r]);
        let (mut a, mut b) = (span.start, span.end);
        let (mut pre, mut suf) = (Vec::new(), Vec::new());
        let mut last = 0;
        while a < b && b - a != last {
            if matches_start(&self.token_match, &text[a..b])? || is_special(a..b) {
                break;
            }
            last = b - a;
            let p = match_len(&self.prefix, &text[a..b])?;
            if p != 0 && a + p < b && is_special(a + p..b) {
                pre.push(a..a + p);
                a += p;
                break;
            }
            let s = match_len(&self.suffix, &text[a + p..b])?;
            if s != 0 && a + s < b && is_special(a..b - s) {
                suf.push(b - s..b);
                b -= s;
                break;
            }
            if p != 0 && s != 0 && p + s <= b - a {
                pre.push(a..a + p);
                suf.push(b - s..b);
                (a, b) = (a + p, b - s);
            } else if p != 0 {
                pre.push(a..a + p);
                a += p;
            } else if s != 0 {
                suf.push(b - s..b);
                b -= s;
            }
        }
        Ok((pre, a..b, suf))
    }

    /// Splits the rest at each infix match.
    fn infixes(
        &self,
        text: &str,
        span: Range<usize>,
        specials: bool,
        out: &mut Vec<Word>,
    ) -> Result<(), Error> {
        if specials && self.special(text, span.clone(), out) {
            return Ok(());
        }
        let s = &text[span.clone()];
        if matches_start(&self.token_match, s)? || matches_start(&self.url_match, s)? {
            out.push(plain(span));
            return Ok(());
        }
        let (base, mut start) = (span.start, 0);
        if let Some(infix) = &self.infix {
            for m in infix.find_iter(s) {
                let m = m?;
                // spaCy skips a match at the very start.
                if m.start() == 0 {
                    continue;
                }
                if m.start() != start {
                    out.push(plain(base + start..base + m.start()));
                }
                if m.start() != m.end() {
                    out.push(plain(base + m.start()..base + m.end()));
                }
                start = m.end();
            }
        }
        if start < s.len() {
            out.push(plain(base + start..span.end));
        }
        Ok(())
    }
}

impl Tokenizer for Spacy {
    fn tokenize(&self, text: &str) -> Result<Vec<Token>, Error> {
        let words = self.words(text)?;
        Ok(words
            .into_iter()
            .map(|w| Token {
                span: w.span,
                id: None,
            })
            .collect())
    }
}
