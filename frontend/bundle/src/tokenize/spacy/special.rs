// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

//! The second pass: special cases that span several first-pass tokens.

use super::{Error, Spacy, Word, match_len};
use std::collections::{HashMap, HashSet};
use std::ops::Range;

/// Special cases as token texts, keyed by their first token.
#[derive(Default)]
pub(super) struct Phrases(HashMap<String, Vec<Vec<String>>>);

impl Phrases {
    /// Special cases the first pass splits. `faster` skips plain ones.
    pub(super) fn new(t: &Spacy, faster: bool) -> Result<Self, Error> {
        let mut by_first: HashMap<String, Vec<Vec<String>>> = HashMap::new();
        for text in t.specials.keys() {
            let has_infix = match &t.infix {
                Some(re) => re.find_iter(text).next().transpose()?.is_some(),
                None => false,
            };
            let affixed = match_len(&t.prefix, text)? != 0 || match_len(&t.suffix, text)? != 0;
            if faster && !affixed && !has_infix && !text.contains(' ') {
                continue;
            }
            let words = t.split(text, false)?;
            let orths: Vec<String> = words
                .iter()
                .map(|w| text[w.span.clone()].to_string())
                .collect();
            if let Some(first) = orths.first() {
                by_first.entry(first.clone()).or_default().push(orths);
            }
        }
        Ok(Self(by_first))
    }

    /// Token ranges whose texts match a special case.
    fn matches(&self, text: &str, words: &[Word]) -> Vec<Range<usize>> {
        let mut found = Vec::new();
        for (i, w) in words.iter().enumerate() {
            let Some(phrases) = self.0.get(&text[w.span.clone()]) else {
                continue;
            };
            for phrase in phrases {
                let end = i + phrase.len();
                if end > words.len() {
                    continue;
                }
                let same = phrase
                    .iter()
                    .zip(&words[i..end])
                    .all(|(p, w)| *p == text[w.span.clone()]);
                if same {
                    found.push(i..end);
                }
            }
        }
        found
    }

    /// Replaces matched tokens with the special case's pieces.
    pub(super) fn apply(&self, t: &Spacy, text: &str, words: &mut Vec<Word>) {
        let mut found = self.matches(text, words);
        // Longest first, then earliest. Taken ends drop a match.
        found.sort_by(|a, b| b.len().cmp(&a.len()).then(a.start.cmp(&b.start)));
        let mut seen = HashSet::new();
        let mut kept = Vec::new();
        for f in found {
            if !seen.contains(&f.start) && !seen.contains(&(f.end - 1)) {
                kept.push(f.clone());
            }
            seen.extend(f);
        }
        kept.sort_by_key(|f| f.start);
        for f in kept.into_iter().rev() {
            let span = words[f.start].span.start..words[f.end - 1].span.end;
            let space_after = words[f.end - 1].space_after;
            let mut pieces = Vec::new();
            if !t.special(text, span, &mut pieces) {
                continue;
            }
            if let Some(last) = pieces.last_mut() {
                last.space_after = space_after;
            }
            words.splice(f, pieces);
        }
    }
}
