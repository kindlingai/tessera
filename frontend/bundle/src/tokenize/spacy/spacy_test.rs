// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

use super::{Spacy, Word};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
struct Saved {
    prefix_search: Option<&'static str>,
    suffix_search: Option<&'static str>,
    infix_finditer: Option<&'static str>,
    token_match: Option<&'static str>,
    url_match: Option<&'static str>,
    exceptions: HashMap<&'static str, Vec<HashMap<u64, &'static str>>>,
    faster_heuristics: bool,
}

/// Brackets, stops, hyphens and two special cases.
fn small() -> Spacy {
    let piece = |orth, norm: Option<&'static str>| {
        let mut p = HashMap::from([(super::ORTH, orth)]);
        p.extend(norm.map(|n| (super::NORM, n)));
        p
    };
    let saved = Saved {
        prefix_search: Some(r"^\("),
        suffix_search: Some(r"\)$|\.$"),
        infix_finditer: Some(r"(?<=[a-z])-(?=[a-z])"),
        token_match: None,
        url_match: None,
        exceptions: HashMap::from([
            ("don't", vec![piece("do", None), piece("n't", Some("not"))]),
            ("e.g.", vec![piece("e.g.", None)]),
        ]),
        faster_heuristics: true,
    };
    Spacy::from_saved(&rmp_serde::to_vec_named(&saved).unwrap()).unwrap()
}

fn texts<'a>(text: &'a str, words: &[Word]) -> Vec<&'a str> {
    words.iter().map(|w| &text[w.span.clone()]).collect()
}

#[test]
fn rules_split() {
    let text = "don't  go (e.g. home-made).";
    let words = small().words(text).unwrap();
    let want = [
        "do", "n't", " ", "go", "(", "e.g.", "home", "-", "made", ")", ".",
    ];
    assert_eq!(texts(text, &words), want);
    assert_eq!(words[1].norm.as_deref(), Some("not"));
    let spaced: Vec<bool> = words.iter().map(|w| w.space_after).collect();
    assert_eq!(
        spaced,
        [
            false, true, false, true, false, true, false, false, false, false, false
        ]
    );
}

/// Each case: the text, then spaCy's tokens as `[text, space_after]`.
type Case = (String, Vec<(String, bool)>);

/// Matches spaCy on the texts in `$TESSERA_SPACY/tokens.json`.
#[test]
fn spacy_parity() {
    let Some(dir) = settings::read().spacy else {
        eprintln!("TESSERA_SPACY unset: skipped");
        return;
    };
    let t = Spacy::open(&dir).unwrap();
    let cases: Vec<Case> =
        serde_json::from_slice(&std::fs::read(dir.join("tokens.json")).unwrap()).unwrap();
    let mut wrong = 0;
    for (text, want) in &cases {
        let words = t.words(text).unwrap();
        let got: Vec<(String, bool)> = words
            .iter()
            .map(|w| (text[w.span.clone()].into(), w.space_after))
            .collect();
        if got != *want {
            wrong += 1;
            eprintln!("{text:?}\n  got  {got:?}\n  want {want:?}");
        }
    }
    assert_eq!(wrong, 0, "{wrong} of {} texts differ", cases.len());
}
