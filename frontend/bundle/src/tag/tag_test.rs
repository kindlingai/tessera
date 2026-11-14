// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

use super::Tagger;
use super::features::{Features, hash_string, shape, thinc_hash};
use crate::tokenize::spacy::Word;
use std::collections::HashMap;

/// spaCy 3.8 and Thinc 8 give these values.
#[test]
fn hashes_match_spacy() {
    let loader = hash_string("loader");
    assert_eq!(loader, 9_424_920_502_403_081_132);
    assert_eq!(
        thinc_hash(loader, 8),
        [3_554_004_892, 3_057_108_156, 1_036_040_042, 2_438_178_587]
    );
    let f = Features::new(HashMap::new());
    assert_eq!((f.id("X"), f.id("Xxxx")), (101, 10_887_629_174_180_191_697));
    assert_eq!(
        (shape("C3PO"), shape("aaaaaa")),
        ("XdXX".into(), "xxxx".into())
    );
    let word = Word {
        span: 0..3,
        norm: None,
        space_after: false,
    };
    assert_eq!(f.norm("…", &word), "...");
}

/// Each case: the text, then spaCy's tokens as `[text, tag]`.
type Case = (String, Vec<(String, String)>);

/// Matches spaCy's tags on the texts in `$TESSERA_SPACY/tags.json`.
#[test]
fn spacy_tags() {
    let Some(dir) = settings::read().spacy else {
        eprintln!("TESSERA_SPACY unset: skipped");
        return;
    };
    let tagger = Tagger::open(&dir).unwrap();
    let cases: Vec<Case> =
        serde_json::from_slice(&std::fs::read(dir.join("tags.json")).unwrap()).unwrap();
    let (mut wrong, mut tokens) = (0, 0);
    for (text, want) in &cases {
        let got: Vec<(String, String)> = tagger
            .tag(text)
            .unwrap()
            .into_iter()
            .map(|(w, t)| (text[w.span].into(), t.into()))
            .collect();
        tokens += want.len();
        let misses =
            got.iter().zip(want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len());
        if misses > 0 {
            wrong += misses;
            eprintln!("{text:?}\n  got  {got:?}\n  want {want:?}");
        }
    }
    assert_eq!(wrong, 0, "{wrong} of {tokens} tokens differ");
}
