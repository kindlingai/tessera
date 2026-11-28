// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

use super::features::{Features, hash_string, thinc_hash, with_shape};
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
    let mut lex = HashMap::new();
    lex.insert(hash_string("colour"), "color".into());
    let f = Features::new(lex);
    assert_eq!((f.id("X"), f.id("Xxxx")), (101, 10_887_629_174_180_191_697));
    assert_eq!(
        (
            with_shape("C3PO", str::to_owned),
            with_shape("aaaaaa", str::to_owned),
            with_shape(&"a".repeat(100), str::to_owned),
        ),
        ("XdXX".into(), "xxxx".into(), "LONG".into())
    );
    let word = Word {
        span: 0..3,
        norm: None,
        space_after: false,
    };
    assert_eq!(f.with_norm("…", &word, str::to_owned), "...");
    assert_eq!(f.with_norm("Colour", &word, str::to_owned), "color");
    let got = f.of("café", &word);
    assert_eq!(
        got,
        [f.id("café"), f.id("c"), f.id("afé"), f.id("xxxx"), 0, 0]
    );
    assert_eq!(f.of("  ", &word)[5], 1);
}
