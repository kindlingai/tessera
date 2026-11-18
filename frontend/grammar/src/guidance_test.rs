use crate::Vocab;
use sensai_bundle::tokenize::hf::Hf;
use sensai_engine::{Allowed, Guide};
use serde_json::json;

const EOS: u32 = 256;
const ROWS: usize = 300;

/// GPT-2's printable stand-in for each byte.
fn byte_char(b: u8) -> char {
    let printable = |b: u8| matches!(b, 33..=126 | 161..=172 | 174..=255);
    if printable(b) {
        return b as char;
    }
    let skipped = (0..b).filter(|&c| !printable(c)).count() as u32;
    char::from_u32(256 + skipped).unwrap_or_default()
}

/// A byte-level BPE file: one token per byte, then `<eos>`.
fn tokenizer() -> Vec<u8> {
    let vocab: serde_json::Map<_, _> = (0..=255u8)
        .map(|b| (byte_char(b).to_string(), json!(b)))
        .collect();
    let level = json!({ "type": "ByteLevel", "add_prefix_space": false,
        "trim_offsets": true, "use_regex": true });
    let eos = json!({ "id": EOS, "content": "<eos>", "single_word": false,
        "lstrip": false, "rstrip": false, "normalized": false, "special": true });
    let file = json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [eos], "normalizer": null, "post_processor": null,
        "pre_tokenizer": level, "decoder": level,
        "model": { "type": "BPE", "vocab": vocab, "merges": [] },
    });
    serde_json::to_vec(&file).unwrap_or_default()
}

fn allowed(mask: &[u32], id: u32) -> bool {
    mask[id as usize / 32] >> (id % 32) & 1 == 1
}

#[test]
fn schema_steps() {
    let vocab = Vocab::new(&tokenizer(), &[EOS], ROWS).unwrap();
    let schema = json!({ "type": "object", "properties": { "a": { "type": "integer" } },
        "required": ["a"], "additionalProperties": false });
    let mut m = vocab.schema(schema).unwrap();
    let mut mask = vec![0; vocab.words()];
    m.mask(&mut mask).unwrap();
    assert!(!allowed(&mask, b'x' as u32));
    for b in br#"{"a":12}"# {
        m.mask(&mut mask).unwrap();
        assert!(allowed(&mask, *b as u32), "{}", *b as char);
        assert!(!allowed(&mask, EOS));
        m.accept(*b as u32).unwrap();
    }
    m.mask(&mut mask).unwrap();
    assert!(m.accepting() && allowed(&mask, EOS));
    assert!((EOS + 1..ROWS as u32).all(|id| !allowed(&mask, id)));
}

#[test]
fn sets_pick_tiers() {
    let json = tokenizer();
    let digits: Vec<u32> = (b'0'..=b'9').map(u32::from).collect();
    assert_eq!(Hf::from_json(&json).unwrap().digits(), digits);
    let vocab = Vocab::new(&json, &[EOS], ROWS).unwrap();
    let mut m = vocab.regex("[0-9]+").unwrap();
    assert_eq!(m.allowed().unwrap(), Allowed::Few(digits));
}

#[test]
fn split_characters_kept() {
    let vocab = Vocab::new(&tokenizer(), &[EOS], ROWS).unwrap();
    // The first byte of `é` stays itself, not U+FFFD.
    assert_eq!(vocab.tokenize(&[b'a', 0xC3]), [u32::from(b'a'), 0xC3]);
}
