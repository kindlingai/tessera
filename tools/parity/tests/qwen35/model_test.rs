//! The whole model, and its config.json from Hugging Face.

use super::{LEN, SEQS, check, part, positions, tiny};
use tessera_bundle::Overrides;
use tessera_interp::Tensor;
use tessera_models::qwen35::{Config, Mixing, program};

#[test]
fn logits_match_hf() {
    let Some(dir) = part("model") else {
        return eprintln!("qwen35-tiny: no reference here; skipped");
    };
    let bytes = std::fs::read(dir.join("tokens.u32")).unwrap();
    let ids = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b));
    let tokens = Tensor::u32(vec![SEQS, LEN], ids.collect());
    check(&program(&tiny()), &dir, &[tokens, positions()], "logits");
}

/// `config.json` in the fixture's `name` folder, read.
fn read(name: &str) -> Option<Config> {
    let path = part(name)?.join("config.json");
    let none = Overrides::default();
    Some(
        none.read("config.json", Some(&path), |k| {
            Config::read(k, SEQS as u64, 8)
        })
        .unwrap(),
    )
}

#[test]
fn configs_read() {
    let (Some(small), Some(clef)) = (read("model"), read("clef")) else {
        return eprintln!("qwen35-tiny: no reference here; skipped");
    };
    assert_eq!(program(&small).to_json(), program(&tiny()).to_json());
    // Clef nests its text model's keys under text_config.
    let s = &clef.stack;
    let sizes = [s.vocab, s.hidden, s.layers, s.heads, s.head_dim, s.ffn];
    assert_eq!(sizes, [248_320, 5120, 64, 24, 256, 17_408]);
    assert_eq!((clef.kv_heads, clef.rotary, clef.tied), (4, 64, false));
    let l = &clef.linear;
    let linear = [l.key_heads, l.value_heads, l.key_dim, l.value_dim, l.window];
    assert_eq!(linear, [16, 48, 128, 128, 4]);
    let full = |i: usize| clef.layers[i] == Mixing::Full;
    assert!((0..64).all(|i| full(i) == (i % 4 == 3)));
}
