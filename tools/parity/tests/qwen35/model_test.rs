//! The whole model, and its config.json from Hugging Face.

use super::{LEN, SEQS, check, part, positions, tiny};
use tessera_bundle::Overrides;
use tessera_interp::Tensor;
use tessera_models::qwen35::{Config, program};

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
            Config::read(k, SEQS as u64, 16)
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
    assert_eq!(
        program(&clef).to_json(),
        program(&super::clef(SEQS as u64, 16)).to_json()
    );
}
