//! The whole model, and its config.json from Hugging Face.

use super::{SEQS, check, part, positions, tiny, tokens};
use tessera_bundle::Overrides;
use tessera_bundle::qwen35::config;
use tessera_models::qwen35::{Config, program};

#[test]
fn logits_match_hf() {
    let Some(dir) = part("model") else {
        return eprintln!("qwen35-tiny: no reference here; skipped");
    };
    check(
        &program(&tiny()),
        &dir,
        &[tokens(&dir), positions()],
        "logits",
    );
}

/// `config.json` in the fixture's `name` folder, read.
fn read(name: &str) -> Option<Config> {
    let sizes = [SEQS as u64, 16];
    Some(config(&part(name)?, sizes, &Overrides::default()).unwrap())
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
