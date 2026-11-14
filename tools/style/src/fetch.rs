//! Fetches spaCy's small English pipeline from Hugging Face.

use std::io::Read;
use std::path::Path;

const REPO: &str = "https://huggingface.co/spacy/en_core_web_sm/resolve";
/// en_core_web_sm 3.7.1. The parity tests check this revision.
pub const REVISION: &str = "8fb6f8155286360bf5486fe399e2738d1df7ded2";
/// Files for the tokenizer and the tagger.
const FILES: [&str; 5] = [
    "tokenizer",
    "tok2vec/model",
    "tagger/model",
    "tagger/cfg",
    "vocab/lookups.bin",
];

pub fn fetch(dir: &Path) -> Result<(), String> {
    for name in FILES {
        let url = format!("{REPO}/{REVISION}/{name}");
        let mut bytes = Vec::new();
        let response = ureq::get(&url).call().map_err(|e| format!("{url}: {e}"))?;
        response
            .into_reader()
            .read_to_end(&mut bytes)
            .map_err(|e| format!("{url}: {e}"))?;
        let path = dir.join(name);
        let parent = path.parent().unwrap_or(dir);
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        std::fs::write(&path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        eprintln!("{name}: {} bytes", bytes.len());
    }
    Ok(())
}
