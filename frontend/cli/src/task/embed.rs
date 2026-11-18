//! `task embed`: one JSON vector per line of text.

use super::{load, text};
use crate::source::{Source, Stop};
use serde_json::json;

pub fn run(source: Source, words: &[String]) -> Result<(), Stop> {
    let input = text(words)?;
    let mut texts = vec![input.trim().to_string()];
    if words.is_empty() {
        texts = input.lines().map(str::trim).map(String::from).collect();
    }
    texts.retain(|t| !t.is_empty());
    if texts.is_empty() {
        return Err(Stop::Usage);
    }
    let mut options = source.options(&["--task".into(), "embed".into()])?;
    options.slots = 1;
    let slot = load(options);
    let embed = slot.get().and_then(|m| m.embed.as_ref());
    let embed = embed.ok_or_else(|| Stop::Failed("the model does not embed text".into()))?;
    let got = embed
        .embed(&texts)
        .map_err(|e| Stop::Failed(e.to_string()))?;
    for (index, embedding) in got.vectors.iter().enumerate() {
        println!("{}", json!({"index": index, "embedding": embedding}));
    }
    Ok(())
}
