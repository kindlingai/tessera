//! `task tag`: one JSON line per token and its tag.

use super::text;
use crate::source::{Source, Stop};
use sensai_tasks::Tagger;
use serde_json::json;

pub fn run(source: Source, words: &[String]) -> Result<(), Stop> {
    let input = text(words)?;
    let options = source.options(&[])?;
    let failed = |e: sensai_tasks::Error| Stop::Failed(e.to_string());
    let mut tagger =
        Tagger::open(&options.model, &sensai_frontend_settings::read()).map_err(failed)?;
    for (line, text) in input.lines().enumerate() {
        for (word, tag) in tagger.tag(text).map_err(failed)? {
            let token = &text[word.span];
            println!("{}", json!({"line": line, "token": token, "tag": tag}));
        }
    }
    Ok(())
}
