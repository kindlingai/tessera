//! `embed` and `tag`: one JSON record per line of text.

use super::{Run, Sink, Task};
use crate::Error;
use crate::Tagger;
use crate::api::Model;
use serde_json::json;

/// Print an embedding vector for some text.
#[derive(Task)]
#[task(kind = "embed", example = "embed the text 'tea and toast'")]
pub struct Embed {
    /// The text. Each line gets its own vector.
    #[arg(positional, min = 1, max = 500, example = "tea and toast")]
    pub text: String,
}

impl Run<&Model> for Embed {
    fn run(self, model: &Model, sink: &mut dyn Sink) -> Result<(), Error> {
        let mut texts = Vec::new();
        for line in self.text.lines() {
            if !line.trim().is_empty() {
                texts.push(line.trim().to_string());
            }
        }
        let embed = model.embed.as_ref();
        let embed = embed.ok_or("the model does not embed text")?;
        let got = embed.embed(&texts)?;
        for (index, embedding) in got.vectors.iter().enumerate() {
            sink.record(json!({"index": index, "embedding": embedding}));
        }
        Ok(())
    }
}

/// Print part-of-speech tags for some text.
#[derive(Task)]
#[task(kind = "tag", example = "tag the words in 'dogs bark'")]
pub struct Tag {
    /// The text. Tags name the line they come from.
    #[arg(positional, min = 1, max = 500, example = "dogs bark")]
    pub text: String,
}

impl Run<&mut Tagger> for Tag {
    fn run(self, tagger: &mut Tagger, sink: &mut dyn Sink) -> Result<(), Error> {
        for (line, text) in self.text.lines().enumerate() {
            for (word, tag) in tagger.tag(text)? {
                let token = &text[word.span];
                sink.record(json!({"line": line, "token": token, "tag": tag}));
            }
        }
        Ok(())
    }
}
