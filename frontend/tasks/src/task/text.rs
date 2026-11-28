//! `embed` and `tag`: one JSON record per line of text.

use super::{Run, Sink, Task};
use crate::Error;
use crate::Tagger;
use crate::api::{EmbedInput, EmbedMessage, EmbedOptions, EmbedPart, Model};
use serde_json::json;
use std::path::PathBuf;

/// Print an embedding vector for some text or image.
#[derive(Task)]
#[task(kind = "embed", example = "embed the text 'tea and toast'")]
pub struct Embed {
    /// The text. Each line gets its own vector.
    #[arg(positional, min = 1, max = 500, example = "tea and toast")]
    pub text: String,
    /// Optional Matryoshka output dimension count.
    #[arg(value = "<d>", min = 1)]
    pub dimensions: Option<usize>,
    /// Optional image file path for multimodal embedding.
    #[arg(value = "<path>", cli_only)]
    pub image: Option<PathBuf>,
}

impl Run<&Model> for Embed {
    fn run(self, model: &Model, sink: &mut dyn Sink) -> Result<(), Error> {
        let embed = model.embed.as_ref();
        let embed = embed.ok_or("the model does not embed text")?;
        let options = EmbedOptions {
            dimensions: self.dimensions,
            normalize: true,
        };
        let got = if let Some(path) = self.image {
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut parts = vec![EmbedPart::Image(bytes)];
            if !self.text.trim().is_empty() {
                parts.push(EmbedPart::Text(self.text.trim().to_string()));
            }
            let item = EmbedInput::Messages {
                messages: vec![EmbedMessage {
                    role: "user".into(),
                    parts,
                }],
                add_generation_prompt: true,
            };
            embed.embed_inputs(&[item], options)?
        } else {
            let mut inputs = Vec::new();
            for line in self.text.lines() {
                if !line.trim().is_empty() {
                    inputs.push(EmbedInput::Text(line.trim().to_string()));
                }
            }
            embed.embed_inputs(&inputs, options)?
        };
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
