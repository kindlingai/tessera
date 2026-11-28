//! Vectors for `/v1/embeddings`.

use tessera_tasks::Error;
use tessera_tasks::api::{Embed, EmbedInput, EmbedOptions, EmbedPart, Embedded, finish_vec};

/// One vector per text: its length in bytes.
pub(super) struct MockEmbed;

impl Embed for MockEmbed {
    fn embed(&self, texts: &[String]) -> Result<Embedded, Error> {
        Ok(Embedded {
            vectors: texts.iter().map(|t| vec![t.len() as f32]).collect(),
            tokens: texts.len(),
        })
    }

    fn embed_inputs(
        &self,
        inputs: &[EmbedInput],
        options: EmbedOptions,
    ) -> Result<Embedded, Error> {
        let mut vectors = Vec::with_capacity(inputs.len());
        let mut tokens = 0usize;
        for item in inputs {
            let v = match item {
                EmbedInput::Text(t) => {
                    tokens += 1;
                    if options.dimensions.is_some() {
                        vec![t.len() as f32, 1.0, 2.0, 3.0]
                    } else {
                        vec![t.len() as f32]
                    }
                }
                EmbedInput::Tokens(ids) => {
                    tokens += ids.len();
                    vec![ids.len() as f32, 1.0, 2.0, 3.0]
                }
                EmbedInput::Messages { messages, .. } => {
                    let mut n = 0usize;
                    for m in messages {
                        for p in &m.parts {
                            n += match p {
                                EmbedPart::Text(t) => t.len(),
                                EmbedPart::Image(b) => b.len(),
                            };
                        }
                    }
                    tokens += 1;
                    vec![n as f32, 1.0, 2.0, 3.0]
                }
            };
            if options.dimensions.is_some() || !options.normalize {
                vectors.push(finish_vec(v, options.dimensions, options.normalize)?);
            } else {
                vectors.push(v);
            }
        }
        Ok(Embedded { vectors, tokens })
    }
}
