//! Answers for `/v1/completions` and `/v1/chat/completions`.

use std::sync::Mutex;
use tessera_tasks::Error;
use tessera_tasks::api::{Finish, Format, Generator, Params, Read, Refused, Score, Sink, Step};

/// Answers "hello" in two tokens, or custom `pieces`.
#[derive(Default)]
pub struct MockGenerator {
    /// The params of the last answer.
    pub asked: Mutex<Option<Params>>,
    /// Custom output chunks, with the last sent in `Done`.
    pub pieces: Vec<String>,
}

impl Generator for MockGenerator {
    fn generate(&self, _prompt: Vec<u32>, params: Params, mut sink: Sink) -> Result<(), Error> {
        // Schemas that are not objects are refused, as llguidance does.
        if let Some(Format::Schema(schema)) = &params.format
            && !schema.is_object()
        {
            return Err(Box::new(Refused(
                "a schema is an object".into(),
                "response_format",
            )));
        }
        *self.asked.lock().unwrap() = Some(params);
        if let Some((last, head)) = self.pieces.split_last() {
            for (i, piece) in head.iter().enumerate() {
                sink(Step::Token {
                    id: i as u32 + 1,
                    text: piece.clone(),
                });
            }
            sink(Step::Done {
                text: last.clone(),
                finish: Finish::Stop,
                tokens: self.pieces.len(),
                cached: 0,
            });
            return Ok(());
        }
        sink(Step::Token {
            id: 1,
            text: "he".into(),
        });
        sink(Step::Token {
            id: 2,
            text: String::new(),
        });
        sink(Step::Done {
            text: "llo".into(),
            finish: Finish::Stop,
            tokens: 2,
            cached: 3,
        });
        Ok(())
    }
}

impl Score for MockGenerator {
    /// Each read's logits favor its last id.
    fn score(&self, _prompt: Vec<u32>, reads: Vec<Read>) -> Result<Vec<Vec<f32>>, Error> {
        let rising = |r: &Read| (0..r.ids.len()).map(|i| i as f32).collect();
        Ok(reads.iter().map(rising).collect())
    }
}
