//! Answers for `/v1/completions` and `/v1/chat/completions`.

use sensai_tasks::Error;
use sensai_tasks::api::{Finish, Format, Generator, Params, Refused, Sink, Step};
use std::sync::Mutex;

/// Answers "hello" in two tokens. Keeps what it was asked.
#[derive(Default)]
pub struct MockGenerator {
    /// The params of the last answer.
    pub asked: Mutex<Option<Params>>,
}

impl Generator for MockGenerator {
    fn generate(&self, _prompt: Vec<u32>, params: Params, mut sink: Sink) -> Result<(), Error> {
        // Schemas that are not objects are refused, as llguidance does.
        if let Some(Format::Schema(schema)) = &params.format
            && !schema.is_object()
        {
            return Err(Box::new(Refused("a schema is an object".into())));
        }
        *self.asked.lock().unwrap() = Some(params);
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
        });
        Ok(())
    }
}
