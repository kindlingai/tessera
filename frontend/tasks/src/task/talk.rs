//! `ask` and `chat`: answers from a text model.

use super::{Run, Sink, Task, Usage};
use crate::Error;
use crate::api::{Format, Model, Params, Refused, Step};
use std::sync::mpsc;
use tessera_bundle::Message;

impl Model {
    /// The messages as one prompt, by the model's template.
    pub fn render(&self, messages: &[Message]) -> Result<String, Error> {
        let none = || Refused("the model has no chat template".into(), "messages");
        self.chat.as_ref().ok_or_else(none)?.render(messages, false)
    }

    /// The answer to `prompt`, each piece passed to `each`.
    pub fn answer(
        &self,
        prompt: &str,
        max_tokens: Option<usize>,
        format: Option<Format>,
        each: &mut dyn FnMut(&str),
    ) -> Result<String, Error> {
        let none = || Refused("the model does not generate text".into(), "model");
        let generator = self.generator.as_ref().ok_or_else(none)?;
        let ids = self.tokens.encode(prompt)?;
        let room = self.info.context.saturating_sub(ids.len());
        if ids.is_empty() || room == 0 {
            return Err("the prompt does not fit the context".into());
        }
        let params = Params {
            max_tokens: max_tokens.map_or(room, |n| n.min(room)),
            format,
            ..Params::default()
        };
        let (sink, steps) = mpsc::channel();
        generator.generate(ids, params, Box::new(move |step| sink.send(step).is_ok()))?;
        let mut answer = String::new();
        while let Ok(step) = steps.recv() {
            let (text, done) = match step {
                Step::Token { text, .. } | Step::Text { text, .. } => (text, false),
                Step::Done { text, .. } => (text, true),
            };
            each(&text);
            answer += &text;
            if done {
                return Ok(answer);
            }
        }
        Err("the engine stopped".into())
    }
}

/// Answer one prompt, from stdin if none.
#[derive(Task)]
#[task(kind = "generate")]
pub struct Ask {
    /// The most tokens in the answer.
    #[arg(value = "<n>")]
    pub max_tokens: Option<usize>,
    /// The prompt to answer.
    #[arg(positional)]
    pub prompt: String,
}

impl Run<&Model> for Ask {
    fn run(self, model: &Model, sink: &mut dyn Sink) -> Result<(), Error> {
        let prompt = self.prompt.trim();
        if prompt.is_empty() {
            return Err(Usage("ask needs a prompt".into()).into());
        }
        let text = model.render(&[Message::text("user", prompt)]);
        let text = text.unwrap_or_else(|_| prompt.into());
        model.answer(&text, self.max_tokens, None, &mut |t| sink.text(t))?;
        Ok(())
    }
}

/// Talk with the model, one line a turn.
#[derive(Task)]
#[task(kind = "generate")]
pub struct Chat {
    /// The most tokens in each answer.
    #[arg(value = "<n>")]
    pub max_tokens: Option<usize>,
}

impl Run<&Model> for Chat {
    fn run(self, model: &Model, sink: &mut dyn Sink) -> Result<(), Error> {
        let mut history = Vec::new();
        while let Some(line) = sink.turn() {
            if line.trim().is_empty() {
                break;
            }
            history.push(Message::text("user", line.trim()));
            let mut each = |t: &str| sink.text(t);
            let answer = model.render(&history);
            let answer =
                answer.and_then(|text| model.answer(&text, self.max_tokens, None, &mut each));
            match answer {
                Ok(text) => history.push(Message::text("assistant", text.trim())),
                Err(e) => {
                    history.pop();
                    sink.note(&e.to_string());
                }
            }
        }
        Ok(())
    }
}
