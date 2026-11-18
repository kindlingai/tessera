//! `task <name>`: one job on the loaded model.

mod ask;
mod chat;
mod embed;
mod image;
#[cfg(test)]
mod image_test;
mod tag;

use crate::source::{Origin, Source, Stop};
use sensai_bundle::Message;
use sensai_serve::start::{self, Options};
use sensai_serve::{Model, Slot};
use sensai_tasks::api::{Params, Step};
use std::io::{Read, Write};
use std::sync::mpsc;
use std::time::Duration;

pub const LIST: &str = "tasks:
  ask [--max-tokens <n>] [prompt...]  answer one prompt, from stdin if none
  chat [--max-tokens <n>]             talk with the model, one line a turn
  image [--n <k>] [--seed <s>] [--image-size <px>] [--task paint|klein] [--clip <dir>] [-o <path>] [prompt...]
                                      paint PNGs, from stdin if no prompt
  embed [text]                        one JSON vector per line of text, from stdin if none
  tag [text]                          one JSON line per token and its tag";

pub fn run(origin: Option<Origin>, rest: &[String]) -> Result<(), Stop> {
    let Some((name, args)) = rest.split_first() else {
        return Err(Stop::Tasks(None));
    };
    match name.as_str() {
        "image" => return image::run(Source::open(origin)?, args),
        "embed" => return embed::run(Source::open(origin)?, args),
        "tag" => return tag::run(Source::open(origin)?, args),
        _ => {}
    }
    let task = match name.as_str() {
        "ask" => ask::run,
        "chat" => chat::run,
        other => return Err(Stop::Tasks(Some(other.into()))),
    };
    let source = Source::open(origin)?;
    let (max_tokens, args) = flags(args)?;
    let mut options = source.options(&[])?;
    options.slots = 1;
    task(Talk::open(options, max_tokens), args)
}

/// Takes `--max-tokens <n>` from the front.
fn flags(args: &[String]) -> Result<(Option<usize>, &[String]), Stop> {
    match args {
        [flag, n, rest @ ..] if flag == "--max-tokens" => {
            let n = n.parse().map_err(|_| Stop::Usage)?;
            Ok((Some(n), rest))
        }
        [dashes, rest @ ..] if dashes == "--" => Ok((None, rest)),
        [flag, ..] if flag.starts_with("--") => Err(Stop::Usage),
        _ => Ok((None, args)),
    }
}

/// Starts the load and waits for the model.
fn load(options: Options) -> Slot {
    let slot = start::spawn(options);
    while slot.get().is_none() {
        std::thread::sleep(Duration::from_millis(20));
    }
    slot
}

/// The prompt from `words`, or from stdin if none.
fn text(words: &[String]) -> Result<String, Stop> {
    let mut text = words.join(" ");
    if words.is_empty() {
        std::io::stdin()
            .read_to_string(&mut text)
            .map_err(|e| Stop::Failed(e.to_string()))?;
    }
    Ok(text)
}

/// A loaded model that answers to stdout.
pub struct Talk {
    slot: Slot,
    max_tokens: Option<usize>,
}

impl Talk {
    /// Loads the model and waits for it.
    fn open(options: Options, max_tokens: Option<usize>) -> Talk {
        let slot = load(options);
        Talk { slot, max_tokens }
    }

    fn model(&self) -> &Model {
        self.slot.get().expect("the model loaded")
    }

    /// The messages as one prompt, by the model's template.
    fn render(&self, messages: &[Message]) -> Result<String, Stop> {
        let chat = self.model().chat.as_ref();
        let chat = chat.ok_or_else(|| Stop::Failed("the model has no chat template".into()))?;
        chat.render(messages, false)
            .map_err(|e| Stop::Failed(e.to_string()))
    }

    /// Prints the answer to `prompt` as it comes. Returns it.
    fn say(&self, prompt: &str) -> Result<String, Stop> {
        let model = self.model();
        let failed = |e: String| Stop::Failed(e);
        let generator = model.generator.as_ref();
        let generator =
            generator.ok_or_else(|| failed("the model does not generate text".into()))?;
        let ids = model
            .tokens
            .encode(prompt)
            .map_err(|e| failed(e.to_string()))?;
        let room = model.info.context.saturating_sub(ids.len());
        if ids.is_empty() || room == 0 {
            return Err(failed("the prompt does not fit the context".into()));
        }
        let params = Params {
            max_tokens: self.max_tokens.map_or(room, |n| n.min(room)),
            stop: Vec::new(),
            format: None,
        };
        let (sink, steps) = mpsc::channel();
        let send = Box::new(move |step| sink.send(step).is_ok());
        generator
            .generate(ids, params, send)
            .map_err(|e| failed(e.to_string()))?;
        let (mut answer, mut out) = (String::new(), std::io::stdout());
        while let Ok(step) = steps.recv() {
            let (text, done) = match step {
                Step::Token { text, .. } => (text, false),
                Step::Done { text, .. } => (text, true),
            };
            answer += &text;
            let _ = write!(out, "{text}");
            let _ = out.flush();
            if done {
                return Ok(answer);
            }
        }
        Err(failed("the engine stopped".into()))
    }
}
