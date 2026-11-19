//! Completions and chat completions, whole or streamed.

use crate::http::{Body, Response};
use crate::wire::{self, Chat, Common, Completion, Prompt, Refusal, ResponseFormat, bad, refuse};
use crate::{Model, Slot, with_model};
use sensai_tasks::api::{Finish, Params, Refused, Step};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;

/// Which OpenAI shape an answer takes.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Text,
    Chat,
}

/// An answer in flight, and how to report it.
struct Started {
    kind: Kind,
    id: String,
    model: String,
    prompt_tokens: usize,
    stream: bool,
    usage: bool,
    steps: mpsc::Receiver<Step>,
}

fn finish(f: Finish) -> &'static str {
    match f {
        Finish::Stop => "stop",
        Finish::Length => "length",
    }
}

/// An id no other answer from this process shares.
fn fresh_id(kind: Kind) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let prefix = match kind {
        Kind::Text => "cmpl",
        Kind::Chat => "chatcmpl",
    };
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{:x}{n:x}", wire::now())
}

/// The tokens asked for, or the room the context leaves.
fn budget(model: &Model, prompt: usize, asked: Option<usize>) -> Result<usize, Refusal> {
    let context = model.info.context;
    let room = context.saturating_sub(prompt);
    match asked {
        Some(0) => Err(bad("max_tokens must be at least 1", "max_tokens")),
        Some(n) if n <= room => Ok(n),
        None if room > 0 => Ok(room),
        _ => {
            let n = asked.unwrap_or(0);
            let message = format!(
                "This model's maximum context length is {context} tokens. However, you \
                 requested {} tokens ({prompt} in the messages, {n} in the completion). \
                 Please reduce the length of the messages or completion.",
                prompt + n
            );
            Err(bad(message, "max_tokens"))
        }
    }
}

/// Starts the answer to `prompt` on the model's generator.
fn start(
    model: &Model,
    kind: Kind,
    prompt: Vec<u32>,
    asked: Option<usize>,
    common: Common,
) -> Result<Started, Refusal> {
    let generator = model
        .generator
        .as_ref()
        .ok_or_else(|| refuse(404, "the model does not generate text", None))?;
    if prompt.is_empty() {
        return Err(bad("the prompt is empty", "prompt"));
    }
    let params = Params {
        max_tokens: budget(model, prompt.len(), asked)?,
        stop: common.stop.map(wire::Strings::list).unwrap_or_default(),
        format: common.response_format.and_then(ResponseFormat::format),
    };
    let (sink, steps) = mpsc::channel();
    let prompt_tokens = prompt.len();
    generator
        .generate(
            prompt,
            params,
            Box::new(move |step| sink.send(step).is_ok()),
        )
        .map_err(|e| match e.downcast_ref::<Refused>() {
            Some(why) => bad(why.0.clone(), "response_format"),
            None => refuse(503, e.to_string(), None),
        })?;
    Ok(Started {
        kind,
        id: fresh_id(kind),
        model: model.info.name.clone(),
        prompt_tokens,
        stream: common.stream,
        usage: common.stream_options.is_some_and(|o| o.include_usage),
        steps,
    })
}

fn complete(model: &Model, body: &[u8]) -> Result<Started, Refusal> {
    let c: Completion = wire::parse(model, body)?;
    let prompt = match c.prompt {
        Prompt::Ids(ids) => ids,
        Prompt::Text(texts) => match texts.list().as_slice() {
            [text] => model
                .tokens
                .encode(text)
                .map_err(|e| bad(e.to_string(), "prompt"))?,
            _ => return Err(bad("give one prompt per request", "prompt")),
        },
    };
    start(model, Kind::Text, prompt, c.common.max_tokens, c.common)
}

fn chat(model: &Model, body: &[u8]) -> Result<Started, Refusal> {
    let c: Chat = wire::parse(model, body)?;
    let prompt = crate::render(model, c.messages)?;
    let prompt = model
        .tokens
        .encode(&prompt)
        .map_err(|e| bad(e.to_string(), "messages"))?;
    let asked = c.max_completion_tokens.or(c.common.max_tokens);
    start(model, Kind::Chat, prompt, asked, c.common)
}

impl Started {
    fn object(&self) -> &'static str {
        match (self.kind, self.stream) {
            (Kind::Text, _) => "text_completion",
            (Kind::Chat, false) => "chat.completion",
            (Kind::Chat, true) => "chat.completion.chunk",
        }
    }

    /// One choice: whole text, or a piece of a stream.
    fn choice(&self, text: &str, finish: Option<Finish>) -> Value {
        let reason = finish.map(self::finish);
        match (self.kind, self.stream) {
            (Kind::Text, _) => {
                json!({"index": 0, "text": text, "logprobs": null, "finish_reason": reason})
            }
            (Kind::Chat, false) => json!({
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "logprobs": null,
                "finish_reason": reason,
            }),
            (Kind::Chat, true) => {
                let delta = match text.is_empty() {
                    true => json!({}),
                    false => json!({"content": text}),
                };
                json!({"index": 0, "delta": delta, "logprobs": null, "finish_reason": reason})
            }
        }
    }

    /// `usage` holds the answer's tokens and the cached prompt tokens.
    fn body(&self, choices: Value, usage: Option<(usize, usize)>) -> Value {
        let mut body = json!({
            "id": self.id,
            "object": self.object(),
            "created": wire::now(),
            "model": self.model,
            "choices": choices,
        });
        if let Some((tokens, cached)) = usage {
            body["usage"] = json!({
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": tokens,
                "total_tokens": self.prompt_tokens + tokens,
                "prompt_tokens_details": {"cached_tokens": cached},
            });
        }
        body
    }

    /// Waits for the end and answers in one body.
    fn whole(self) -> Response {
        let mut text = String::new();
        while let Ok(step) = self.steps.recv() {
            match step {
                Step::Token { text: piece, .. } => text += &piece,
                Step::Done {
                    text: rest,
                    finish,
                    tokens,
                    cached,
                } => {
                    text += &rest;
                    let choice = self.choice(&text, Some(finish));
                    let usage = Some((tokens, cached));
                    return Response::json(200, &self.body(json!([choice]), usage));
                }
            }
        }
        refuse(503, "the engine stopped", None).into()
    }

    /// Sends each step as a chunk, then `[DONE]`.
    fn forward(self, out: &mpsc::Sender<String>) {
        let send = |v: Value| out.send(v.to_string()).is_ok();
        if self.kind == Kind::Chat {
            let delta = json!([{"index": 0, "delta": {"role": "assistant", "content": ""},
                "logprobs": null, "finish_reason": null}]);
            if !send(self.body(delta, None)) {
                return;
            }
        }
        while let Ok(step) = self.steps.recv() {
            let (text, end) = match step {
                Step::Token { text, .. } if text.is_empty() => continue,
                Step::Token { text, .. } => (text, None),
                Step::Done {
                    text,
                    finish,
                    tokens,
                    cached,
                } => (text, Some((finish, (tokens, cached)))),
            };
            let choice = self.choice(&text, end.map(|(f, _)| f));
            if !send(self.body(json!([choice]), None)) {
                return;
            }
            if let Some((_, usage)) = end {
                if self.usage {
                    send(self.body(json!([]), Some(usage)));
                }
                let _ = out.send("[DONE]".into());
                return;
            }
        }
        // The steps stopped before `Done`.
        send(refuse(503, "the engine stopped", None).body());
        let _ = out.send("[DONE]".into());
    }

    fn respond(self) -> Response {
        if !self.stream {
            return self.whole();
        }
        let (out, events) = mpsc::channel();
        std::thread::spawn(move || self.forward(&out));
        Response {
            status: 200,
            body: Body::Events(events),
        }
    }
}

pub(crate) fn completions(slot: &Slot, body: &[u8]) -> Response {
    match with_model(slot, |m| complete(m, body)) {
        Ok(started) => started.respond(),
        Err(refused) => refused.into(),
    }
}

pub(crate) fn chat_completions(slot: &Slot, body: &[u8]) -> Response {
    match with_model(slot, |m| chat(m, body)) {
        Ok(started) => started.respond(),
        Err(refused) => refused.into(),
    }
}
