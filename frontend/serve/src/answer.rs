//! Completions and chat completions, whole or streamed.

use crate::http::{Body, Request, Response};
use crate::wire::{self, Chat, Common, Completion, Prompt, Refusal, ResponseFormat, bad, refuse};
use crate::{Model, Slot, with_model};
use serde_json::{Map, Value, json};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use tessera_bundle::{ResponseEvent, ResponseParser, ResponseStream, ToolCall};
use tessera_engine::Ask;
use tessera_tasks::api::{Finish, Params, Refused, Step, ToolChoice};

/// An answer's OpenAI shape.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Text,
    Chat,
}

/// How a chat answer parses structured output blocks.
struct Parsing {
    parser: Option<ResponseParser>,
    tools: bool,
    parallel_tools: bool,
    thinking: bool,
    tool_choice: ToolChoice,
    forced_prefix: String,
}

impl Parsing {
    fn plain() -> Self {
        Self {
            parser: None,
            tools: false,
            parallel_tools: true,
            thinking: false,
            tool_choice: ToolChoice::Auto,
            forced_prefix: String::new(),
        }
    }
}

/// An answer in flight, and how to report it.
struct Started {
    kind: Kind,
    id: String,
    model: String,
    prompt_tokens: usize,
    stream: bool,
    usage: bool,
    parsing: Parsing,
    steps: mpsc::Receiver<Step>,
}

fn finish(f: Finish, called: bool) -> &'static str {
    match (f, called) {
        (Finish::Stop, true) => "tool_calls",
        (Finish::Stop, false) => "stop",
        (Finish::Length, _) => "length",
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

/// Generates a fresh tool-call id unique within this process.
fn next_call_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("call_{:x}{n:x}", wire::now())
}

/// A tool-call id unique within this process.
fn call_id(call: &ToolCall) -> String {
    call.id.clone().unwrap_or_else(next_call_id)
}

/// Formats one tool call in OpenAI's JSON shape.
fn format_call(call: &ToolCall, index: Option<usize>) -> Value {
    let args = match &call.function.arguments {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "{}".into()),
    };
    let mut obj = Map::new();
    if let Some(i) = index {
        obj.insert("index".into(), json!(i));
    }
    obj.insert("id".into(), json!(call_id(call)));
    obj.insert("type".into(), json!(call.kind));
    obj.insert(
        "function".into(),
        json!({"name": call.function.name, "arguments": args}),
    );
    Value::Object(obj)
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
    parsing: Parsing,
    ask: Option<Ask>,
) -> Result<Started, Refusal> {
    let generator = model
        .generator
        .as_ref()
        .ok_or_else(|| refuse(404, "the model does not generate text", None))?;
    if prompt.is_empty() {
        return Err(bad("the prompt is empty", "prompt"));
    }
    let sampling = common.sampling()?;
    let penalties = common.penalties()?;
    let thinking = common.thinking()?;
    let sequence = common.sequence()?;
    let max_tokens = budget(model, prompt.len(), asked)?;
    let min_tokens = common.min_tokens.unwrap_or(0);
    if min_tokens > max_tokens {
        return Err(bad("min_tokens must not exceed max_tokens", "min_tokens"));
    }
    let params = Params {
        max_tokens,
        min_tokens,
        stop: common.stop.map(wire::Strings::list).unwrap_or_default(),
        format: common.response_format.and_then(ResponseFormat::format),
        raw: parsing.parser.is_some(),
        sampling,
        penalties,
        thinking,
        sequence,
        starts_thinking: parsing.thinking,
        tool_choice: parsing.tool_choice.clone(),
    };
    let (sink, steps) = mpsc::channel();
    let prompt_tokens = prompt.len();
    let mut ask = ask;
    let each = move |step: Step| {
        let last = matches!(step, Step::Done { .. });
        let sent = sink.send(step).is_ok();
        // The answer ended, or the client left.
        if last || !sent {
            drop(ask.take());
        }
        sent
    };
    generator
        .generate(prompt, params, Box::new(each))
        .map_err(|e| match e.downcast_ref::<Refused>() {
            Some(why) => bad(why.0.clone(), why.1),
            None => refuse(503, e.to_string(), None),
        })?;
    Ok(Started {
        kind,
        id: fresh_id(kind),
        model: model.info.name.clone(),
        prompt_tokens,
        stream: common.stream,
        usage: common.stream_options.is_some_and(|o| o.include_usage),
        parsing,
        steps,
    })
}

fn complete(model: &Model, body: &[u8], ask: Option<Ask>) -> Result<Started, Refusal> {
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
    start(
        model,
        Kind::Text,
        prompt,
        c.common.max_tokens,
        c.common,
        Parsing::plain(),
        ask,
    )
}

fn chat(model: &Model, body: &[u8], ask: Option<Ask>) -> Result<Started, Refusal> {
    let c: Chat = wire::parse(model, body)?;
    let given_tools = c.messages.tools.as_deref().unwrap_or(&[]);
    let (allow_tools, tool_choice) = match &c.tool_choice {
        Some(choice) => choice.resolve(given_tools)?,
        None => (true, ToolChoice::Auto),
    };
    if tool_choice != ToolChoice::Auto && c.common.response_format.is_some() {
        return Err(bad(
            "`tool_choice` cannot force a tool call with `response_format`",
            "tool_choice",
        ));
    }
    let parser = model.chat.as_ref().and_then(|ch| ch.response().cloned());
    let has_tools = allow_tools && !given_tools.is_empty();
    if has_tools && parser.as_ref().is_none_or(|p| !p.supports_tools()) {
        return Err(bad("this model does not support tool calling", "tools"));
    }
    let (mut prompt, tools) = crate::render(model, c.messages, allow_tools)?;
    let mut thinking = parser.as_ref().is_some_and(|p| p.starts_thinking(&prompt));
    let mut forced_prefix = String::new();
    let mut tool_choice = tool_choice;
    if tool_choice != ToolChoice::Auto
        && model
            .generator
            .as_ref()
            .is_some_and(|g| !g.supports_guides())
    {
        let name = match &tool_choice {
            ToolChoice::Function(n) => Some(n.as_str()),
            ToolChoice::Required | ToolChoice::Auto => None,
        };
        let open = parser
            .as_ref()
            .and_then(|p| p.call_prefix(name))
            .ok_or_else(|| bad("this model cannot force a tool call", "tool_choice"))?;
        if thinking {
            let close = parser
                .as_ref()
                .and_then(|p| p.markers().think_close.into_iter().next())
                .unwrap_or("</think>");
            prompt.push('\n');
            prompt.push_str(close);
            prompt.push_str("\n\n");
            thinking = false;
        }
        prompt.push_str(&open);
        forced_prefix = open;
        tool_choice = ToolChoice::Auto;
    }
    let prompt = model
        .tokens
        .encode(&prompt)
        .map_err(|e| bad(e.to_string(), "messages"))?;
    let asked = c.max_completion_tokens.or(c.common.max_tokens);
    let parsing = Parsing {
        parser,
        tools,
        parallel_tools: c.parallel_tool_calls.unwrap_or(true),
        thinking,
        tool_choice,
        forced_prefix,
    };
    start(model, Kind::Chat, prompt, asked, c.common, parsing, ask)
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
    fn choice(&self, text: &str, end: Option<Finish>) -> Value {
        let reason = end.map(|f| finish(f, false));
        match (self.kind, self.stream) {
            (Kind::Text, _) => {
                json!({"index": 0, "text": text, "logprobs": null, "finish_reason": reason})
            }
            (Kind::Chat, false) => self.chat_choice(text, end),
            (Kind::Chat, true) => {
                let delta = match text.is_empty() {
                    true => json!({}),
                    false => json!({"content": text}),
                };
                self.delta_choice(delta, reason)
            }
        }
    }

    fn delta_choice(&self, delta: Value, reason: Option<&str>) -> Value {
        json!({"index": 0, "delta": delta, "logprobs": null, "finish_reason": reason})
    }

    fn chat_choice(&self, text: &str, end: Option<Finish>) -> Value {
        let Some(parser) = &self.parsing.parser else {
            return json!({
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "logprobs": null,
                "finish_reason": end.map(|f| finish(f, false)),
            });
        };
        let parsed = parser.parse(text, self.parsing.tools, self.parsing.thinking);
        let called = !parsed.tool_calls.is_empty();
        let mut msg = Map::new();
        msg.insert("role".into(), json!("assistant"));
        let content = match (called, parsed.content.is_empty()) {
            (true, true) => Value::Null,
            _ => Value::String(parsed.content),
        };
        msg.insert("content".into(), content);
        if !parsed.reasoning.is_empty() {
            msg.insert("reasoning_content".into(), Value::String(parsed.reasoning));
        }
        if called {
            let limit = match self.parsing.parallel_tools {
                true => parsed.tool_calls.len(),
                false => 1,
            };
            let calls: Vec<Value> = parsed
                .tool_calls
                .iter()
                .take(limit)
                .map(|c| format_call(c, None))
                .collect();
            msg.insert("tool_calls".into(), Value::Array(calls));
        }
        json!({
            "index": 0,
            "message": Value::Object(msg),
            "logprobs": null,
            "finish_reason": end.map(|f| finish(f, called)),
        })
    }

    /// `usage` holds answer, reasoning, and cached prompt token counts.
    fn body(&self, choices: Value, usage: Option<(usize, usize, usize)>) -> Value {
        let mut body = json!({
            "id": self.id,
            "object": self.object(),
            "created": wire::now(),
            "model": self.model,
            "choices": choices,
        });
        if let Some((tokens, reasoning, cached)) = usage {
            let mut u = json!({
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": tokens,
                "total_tokens": self.prompt_tokens + tokens,
                "prompt_tokens_details": {"cached_tokens": cached},
            });
            if reasoning > 0 {
                u["completion_tokens_details"] = json!({"reasoning_tokens": reasoning});
            }
            body["usage"] = u;
        }
        body
    }

    /// Waits for the end and answers in one body.
    fn whole(self) -> Response {
        let mut text = self.parsing.forced_prefix.clone();
        let mut reasoning = 0usize;
        let mut stream = self
            .parsing
            .parser
            .as_ref()
            .map(|p| p.stream(self.parsing.tools, self.parsing.thinking));
        let mut events = Vec::new();
        if !self.parsing.forced_prefix.is_empty()
            && let Some(s) = &mut stream
        {
            feed_stream(
                s,
                &self.parsing.forced_prefix,
                false,
                &mut events,
                &mut reasoning,
            );
            events.clear();
        }
        while let Ok(step) = self.steps.recv() {
            match step {
                Step::Token { text: piece, .. } => {
                    if let Some(s) = &mut stream {
                        events.clear();
                        feed_stream(s, &piece, false, &mut events, &mut reasoning);
                    }
                    text += &piece;
                }
                Step::Done {
                    text: rest,
                    finish,
                    tokens,
                    cached,
                } => {
                    if let Some(s) = &mut stream {
                        events.clear();
                        feed_stream(s, &rest, true, &mut events, &mut reasoning);
                    }
                    text += &rest;
                    let choice = self.choice(&text, Some(finish));
                    let usage = Some((tokens, reasoning.min(tokens), cached));
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
        if self.kind == Kind::Chat
            && let Some(parser) = self.parsing.parser.clone()
        {
            self.forward_parsed(&parser, out);
            return;
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
                } => (text, Some((finish, (tokens, 0, cached)))),
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

    /// Streams parsed content, reasoning and tool calls.
    fn forward_parsed(&self, parser: &ResponseParser, out: &mpsc::Sender<String>) {
        let send = |v: Value| out.send(v.to_string()).is_ok();
        let mut stream = parser.stream(self.parsing.tools, self.parsing.thinking);
        let mut events = Vec::new();
        let mut calls = 0usize;
        let mut active_call: Option<usize> = None;
        let mut suppress_call = false;
        let mut reasoning = 0usize;
        if !self.parsing.forced_prefix.is_empty() {
            feed_stream(
                &mut stream,
                &self.parsing.forced_prefix,
                false,
                &mut events,
                &mut reasoning,
            );
        }
        while let Ok(step) = self.steps.recv() {
            let (text, end) = match step {
                Step::Token { text, .. } => (text, None),
                Step::Done {
                    text,
                    finish,
                    tokens,
                    cached,
                } => (text, Some((finish, (tokens, cached)))),
            };
            feed_stream(
                &mut stream,
                &text,
                end.is_some(),
                &mut events,
                &mut reasoning,
            );
            for event in events.drain(..) {
                let delta = match event {
                    ResponseEvent::Content(s) => json!({"content": s}),
                    ResponseEvent::Reasoning(s) => json!({"reasoning_content": s}),
                    ResponseEvent::ToolCall(tc) => {
                        if !self.parsing.parallel_tools && calls > 0 {
                            continue;
                        }
                        let chunk = format_call(&tc, Some(calls));
                        calls += 1;
                        json!({"tool_calls": [chunk]})
                    }
                    ResponseEvent::ToolCallStart { id, kind, name } => {
                        if !self.parsing.parallel_tools && calls > 0 {
                            suppress_call = true;
                            continue;
                        }
                        suppress_call = false;
                        let idx = calls;
                        calls += 1;
                        active_call = Some(idx);
                        let cid = id.unwrap_or_else(next_call_id);
                        json!({"tool_calls": [{
                            "index": idx,
                            "id": cid,
                            "type": kind,
                            "function": {"name": name, "arguments": ""},
                        }]})
                    }
                    ResponseEvent::ToolCallArgs(args) => {
                        if suppress_call || args.is_empty() {
                            continue;
                        }
                        let Some(idx) = active_call else {
                            continue;
                        };
                        json!({"tool_calls": [{
                            "index": idx,
                            "function": {"arguments": args},
                        }]})
                    }
                    ResponseEvent::ToolCallEnd => {
                        active_call = None;
                        suppress_call = false;
                        continue;
                    }
                };
                let choice = self.delta_choice(delta, None);
                if !send(self.body(json!([choice]), None)) {
                    return;
                }
            }
            if let Some((reason, (tokens, cached))) = end {
                let choice = self.delta_choice(json!({}), Some(finish(reason, calls > 0)));
                if !send(self.body(json!([choice]), None)) {
                    return;
                }
                if self.usage {
                    let usage = (tokens, reasoning.min(tokens), cached);
                    send(self.body(json!([]), Some(usage)));
                }
                let _ = out.send("[DONE]".into());
                return;
            }
        }
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

/// Asks to profile launches while the answer runs.
const PROFILE: &str = "x-tessera-profile";

/// A hold on the profile count, if `request` asks.
fn ask(model: &Model, request: &Request) -> Result<Option<Ask>, Refusal> {
    let Some(value) = request.header(PROFILE).filter(|v| *v != "0") else {
        return Ok(None);
    };
    match (&model.asks, value) {
        (Some(asks), "1") => Ok(Some(asks.hold())),
        (Some(_), _) => Err(bad(format!("{PROFILE} must be 0 or 1"), PROFILE)),
        (None, _) => Err(bad(
            "start the server with TESSERA_PROFILE=dynamic",
            PROFILE,
        )),
    }
}

pub(crate) fn completions(slot: &Slot, request: &Request) -> Response {
    match with_model(slot, |m| complete(m, &request.body, ask(m, request)?)) {
        Ok(started) => started.respond(),
        Err(refused) => refused.into(),
    }
}

pub(crate) fn chat_completions(slot: &Slot, request: &Request) -> Response {
    match with_model(slot, |m| chat(m, &request.body, ask(m, request)?)) {
        Ok(started) => started.respond(),
        Err(refused) => refused.into(),
    }
}

/// Feeds one token's text to `stream` and counts reasoning tokens.
fn feed_stream(
    stream: &mut ResponseStream<'_>,
    text: &str,
    done: bool,
    events: &mut Vec<ResponseEvent>,
    reasoning: &mut usize,
) {
    let was = stream.in_thinking();
    let start = events.len();
    stream.push(text, events);
    if done {
        stream.finish(events);
    }
    let saw = events[start..]
        .iter()
        .any(|e| matches!(e, ResponseEvent::Reasoning(_)));
    if saw || (!done && (was || stream.in_thinking())) {
        *reasoning += 1;
    }
}
