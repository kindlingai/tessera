//! Anthropic `/v1/messages` route, whole or streamed.

#[cfg(test)]
mod messages_test;

use crate::http::{Body, Request, Response};
use crate::wire;
use crate::{Model, Slot};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use tessera_bundle::{
    FunctionCall, Message, RenderOptions, ResponseEvent, ResponseParser, ToolCall,
};
use tessera_engine::Sampling;
use tessera_tasks::api::{Finish, Params, Refused, Step};

/// Anthropic error payload with HTTP status.
#[derive(Debug)]
struct Fail {
    status: u16,
    kind: &'static str,
    message: String,
}

impl Fail {
    fn bad(msg: impl Into<String>) -> Self {
        Self {
            status: 400,
            kind: "invalid_request_error",
            message: msg.into(),
        }
    }

    fn missing(msg: impl Into<String>) -> Self {
        Self {
            status: 404,
            kind: "not_found_error",
            message: msg.into(),
        }
    }

    fn busy(msg: impl Into<String>) -> Self {
        Self {
            status: 503,
            kind: "overloaded_error",
            message: msg.into(),
        }
    }

    fn body(&self) -> Value {
        json!({"type": "error", "error": {"type": self.kind, "message": self.message}})
    }

    fn response(self) -> Response {
        Response::json(self.status, &self.body())
    }
}

#[derive(Deserialize)]
struct TextBlock {
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum System {
    Text(String),
    Blocks(Vec<TextBlock>),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ResultText {
    Text(String),
    Blocks(Vec<TextBlock>),
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Block {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<ResultText>,
    },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Blocks(Vec<Block>),
}

#[derive(Deserialize)]
struct Turn {
    role: String,
    content: Content,
}

#[derive(Deserialize)]
struct ToolDef {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    input_schema: Option<Value>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Choice {
    Auto,
    Any,
    None,
    Tool { name: String },
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Think {
    Enabled,
    Disabled,
}

#[derive(Deserialize)]
struct Input {
    #[serde(default)]
    model: Option<String>,
    max_tokens: usize,
    messages: Vec<Turn>,
    #[serde(default)]
    system: Option<System>,
    #[serde(default)]
    tools: Option<Vec<ToolDef>>,
    #[serde(default)]
    tool_choice: Option<Choice>,
    #[serde(default)]
    thinking: Option<Think>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i64>,
    #[serde(default)]
    stop_sequences: Vec<String>,
    #[serde(default)]
    stream: bool,
}

/// Unique id for a message or tool use block.
fn next_id(prefix: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}_{:x}{n:x}", wire::now())
}

/// Joins text blocks into one string.
fn join_blocks(blocks: Vec<TextBlock>) -> Result<String, Fail> {
    let mut out = String::new();
    for b in blocks {
        if b.kind != "text" {
            return Err(Fail::bad("block type must be \"text\""));
        }
        out.push_str(&b.text);
    }
    Ok(out)
}

/// Converts Anthropic turns and system prompt into template messages.
fn convert(system: Option<System>, turns: Vec<Turn>) -> Result<Vec<Message>, Fail> {
    let mut out = Vec::new();
    if let Some(sys) = system {
        let text = match sys {
            System::Text(s) => s,
            System::Blocks(bs) => join_blocks(bs)?,
        };
        if !text.is_empty() {
            out.push(Message::text("system", text));
        }
    }
    for turn in turns {
        match turn.content {
            Content::Text(text) => out.push(Message::text(turn.role, text)),
            Content::Blocks(blocks) => push_blocks(&mut out, turn.role, blocks)?,
        }
    }
    Ok(out)
}

/// Flattens one multi-block turn into template messages.
fn push_blocks(out: &mut Vec<Message>, role: String, blocks: Vec<Block>) -> Result<(), Fail> {
    let (mut text, mut thinking) = (String::new(), String::new());
    let (mut calls, mut results) = (Vec::new(), Vec::new());
    for block in blocks {
        match block {
            Block::Text { text: s } => text.push_str(&s),
            Block::Thinking { thinking: s } => thinking.push_str(&s),
            Block::ToolUse { id, name, input } => {
                let arguments = match input {
                    Value::Object(m) => Value::Object(m),
                    Value::Null => json!({}),
                    _ => return Err(Fail::bad("tool_use input must be a JSON object")),
                };
                calls.push(ToolCall {
                    id: Some(id),
                    kind: "function".into(),
                    function: FunctionCall { name, arguments },
                });
            }
            Block::ToolResult {
                tool_use_id,
                content,
            } => {
                let body = match content {
                    None => String::new(),
                    Some(ResultText::Text(s)) => s,
                    Some(ResultText::Blocks(bs)) => join_blocks(bs)?,
                };
                results.push((tool_use_id, body));
            }
        }
    }
    for (id, body) in results {
        out.push(Message {
            role: "tool".into(),
            content: Some(body),
            reasoning: None,
            tool_calls: None,
            tool_call_id: Some(id),
            name: None,
        });
    }
    if !text.is_empty() || !thinking.is_empty() || !calls.is_empty() || out.is_empty() {
        let content = (!text.is_empty() || calls.is_empty()).then_some(text);
        let reasoning = (!thinking.is_empty()).then_some(thinking);
        let tool_calls = (!calls.is_empty()).then_some(calls);
        out.push(Message {
            role,
            content,
            reasoning,
            tool_calls,
            tool_call_id: None,
            name: None,
        });
    }
    Ok(())
}

/// Builds OpenAI-shaped tool schemas honoring `tool_choice`.
fn tool_schemas(tools: Option<Vec<ToolDef>>, choice: Option<&Choice>) -> Result<Vec<Value>, Fail> {
    if matches!(choice, Some(Choice::None)) {
        return Ok(Vec::new());
    }
    let given = tools.unwrap_or_default();
    let pick = match choice {
        Some(Choice::Tool { name }) => Some(name.as_str()),
        _ => None,
    };
    let mut out = Vec::new();
    for t in given {
        if t.name.trim().is_empty() {
            return Err(Fail::bad("tool name is empty"));
        }
        if pick.is_some_and(|want| want != t.name) {
            continue;
        }
        let mut func = Map::new();
        func.insert("name".into(), Value::String(t.name));
        if let Some(desc) = t.description {
            func.insert("description".into(), Value::String(desc));
        }
        if let Some(schema) = t.input_schema {
            func.insert("parameters".into(), schema);
        }
        out.push(json!({"type": "function", "function": Value::Object(func)}));
    }
    if let Some(want) = pick
        && out.is_empty()
    {
        return Err(Fail::bad(format!("tool `{want}` is not in tools")));
    }
    Ok(out)
}

/// Validates Anthropic sampling knobs into engine `Sampling`.
fn sampling(input: &Input) -> Result<Sampling, Fail> {
    let temperature = input.temperature.unwrap_or(0.0);
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(Fail::bad("temperature must be non-negative"));
    }
    let top_p = input.top_p.unwrap_or(1.0);
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(Fail::bad("top_p must be in (0, 1]"));
    }
    let top_k = match input.top_k {
        None | Some(-1) | Some(0) => 0,
        Some(k) if k > 0 => k as usize,
        Some(_) => return Err(Fail::bad("top_k must be non-negative")),
    };
    Ok(Sampling {
        temperature,
        top_p,
        top_k,
        ..Sampling::default()
    })
}

/// Maps engine finish state to Anthropic `stop_reason`.
fn stop_reason(finish: Finish, called: bool) -> &'static str {
    match (finish, called) {
        (Finish::Stop, true) => "tool_use",
        (Finish::Stop, false) => "end_turn",
        (Finish::Length, _) => "max_tokens",
    }
}

/// Formats one Anthropic SSE event line pair.
fn sse(event: &str, data: &Value) -> String {
    format!("event: {event}\ndata: {data}")
}

/// Formats a parsed tool call as an Anthropic `tool_use` block.
fn tool_block(tc: &ToolCall) -> Value {
    let id = tc.id.clone().unwrap_or_else(|| next_id("toolu"));
    let input = match &tc.function.arguments {
        Value::Object(m) => Value::Object(m.clone()),
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        _ => json!({}),
    };
    json!({"type": "tool_use", "id": id, "name": tc.function.name, "input": input})
}

struct Started {
    id: String,
    model: String,
    input_tokens: usize,
    stream: bool,
    parser: Option<ResponseParser>,
    tools: bool,
    thinking: bool,
    steps: mpsc::Receiver<Step>,
}

/// Starts one Anthropic `/v1/messages` generation.
fn start(model: &Model, body: &[u8]) -> Result<Started, Fail> {
    let mut input: Input = serde_json::from_slice(body).map_err(|e| Fail::bad(e.to_string()))?;
    if let Some(name) = &input.model
        && *name != model.info.name
    {
        return Err(Fail::missing(format!("model `{name}` does not exist")));
    }
    if input.max_tokens == 0 {
        return Err(Fail::bad("max_tokens must be at least 1"));
    }
    if input.messages.is_empty() {
        return Err(Fail::bad("messages must be non-empty"));
    }
    let chat = model
        .chat
        .as_ref()
        .ok_or_else(|| Fail::bad("model has no chat template"))?;
    let generator = model
        .generator
        .as_ref()
        .ok_or_else(|| Fail::missing("model does not generate text"))?;
    let parser = chat.response().cloned();
    let tools = tool_schemas(input.tools.take(), input.tool_choice.as_ref())?;
    if !tools.is_empty() && parser.as_ref().is_none_or(|p| !p.supports_tools()) {
        return Err(Fail::bad("model does not support tool calling"));
    }
    let think = match input.thinking {
        Some(Think::Enabled) => Some(true),
        Some(Think::Disabled) => Some(false),
        None => None,
    };
    let sampling = sampling(&input)?;
    let msgs = convert(input.system, input.messages)?;
    let options = RenderOptions {
        think,
        clear_thinking: None,
        reasoning_effort: None,
        tools: &tools,
    };
    let rendered = chat
        .render_options(&msgs, &options)
        .map_err(|e| Fail::bad(e.to_string()))?;
    let thinking = parser
        .as_ref()
        .is_some_and(|p| p.starts_thinking(&rendered));
    let prompt = model
        .tokens
        .encode(&rendered)
        .map_err(|e| Fail::bad(e.to_string()))?;
    let room = model.info.context.saturating_sub(prompt.len());
    if input.max_tokens > room {
        let msg = format!(
            "prompt ({}) + max_tokens ({}) exceeds context ({})",
            prompt.len(),
            input.max_tokens,
            model.info.context
        );
        return Err(Fail::bad(msg));
    }
    let params = Params {
        max_tokens: input.max_tokens,
        stop: input.stop_sequences,
        raw: parser.is_some(),
        sampling,
        starts_thinking: thinking,
        ..Params::default()
    };
    let (sink, steps) = mpsc::channel();
    let input_tokens = prompt.len();
    let each = move |step: Step| sink.send(step).is_ok();
    generator
        .generate(prompt, params, Box::new(each))
        .map_err(|e| match e.downcast_ref::<Refused>() {
            Some(why) => Fail::bad(why.0.clone()),
            None => Fail::busy(e.to_string()),
        })?;
    Ok(Started {
        id: next_id("msg"),
        model: model.info.name.clone(),
        input_tokens,
        stream: input.stream,
        parser,
        tools: !tools.is_empty(),
        thinking,
        steps,
    })
}

impl Started {
    fn whole(self) -> Response {
        let mut text = String::new();
        while let Ok(step) = self.steps.recv() {
            match step {
                Step::Token { text: piece, .. } | Step::Text { text: piece, .. } => {
                    text.push_str(&piece)
                }
                Step::Done {
                    text: tail,
                    finish,
                    tokens,
                    cached,
                } => {
                    text.push_str(&tail);
                    return self.reply(&text, finish, tokens, cached);
                }
            }
        }
        Fail::busy("engine stopped").response()
    }

    fn reply(&self, text: &str, finish: Finish, tokens: usize, cached: usize) -> Response {
        let mut content = Vec::new();
        let mut called = false;
        if let Some(parser) = &self.parser {
            let parsed = parser.parse(text, self.tools, self.thinking);
            if !parsed.reasoning.is_empty() {
                content.push(json!({"type": "thinking", "thinking": parsed.reasoning}));
            }
            if !parsed.content.is_empty() || (parsed.tool_calls.is_empty() && content.is_empty()) {
                content.push(json!({"type": "text", "text": parsed.content}));
            }
            called = !parsed.tool_calls.is_empty();
            for tc in &parsed.tool_calls {
                content.push(tool_block(tc));
            }
        } else {
            content.push(json!({"type": "text", "text": text}));
        }
        let body = json!({
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": self.model,
            "stop_reason": stop_reason(finish, called),
            "stop_sequence": null,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": tokens,
                "cache_read_input_tokens": cached,
            },
        });
        Response::json(200, &body)
    }

    fn forward(self, out: &mpsc::Sender<String>) {
        let send = |ev: &str, data: Value| out.send(sse(ev, &data)).is_ok();
        let head = json!({
            "type": "message_start",
            "message": {
                "id": self.id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
            },
        });
        if !send("message_start", head) {
            return;
        }
        let mut parser = self
            .parser
            .as_ref()
            .map(|p| p.stream(self.tools, self.thinking));
        let mut events = Vec::new();
        let mut open: Option<(&'static str, usize)> = None;
        let mut next_idx = 0usize;
        let mut called = false;
        while let Ok(step) = self.steps.recv() {
            let (piece, end) = match step {
                Step::Token { text, .. } | Step::Text { text, .. } => (text, None),
                Step::Done {
                    text,
                    finish,
                    tokens,
                    cached,
                } => (text, Some((finish, tokens, cached))),
            };
            if let Some(s) = &mut parser {
                s.push(&piece, &mut events);
                if end.is_some() {
                    s.finish(&mut events);
                }
            } else if !piece.is_empty() {
                events.push(ResponseEvent::Content(piece));
            }
            for ev in events.drain(..) {
                if !emit_event(ev, &mut open, &mut next_idx, &mut called, &send) {
                    return;
                }
            }
            if let Some((finish, tokens, cached)) = end {
                if let Some((_, idx)) = open.take() {
                    let stop = json!({"type": "content_block_stop", "index": idx});
                    if !send("content_block_stop", stop) {
                        return;
                    }
                }
                let delta = json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason(finish, called), "stop_sequence": null},
                    "usage": {"output_tokens": tokens, "cache_read_input_tokens": cached},
                });
                let _ = send("message_delta", delta);
                let _ = send("message_stop", json!({"type": "message_stop"}));
                return;
            }
        }
        let _ = send("error", Fail::busy("engine stopped").body());
    }
}

/// Emits one parsed response event as Anthropic SSE blocks.
fn emit_event(
    event: ResponseEvent,
    open: &mut Option<(&'static str, usize)>,
    next_idx: &mut usize,
    called: &mut bool,
    send: &impl Fn(&str, Value) -> bool,
) -> bool {
    match event {
        ResponseEvent::Reasoning(s) => {
            if !open_block(
                "thinking",
                json!({"type": "thinking", "thinking": ""}),
                open,
                next_idx,
                send,
            ) {
                return false;
            }
            let idx = open.unwrap().1;
            let delta = json!({"type": "content_block_delta", "index": idx, "delta": {"type": "thinking_delta", "thinking": s}});
            send("content_block_delta", delta)
        }
        ResponseEvent::Content(s) => {
            if !open_block(
                "text",
                json!({"type": "text", "text": ""}),
                open,
                next_idx,
                send,
            ) {
                return false;
            }
            let idx = open.unwrap().1;
            let delta = json!({"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": s}});
            send("content_block_delta", delta)
        }
        ResponseEvent::ToolCallStart { id, name, .. } => {
            *called = true;
            if let Some((_, idx)) = open.take()
                && !send(
                    "content_block_stop",
                    json!({"type": "content_block_stop", "index": idx}),
                )
            {
                return false;
            }
            let idx = *next_idx;
            *next_idx += 1;
            *open = Some(("tool_use", idx));
            let id = id.unwrap_or_else(|| next_id("toolu"));
            let start = json!({"type": "content_block_start", "index": idx, "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}});
            send("content_block_start", start)
        }
        ResponseEvent::ToolCallArgs(raw) => {
            if raw.is_empty() {
                return true;
            }
            let Some(("tool_use", idx)) = *open else {
                return true;
            };
            let delta = json!({"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": raw}});
            send("content_block_delta", delta)
        }
        ResponseEvent::ToolCallEnd => {
            if let Some(("tool_use", idx)) = open.take() {
                return send(
                    "content_block_stop",
                    json!({"type": "content_block_stop", "index": idx}),
                );
            }
            true
        }
        ResponseEvent::ToolCall(tc) => {
            *called = true;
            if let Some((_, idx)) = open.take()
                && !send(
                    "content_block_stop",
                    json!({"type": "content_block_stop", "index": idx}),
                )
            {
                return false;
            }
            let idx = *next_idx;
            *next_idx += 1;
            let id = tc.id.clone().unwrap_or_else(|| next_id("toolu"));
            let start = json!({"type": "content_block_start", "index": idx, "content_block": {"type": "tool_use", "id": id, "name": tc.function.name, "input": {}}});
            if !send("content_block_start", start) {
                return false;
            }
            let raw = match &tc.function.arguments {
                Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_else(|_| "{}".into()),
            };
            let delta = json!({"type": "content_block_delta", "index": idx, "delta": {"type": "input_json_delta", "partial_json": raw}});
            if !send("content_block_delta", delta) {
                return false;
            }
            send(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": idx}),
            )
        }
    }
}

/// Switches the active streaming block to `kind` when needed.
fn open_block(
    kind: &'static str,
    block: Value,
    open: &mut Option<(&'static str, usize)>,
    next_idx: &mut usize,
    send: &impl Fn(&str, Value) -> bool,
) -> bool {
    if open.is_some_and(|(k, _)| k == kind) {
        return true;
    }
    if let Some((_, idx)) = open.take()
        && !send(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": idx}),
        )
    {
        return false;
    }
    let idx = *next_idx;
    *next_idx += 1;
    *open = Some((kind, idx));
    send(
        "content_block_start",
        json!({"type": "content_block_start", "index": idx, "content_block": block}),
    )
}

/// Handles `POST /v1/messages`.
pub(crate) fn messages(slot: &Slot, request: &Request) -> Response {
    let Some(model) = slot.get() else {
        return Fail::busy("the model is loading").response();
    };
    let started = match start(model, &request.body) {
        Ok(s) => s,
        Err(fail) => return fail.response(),
    };
    if !started.stream {
        return started.whole();
    }
    let (out, events) = mpsc::channel();
    std::thread::spawn(move || started.forward(&out));
    Response {
        status: 200,
        body: Body::Events(events),
    }
}
