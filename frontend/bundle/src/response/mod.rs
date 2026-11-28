//! Generic response parser driven by `tokenizer_config.json`.

mod close;
#[cfg(test)]
mod close_test;
mod formats;
#[cfg(test)]
mod response_test;
mod stream;

use crate::template::{FunctionCall, Message, Template, ToolCall, spaced};
use close::{Closers, literal_prefix};
use fancy_regex::Regex;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::error::Error;
use stream::{StepOutcome, TagDelims, ToolState, parse_json};

pub use formats::Markers;

/// One parsed piece from a model response.
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseEvent {
    Content(String),
    Reasoning(String),
    ToolCall(ToolCall),
    ToolCallStart {
        id: Option<String>,
        kind: String,
        name: String,
    },
    ToolCallArgs(String),
    ToolCallEnd,
}

/// A full parsed assistant response.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedResponse {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum Strings {
    One(String),
    Many(Vec<String>),
}

impl Strings {
    fn list(self) -> Vec<String> {
        match self {
            Strings::One(s) => vec![s],
            Strings::Many(v) => v,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ContentArgs {
    #[serde(default)]
    string_delims: Vec<[String; 2]>,
    #[serde(default)]
    unquoted_keys: bool,
    /// `tags` content: a regex with `key` and `value` groups.
    #[serde(alias = "tag_pattern")]
    pattern: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct FieldSpec {
    open: Option<Strings>,
    open_pattern: Option<String>,
    close: Option<Strings>,
    close_pattern: Option<String>,
    content: Option<String>,
    #[serde(default)]
    content_args: ContentArgs,
    transform: Option<Value>,
    /// Wraps a run of calls. Text between calls is dropped.
    outer: Option<[String; 2]>,
    /// Template with `{name}` for forced calls.
    call_prefix: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct ResponseTemplate {
    #[serde(default)]
    fields: BTreeMap<String, FieldSpec>,
    /// Jinja that lists `tools` when the chat template does not.
    tool_prompt: Option<String>,
}

#[derive(Deserialize)]
struct ConfigFile {
    response_template: Option<ResponseTemplate>,
}

#[derive(Clone)]
enum Opener {
    /// Any one of these strings.
    Literal(Vec<String>),
    Pattern {
        prefix: String,
        regex: Regex,
    },
}

impl Opener {
    fn compile(
        open: Option<Strings>,
        pattern: Option<String>,
    ) -> Result<Option<Self>, Box<dyn Error + Send + Sync>> {
        if let Some(pat) = pattern {
            let prefix = literal_prefix(&pat);
            let regex = Regex::new(&pat)?;
            return Ok(Some(Opener::Pattern { prefix, regex }));
        }
        let mut opens = open.map(Strings::list).unwrap_or_default();
        opens.retain(|s| !s.is_empty());
        Ok((!opens.is_empty()).then_some(Opener::Literal(opens)))
    }

    fn triggers(&self) -> Vec<&str> {
        match self {
            Opener::Literal(opens) => opens.iter().map(String::as_str).collect(),
            Opener::Pattern { prefix, .. } if !prefix.is_empty() => vec![prefix.as_str()],
            Opener::Pattern { .. } => Vec::new(),
        }
    }

    fn find(&self, text: &str) -> Option<(usize, usize)> {
        match self {
            Opener::Literal(opens) => find_literal(text, opens),
            Opener::Pattern { prefix, .. } if !prefix.is_empty() => {
                text.find(prefix.as_str()).map(|at| (at, 0))
            }
            Opener::Pattern { regex, .. } => {
                regex.find(text).ok().flatten().map(|m| (m.start(), 0))
            }
        }
    }

    /// Earliest byte offset where a block closer can match.
    fn min_close_offset(&self, text: &str) -> usize {
        let floor = text.chars().next().map_or(0, char::len_utf8);
        match self {
            Opener::Literal(opens) => opens
                .iter()
                .find(|s| text.starts_with(s.as_str()))
                .map_or(floor, String::len),
            Opener::Pattern { prefix, .. } if !prefix.is_empty() && text.starts_with(prefix) => {
                prefix.len()
            }
            Opener::Pattern { .. } => floor,
        }
    }

    /// Strips the opener at byte zero and returns captures.
    fn strip<'a>(&self, block: &'a str) -> Option<(BTreeMap<String, String>, &'a str)> {
        match self {
            Opener::Literal(opens) => {
                let rest = opens.iter().find_map(|s| block.strip_prefix(s.as_str()))?;
                Some((BTreeMap::new(), rest))
            }
            Opener::Pattern { regex, .. } => {
                let caps = regex.captures(block).ok()??;
                let whole = caps.get(0)?;
                if whole.start() != 0 {
                    return None;
                }
                let mut map = BTreeMap::new();
                for name in regex.capture_names().flatten() {
                    if let Some(m) = caps.name(name) {
                        map.insert(name.to_string(), m.as_str().to_string());
                    }
                }
                Some((map, &block[whole.end()..]))
            }
        }
    }
}

/// How a block's body becomes a value.
#[derive(Clone)]
enum Body {
    Text,
    Json(ContentArgs),
    /// One object entry per `key`/`value` match.
    Tags {
        regex: Regex,
        delims: Option<TagDelims>,
    },
}

impl Body {
    fn compile(
        kind: Option<&str>,
        args: ContentArgs,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        Ok(match kind {
            Some("json") => Body::Json(args),
            Some("tags" | "xml-inline" | "xml_params") => {
                let raw = args
                    .pattern
                    .ok_or("tags content needs content_args.pattern")?;
                let pat = if raw.starts_with("(?") {
                    raw
                } else {
                    format!("(?s){raw}")
                };
                let delims = TagDelims::from_pattern(&pat);
                Body::Tags {
                    regex: Regex::new(&pat)?,
                    delims,
                }
            }
            _ => Body::Text,
        })
    }

    fn parse(&self, body: &str) -> Option<Value> {
        match self {
            Body::Text => Some(Value::String(body.to_string())),
            Body::Json(args) => parse_json(body.trim(), args),
            Body::Tags { regex, .. } => {
                let mut map = Map::new();
                for caps in regex.captures_iter(body) {
                    let caps = caps.ok()?;
                    let (key, value) = (caps.name("key")?, caps.name("value")?);
                    // A value that parses as JSON keeps its type.
                    let value = serde_json::from_str(value.as_str())
                        .unwrap_or_else(|_| Value::String(value.as_str().to_string()));
                    map.insert(key.as_str().to_string(), value);
                }
                Some(Value::Object(map))
            }
        }
    }
}

#[derive(Clone)]
struct BlockSpec {
    open: Opener,
    close: Closers,
    body: Body,
    transform: Option<Value>,
    outer: Option<[String; 2]>,
    call_prefix: Option<String>,
}

/// Compiled `response_template` from `tokenizer_config.json`.
#[derive(Clone)]
pub struct ResponseParser {
    thinking: Option<BlockSpec>,
    tool_calls: Option<BlockSpec>,
    content_close: Closers,
    tool_prompt: Option<String>,
}

impl ResponseParser {
    /// Parses `response_template` from `tokenizer_config.json` text.
    pub fn from_config(text: &str) -> Result<Option<Self>, Box<dyn Error + Send + Sync>> {
        let config: ConfigFile = serde_json::from_str(text)?;
        let Some(template) = config.response_template else {
            return Ok(None);
        };
        Self::compile(template).map(Some)
    }

    /// The first built-in format whose markers `chat` writes.
    pub fn builtin(chat: &Template) -> Result<Option<Self>, Box<dyn Error + Send + Sync>> {
        Ok(formats::pick(chat)?.map(|(_, parser)| parser))
    }

    /// Jinja that lists `tools` in the system prompt, if any.
    pub fn tool_prompt(&self) -> Option<&str> {
        self.tool_prompt.as_deref()
    }

    fn compile(template: ResponseTemplate) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let mut thinking = None;
        let mut tool_calls = None;
        let mut content_close = Closers::empty();
        for (name, spec) in template.fields {
            let raw_close: Vec<String> = spec
                .close
                .map(Strings::list)
                .unwrap_or_default()
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect();
            let close = Closers::compile(raw_close, spec.close_pattern)?;
            if name == "content" {
                content_close = close;
                continue;
            }
            let Some(open) = Opener::compile(spec.open, spec.open_pattern)? else {
                continue;
            };
            let block = BlockSpec {
                open,
                close,
                body: Body::compile(spec.content.as_deref(), spec.content_args)?,
                transform: spec.transform,
                outer: spec.outer.filter(|[o, c]| !o.is_empty() && !c.is_empty()),
                call_prefix: spec.call_prefix.filter(|s| !s.is_empty()),
            };
            match name.as_str() {
                "thinking" | "reasoning_content" => thinking = Some(block),
                "tool_calls" => tool_calls = Some(block),
                _ => {}
            }
        }
        Ok(Self {
            thinking,
            tool_calls,
            content_close,
            tool_prompt: template.tool_prompt,
        })
    }

    /// Whether this parser defines a tool-call block.
    pub fn supports_tools(&self) -> bool {
        self.tool_calls.is_some()
    }

    /// `messages` with tool calls written into `content` as model text.
    pub fn inline_calls(&self, messages: &[Message]) -> Vec<Message> {
        let mut out = messages.to_vec();
        let Some(BlockSpec {
            open: Opener::Literal(opens),
            close,
            body: Body::Json(_),
            transform: None,
            ..
        }) = &self.tool_calls
        else {
            return out;
        };
        let open = &opens[0];
        for m in &mut out {
            let Some(calls) = m.tool_calls.take() else {
                continue;
            };
            let mut text = m.content.take().unwrap_or_default();
            let mut plain = Vec::with_capacity(calls.len());
            for call in &calls {
                plain.push(
                    json!({"name": call.function.name, "arguments": call.function.arguments}),
                );
            }
            match close.consuming().into_iter().next() {
                Some(close) => {
                    for call in plain {
                        text += &format!("{open}{}{close}", spaced(&call));
                    }
                }
                None => text += &format!("{open}{}", spaced(&Value::Array(plain))),
            }
            m.content = Some(text);
        }
        out
    }

    /// Whether `prompt` already opened the thinking block.
    pub fn starts_thinking(&self, prompt: &str) -> bool {
        matches!(
            &self.thinking,
            Some(BlockSpec { open: Opener::Literal(opens), .. })
                if opens.iter().any(|s| {
                    prompt.ends_with(s.as_str())
                        || prompt.trim_end_matches('\n').ends_with(s.as_str())
                })
        )
    }

    /// Starts an incremental parser for one assistant reply.
    pub fn stream(&self, parse_tools: bool, in_thinking: bool) -> ResponseStream<'_> {
        let mode = match in_thinking && self.thinking.is_some() {
            true => Mode::Thinking,
            false => Mode::Text,
        };
        ResponseStream {
            parser: self,
            parse_tools: parse_tools && self.tool_calls.is_some(),
            mode,
            opening: false,
            calls: false,
            buf: String::new(),
            lead: false,
            tool: ToolState::Start,
            reasoning_tokens: 0,
            spans: Vec::new(),
        }
    }

    /// Parses a complete raw reply at once.
    pub fn parse(&self, text: &str, parse_tools: bool, in_thinking: bool) -> ParsedResponse {
        let mut stream = self.stream(parse_tools, in_thinking);
        let mut events = Vec::new();
        stream.push(text, &mut events);
        stream.finish(&mut events);
        let mut out = ParsedResponse::default();
        let mut open: Option<(Option<String>, String, String, String)> = None;
        for event in events {
            match event {
                ResponseEvent::Content(s) => out.content.push_str(&s),
                ResponseEvent::Reasoning(s) => out.reasoning.push_str(&s),
                ResponseEvent::ToolCall(tc) => out.tool_calls.push(tc),
                ResponseEvent::ToolCallStart { id, kind, name } => {
                    open = Some((id, kind, name, String::new()));
                }
                ResponseEvent::ToolCallArgs(frag) => {
                    if let Some((_, _, _, args)) = &mut open {
                        args.push_str(&frag);
                    }
                }
                ResponseEvent::ToolCallEnd => {
                    if let Some((id, kind, name, raw)) = open.take() {
                        let arguments = if raw.trim().is_empty() {
                            Value::Object(Map::new())
                        } else {
                            serde_json::from_str(&raw).unwrap_or(Value::String(raw))
                        };
                        out.tool_calls.push(ToolCall {
                            id,
                            kind,
                            function: FunctionCall { name, arguments },
                        });
                    }
                }
            }
        }
        out
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Text,
    Thinking,
    ToolCall,
    Calls,
    Closed,
}

#[derive(Clone, Copy)]
enum Hit {
    Thinking,
    ToolCall,
    Calls,
    Close,
}

#[derive(Clone, Copy)]
struct TokenSpan {
    start: usize,
    end: usize,
    counted: bool,
}

/// Incremental state machine for one assistant reply.
pub struct ResponseStream<'a> {
    parser: &'a ResponseParser,
    parse_tools: bool,
    mode: Mode,
    /// `buf` starts with an unstripped pattern opener.
    opening: bool,
    /// Inside an `outer` wrapper.
    calls: bool,
    buf: String,
    /// Drops newlines that open the content after a thinking block.
    lead: bool,
    /// Active incremental tool-call state.
    tool: ToolState,
    /// Confirmed reasoning tokens pushed so far.
    reasoning_tokens: usize,
    /// Buffered token byte ranges in `buf`.
    spans: Vec<TokenSpan>,
}

impl<'a> ResponseStream<'a> {
    /// Whether the stream is inside a thinking block.
    pub fn in_thinking(&self) -> bool {
        self.mode == Mode::Thinking
    }

    /// Number of pushed chunks that contributed to thinking blocks.
    pub fn reasoning_tokens(&self) -> usize {
        self.reasoning_tokens
    }

    /// Feeds raw decoded text and appends completed events.
    pub fn push(&mut self, piece: &str, out: &mut Vec<ResponseEvent>) {
        if self.mode == Mode::Closed {
            return;
        }
        if piece.is_empty() {
            if self.mode == Mode::Thinking {
                let at = self.buf.len();
                self.spans.push(TokenSpan {
                    start: at,
                    end: at,
                    counted: false,
                });
            }
            return;
        }
        let start = self.buf.len();
        self.buf.push_str(piece);
        self.spans.push(TokenSpan {
            start,
            end: self.buf.len(),
            counted: false,
        });
        self.drain(out, false);
    }

    /// Flushes remaining buffered text when generation ends.
    pub fn finish(&mut self, out: &mut Vec<ResponseEvent>) {
        if self.mode == Mode::Closed {
            self.buf.clear();
            self.spans.clear();
            return;
        }
        self.drain(out, true);
        if self.mode == Mode::Closed {
            self.buf.clear();
            self.spans.clear();
            return;
        }
        if self.mode == Mode::ToolCall
            && let Some(spec) = &self.parser.tool_calls
            && (self.tool.started() || !self.buf.is_empty())
        {
            let rest = std::mem::take(&mut self.buf);
            self.spans.clear();
            self.tool.finish(spec, &rest, out);
            return;
        }
        if self.buf.is_empty() {
            if self.mode == Mode::Thinking && !self.opening {
                self.count_all_spans();
            } else {
                self.spans.clear();
            }
            return;
        }
        let mut rest = std::mem::take(&mut self.buf);
        if std::mem::take(&mut self.opening) {
            match self.opener_len(&rest) {
                Some(cut) => {
                    self.shift_spans(cut, true);
                    drop(rest.drain(..cut));
                }
                None => {
                    self.spans.clear();
                    self.mode = Mode::Text;
                }
            }
            if rest.is_empty() {
                self.spans.clear();
                return;
            }
        }
        match self.mode {
            Mode::Thinking => {
                self.count_all_spans();
                out.push(ResponseEvent::Reasoning(rest));
            }
            Mode::Text | Mode::ToolCall | Mode::Calls => {
                self.spans.clear();
                out.push(ResponseEvent::Content(rest));
            }
            Mode::Closed => {}
        }
    }

    fn drain_buf(&mut self, cut: usize, reasoning: bool) {
        if cut == 0 {
            return;
        }
        self.buf.drain(..cut);
        self.shift_spans(cut, reasoning);
    }

    fn shift_spans(&mut self, cut: usize, reasoning: bool) {
        if cut == 0 {
            return;
        }
        let mut kept = 0;
        for i in 0..self.spans.len() {
            let mut span = self.spans[i];
            if reasoning && !span.counted && span.start < cut {
                span.counted = true;
                self.reasoning_tokens += 1;
            }
            if span.end > cut {
                span.start = span.start.saturating_sub(cut);
                span.end -= cut;
                self.spans[kept] = span;
                kept += 1;
            }
        }
        self.spans.truncate(kept);
    }

    fn count_all_spans(&mut self) {
        for span in &self.spans {
            if !span.counted {
                self.reasoning_tokens += 1;
            }
        }
        self.spans.clear();
    }

    fn drain(&mut self, out: &mut Vec<ResponseEvent>, eof: bool) {
        loop {
            let moved = match self.mode {
                Mode::Text => self.step_text(out, eof),
                Mode::Thinking => self.step_thinking(out, eof),
                Mode::ToolCall => self.step_tool(out, eof),
                Mode::Calls => self.step_calls(),
                Mode::Closed => false,
            };
            if !moved {
                break;
            }
        }
    }

    fn step_text(&mut self, out: &mut Vec<ResponseEvent>, eof: bool) -> bool {
        if self.lead {
            let kept = self.buf.trim_start_matches('\n').len();
            self.drain_buf(self.buf.len() - kept, false);
            if self.buf.is_empty() {
                return false;
            }
            self.lead = false;
        }
        let close_hold = if eof {
            0
        } else {
            self.parser.content_close.tail_hold(&self.buf)
        };
        let safe_end = self.buf.len() - close_hold;
        if let Some((at, skip, hit)) = self.find_hit(eof)
            && (eof || matches!(hit, Hit::Close) || at < safe_end)
        {
            if at > 0 {
                out.push(ResponseEvent::Content(self.buf[..at].to_string()));
                self.drain_buf(at, false);
            }
            self.drain_buf(skip, matches!(hit, Hit::Thinking));
            self.mode = match hit {
                Hit::Thinking => {
                    let thinking = self.parser.thinking.as_ref();
                    self.opening =
                        thinking.is_some_and(|t| matches!(t.open, Opener::Pattern { .. }));
                    Mode::Thinking
                }
                Hit::ToolCall => Mode::ToolCall,
                Hit::Calls => {
                    self.calls = true;
                    Mode::Calls
                }
                Hit::Close => {
                    self.buf.clear();
                    self.spans.clear();
                    Mode::Closed
                }
            };
            return self.mode != Mode::Closed;
        }
        if eof {
            return false;
        }
        let mut needles: Vec<&str> = Vec::new();
        if let Some(t) = &self.parser.thinking {
            needles.extend(t.open.triggers());
        }
        if self.parse_tools
            && let Some(t) = &self.parser.tool_calls
        {
            needles.extend(t.open.triggers());
            needles.extend(t.outer.as_ref().map(|[o, _]| o.as_str()));
        }
        let keep = tail_hold(&self.buf, &needles).max(close_hold);
        let emit = self.buf.len() - keep;
        if emit > 0 {
            out.push(ResponseEvent::Content(self.buf[..emit].to_string()));
            self.drain_buf(emit, false);
        }
        false
    }

    fn step_thinking(&mut self, out: &mut Vec<ResponseEvent>, eof: bool) -> bool {
        let Some(spec) = &self.parser.thinking else {
            self.mode = Mode::Text;
            return true;
        };
        if self.opening {
            let from = spec.open.min_close_offset(&self.buf);
            let close = spec.close.find_from_for(&self.buf, from, eof);
            match (self.opener_len(&self.buf), close) {
                // Text after the match shows that the opener is complete.
                (Some(cut), _) if cut < self.buf.len() || eof => {
                    self.drain_buf(cut, true);
                    self.opening = false;
                }
                (None, Some((at, len))) => {
                    out.push(ResponseEvent::Content(self.buf[..at + len].to_string()));
                    self.drain_buf(at + len, false);
                    self.opening = false;
                    self.mode = Mode::Text;
                    return true;
                }
                _ => return false,
            }
        }
        if let Some((at, len)) = spec.close.find_for(&self.buf, eof) {
            if at > 0 {
                out.push(ResponseEvent::Reasoning(self.buf[..at].to_string()));
            }
            self.drain_buf(at + len, true);
            self.mode = Mode::Text;
            self.lead = true;
            return true;
        }
        if eof {
            return false;
        }
        let keep = spec.close.tail_hold(&self.buf);
        let emit = self.buf.len() - keep;
        if emit > 0 {
            out.push(ResponseEvent::Reasoning(self.buf[..emit].to_string()));
            self.drain_buf(emit, true);
        }
        false
    }

    fn step_tool(&mut self, out: &mut Vec<ResponseEvent>, eof: bool) -> bool {
        let Some(spec) = &self.parser.tool_calls else {
            self.mode = Mode::Text;
            return true;
        };
        let before = self.buf.len();
        let outcome = self.tool.step(spec, &mut self.buf, out, eof);
        self.shift_spans(before - self.buf.len(), false);
        match outcome {
            StepOutcome::Wait => false,
            StepOutcome::Continue => true,
            StepOutcome::Done => {
                self.tool = ToolState::Start;
                self.mode = match self.calls {
                    true => Mode::Calls,
                    false => Mode::Text,
                };
                true
            }
        }
    }

    fn step_calls(&mut self) -> bool {
        let Some(spec) = &self.parser.tool_calls else {
            self.mode = Mode::Text;
            return true;
        };
        let open = spec.open.find(&self.buf).map(|(at, _)| at);
        let close = spec
            .outer
            .as_ref()
            .and_then(|[_, c]| Some((self.buf.find(c.as_str())?, c.len())));
        match (open, close) {
            (Some(at), c) if c.is_none_or(|(end, _)| at < end) => {
                self.drain_buf(at, false);
                self.mode = Mode::ToolCall;
            }
            (_, Some((at, len))) => {
                self.drain_buf(at + len, false);
                self.calls = false;
                self.mode = Mode::Text;
            }
            _ => return false,
        }
        true
    }

    /// Byte length of the thinking opener that starts `text`.
    fn opener_len(&self, text: &str) -> Option<usize> {
        let spec = self.parser.thinking.as_ref()?;
        let (_, rest) = spec.open.strip(text)?;
        Some(text.len() - rest.len())
    }

    fn find_hit(&self, eof: bool) -> Option<(usize, usize, Hit)> {
        let mut best: Option<(usize, usize, Hit)> = None;
        let mut pick = |at: usize, skip: usize, hit: Hit| {
            if best.is_none_or(|(b, _, _)| at < b) {
                best = Some((at, skip, hit));
            }
        };
        if let Some(t) = &self.parser.thinking
            && let Some((at, skip)) = t.open.find(&self.buf)
        {
            pick(at, skip, Hit::Thinking);
        }
        if self.parse_tools
            && let Some(t) = &self.parser.tool_calls
            && let Some((at, _)) = t.open.find(&self.buf)
        {
            // Keep opener bytes for matching when the block closes.
            pick(at, 0, Hit::ToolCall);
        }
        if self.parse_tools
            && let Some([open, _]) = self
                .parser
                .tool_calls
                .as_ref()
                .and_then(|t| t.outer.as_ref())
            && let Some(at) = self.buf.find(open.as_str())
        {
            pick(at, open.len(), Hit::Calls);
        }
        if let Some((at, len)) = self.parser.content_close.find_for(&self.buf, eof) {
            pick(at, len, Hit::Close);
        }
        best
    }
}

fn find_literal(text: &str, literals: &[String]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for s in literals {
        if let Some(at) = text.find(s.as_str())
            && best.is_none_or(|(b, _)| at < b)
        {
            best = Some((at, s.len()));
        }
    }
    best
}

/// Longest suffix of `buf` that prefixes any trigger.
fn tail_hold(buf: &str, needles: &[&str]) -> usize {
    let mut hold = 0;
    for needle in needles {
        let limit = buf.len().min(needle.len().saturating_sub(1));
        for len in (1..=limit).rev() {
            let start = buf.len() - len;
            if buf.is_char_boundary(start) && needle.starts_with(&buf[start..]) {
                hold = hold.max(len);
                break;
            }
        }
    }
    hold
}

fn parse_tool(spec: &BlockSpec, block: &str) -> Option<Vec<ToolCall>> {
    let (captures, rest) = spec.open.strip(block)?;
    let parsed = spec.body.parse(rest)?;
    let shaped = match &spec.transform {
        Some(tmpl) => apply_transform(tmpl, &captures, &parsed),
        None => parsed,
    };
    match shaped {
        Value::Array(items) => items.into_iter().map(to_call).collect(),
        one => to_call(one).map(|call| vec![call]),
    }
}

fn apply_transform(tmpl: &Value, captures: &BTreeMap<String, String>, content: &Value) -> Value {
    match tmpl {
        Value::String(s) if s == "{content}" => content.clone(),
        Value::String(s) => {
            if let Some(key) = s.strip_prefix('{').and_then(|v| v.strip_suffix('}'))
                && let Some(val) = captures.get(key)
            {
                return Value::String(val.clone());
            }
            Value::String(s.clone())
        }
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(apply_transform(item, captures, content));
            }
            Value::Array(out)
        }
        Value::Object(map) => {
            let mut out = Map::new();
            for (k, v) in map {
                out.insert(k.clone(), apply_transform(v, captures, content));
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

fn to_call(value: Value) -> Option<ToolCall> {
    let Value::Object(mut obj) = value else {
        return None;
    };
    let id = obj.remove("id").and_then(|v| match v {
        Value::String(s) => Some(s),
        _ => None,
    });
    let kind = obj
        .remove("type")
        .and_then(|v| match v {
            Value::String(s) => Some(s),
            _ => None,
        })
        .unwrap_or_else(|| "function".into());
    let func = match obj.remove("function") {
        Some(Value::Object(inner)) => inner,
        None => obj,
        _ => return None,
    };
    let name = func.get("name")?.as_str()?.to_string();
    let arguments = match func.get("arguments") {
        Some(Value::String(s)) => {
            serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
        }
        Some(Value::Null) | None => Value::Object(Map::new()),
        Some(other) => other.clone(),
    };
    Some(ToolCall {
        id,
        kind,
        function: FunctionCall { name, arguments },
    })
}
