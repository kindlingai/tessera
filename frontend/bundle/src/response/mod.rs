//! Generic response parser driven by `tokenizer_config.json`.

#[cfg(test)]
mod response_test;

use crate::template::{FunctionCall, ToolCall};
use fancy_regex::Regex;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

/// One parsed piece from a model response.
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseEvent {
    Content(String),
    Reasoning(String),
    ToolCall(ToolCall),
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
    pattern: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct FieldSpec {
    open: Option<String>,
    open_pattern: Option<String>,
    close: Option<Strings>,
    content: Option<String>,
    #[serde(default)]
    content_args: ContentArgs,
    transform: Option<Value>,
}

#[derive(Clone, Debug, Deserialize)]
struct ResponseTemplate {
    #[serde(default)]
    fields: BTreeMap<String, FieldSpec>,
}

#[derive(Deserialize)]
struct ConfigFile {
    response_template: Option<ResponseTemplate>,
}

#[derive(Clone)]
enum Opener {
    Literal(String),
    Pattern { prefix: String, regex: Regex },
}

impl Opener {
    fn compile(
        open: Option<String>,
        pattern: Option<String>,
    ) -> Result<Option<Self>, Box<dyn Error + Send + Sync>> {
        if let Some(pat) = pattern {
            let prefix = literal_prefix(&pat);
            let regex = Regex::new(&pat)?;
            return Ok(Some(Opener::Pattern { prefix, regex }));
        }
        Ok(open.filter(|s| !s.is_empty()).map(Opener::Literal))
    }

    fn trigger(&self) -> Option<&str> {
        match self {
            Opener::Literal(s) => Some(s.as_str()),
            Opener::Pattern { prefix, .. } if !prefix.is_empty() => Some(prefix.as_str()),
            Opener::Pattern { .. } => None,
        }
    }

    fn find(&self, text: &str) -> Option<(usize, usize)> {
        match self {
            Opener::Literal(s) => text.find(s.as_str()).map(|at| (at, s.len())),
            Opener::Pattern { prefix, .. } if !prefix.is_empty() => {
                text.find(prefix.as_str()).map(|at| (at, 0))
            }
            Opener::Pattern { regex, .. } => {
                regex.find(text).ok().flatten().map(|m| (m.start(), 0))
            }
        }
    }

    /// Strips the opener at byte zero and returns captures.
    fn strip<'a>(&self, block: &'a str) -> Option<(BTreeMap<String, String>, &'a str)> {
        match self {
            Opener::Literal(s) => Some((BTreeMap::new(), block.strip_prefix(s.as_str())?)),
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
    Tags(Regex),
}

impl Body {
    fn compile(
        kind: Option<&str>,
        args: ContentArgs,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        Ok(match kind {
            Some("json") => Body::Json(args),
            Some("tags") => {
                let pat = args
                    .pattern
                    .ok_or("tags content needs content_args.pattern")?;
                Body::Tags(Regex::new(&pat)?)
            }
            _ => Body::Text,
        })
    }

    fn parse(&self, body: &str) -> Option<Value> {
        match self {
            Body::Text => Some(Value::String(body.to_string())),
            Body::Json(args) => parse_json(body.trim(), args),
            Body::Tags(regex) => {
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
    close: Vec<String>,
    body: Body,
    transform: Option<Value>,
}

/// Compiled `response_template` from `tokenizer_config.json`.
#[derive(Clone)]
pub struct ResponseParser {
    thinking: Option<BlockSpec>,
    tool_calls: Option<BlockSpec>,
    content_close: Vec<String>,
}

impl ResponseParser {
    /// Loads `response_template` from `dir/tokenizer_config.json`, if present.
    pub fn load(dir: &Path) -> Result<Option<Self>, Box<dyn Error + Send + Sync>> {
        let text = match std::fs::read_to_string(dir.join("tokenizer_config.json")) {
            Ok(text) => text,
            Err(_) => return Ok(None),
        };
        Self::from_config(&text)
    }

    /// Parses `response_template` from `tokenizer_config.json` text.
    pub fn from_config(text: &str) -> Result<Option<Self>, Box<dyn Error + Send + Sync>> {
        let config: ConfigFile = serde_json::from_str(text)?;
        let Some(template) = config.response_template else {
            return Ok(None);
        };
        Self::compile(template).map(Some)
    }

    fn compile(template: ResponseTemplate) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let mut thinking = None;
        let mut tool_calls = None;
        let mut content_close = Vec::new();
        for (name, spec) in template.fields {
            let close: Vec<String> = spec
                .close
                .map(Strings::list)
                .unwrap_or_default()
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect();
            if name == "content" {
                content_close = close;
                continue;
            }
            let Some(open) = Opener::compile(spec.open, spec.open_pattern)? else {
                continue;
            };
            if close.is_empty() {
                continue;
            }
            let block = BlockSpec {
                open,
                close,
                body: Body::compile(spec.content.as_deref(), spec.content_args)?,
                transform: spec.transform,
            };
            match name.as_str() {
                "thinking" => thinking = Some(block),
                "tool_calls" => tool_calls = Some(block),
                _ => {}
            }
        }
        Ok(Self {
            thinking,
            tool_calls,
            content_close,
        })
    }

    /// Whether this parser defines a tool-call block.
    pub fn supports_tools(&self) -> bool {
        self.tool_calls.is_some()
    }

    /// Whether `prompt` already opened the thinking block.
    pub fn starts_thinking(&self, prompt: &str) -> bool {
        matches!(
            &self.thinking,
            Some(BlockSpec { open: Opener::Literal(s), .. }) if prompt.ends_with(s.as_str())
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
            buf: String::new(),
        }
    }

    /// Parses a complete raw reply at once.
    pub fn parse(&self, text: &str, parse_tools: bool, in_thinking: bool) -> ParsedResponse {
        let mut stream = self.stream(parse_tools, in_thinking);
        let mut events = Vec::new();
        stream.push(text, &mut events);
        stream.finish(&mut events);
        let mut out = ParsedResponse::default();
        for event in events {
            match event {
                ResponseEvent::Content(s) => out.content.push_str(&s),
                ResponseEvent::Reasoning(s) => out.reasoning.push_str(&s),
                ResponseEvent::ToolCall(tc) => out.tool_calls.push(tc),
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
    Closed,
}

#[derive(Clone, Copy)]
enum Hit {
    Thinking,
    ToolCall,
    Close,
}

/// Incremental state machine for one assistant reply.
pub struct ResponseStream<'a> {
    parser: &'a ResponseParser,
    parse_tools: bool,
    mode: Mode,
    /// `buf` starts with an unstripped pattern opener.
    opening: bool,
    buf: String,
}

impl<'a> ResponseStream<'a> {
    /// Feeds raw decoded text and appends completed events.
    pub fn push(&mut self, piece: &str, out: &mut Vec<ResponseEvent>) {
        if self.mode == Mode::Closed || piece.is_empty() {
            return;
        }
        self.buf.push_str(piece);
        loop {
            let moved = match self.mode {
                Mode::Text => self.step_text(out),
                Mode::Thinking => self.step_thinking(out),
                Mode::ToolCall => self.step_tool(out),
                Mode::Closed => false,
            };
            if !moved {
                break;
            }
        }
    }

    /// Flushes remaining buffered text when generation ends.
    pub fn finish(&mut self, out: &mut Vec<ResponseEvent>) {
        if self.buf.is_empty() || self.mode == Mode::Closed {
            self.buf.clear();
            return;
        }
        let mut rest = std::mem::take(&mut self.buf);
        if std::mem::take(&mut self.opening) {
            match self.opener_len(&rest) {
                Some(cut) => drop(rest.drain(..cut)),
                None => self.mode = Mode::Text,
            }
            if rest.is_empty() {
                return;
            }
        }
        match self.mode {
            Mode::Thinking => out.push(ResponseEvent::Reasoning(rest)),
            Mode::Text | Mode::ToolCall => out.push(ResponseEvent::Content(rest)),
            Mode::Closed => {}
        }
    }

    fn step_text(&mut self, out: &mut Vec<ResponseEvent>) -> bool {
        if let Some((at, skip, hit)) = self.find_hit() {
            if at > 0 {
                out.push(ResponseEvent::Content(self.buf[..at].to_string()));
            }
            self.buf.drain(..at + skip);
            self.mode = match hit {
                Hit::Thinking => {
                    let thinking = self.parser.thinking.as_ref();
                    self.opening =
                        thinking.is_some_and(|t| matches!(t.open, Opener::Pattern { .. }));
                    Mode::Thinking
                }
                Hit::ToolCall => Mode::ToolCall,
                Hit::Close => {
                    self.buf.clear();
                    Mode::Closed
                }
            };
            return self.mode != Mode::Closed;
        }
        let mut needles: Vec<&str> = Vec::new();
        if let Some(t) = &self.parser.thinking
            && let Some(s) = t.open.trigger()
        {
            needles.push(s);
        }
        if self.parse_tools
            && let Some(t) = &self.parser.tool_calls
            && let Some(s) = t.open.trigger()
        {
            needles.push(s);
        }
        for c in &self.parser.content_close {
            needles.push(c.as_str());
        }
        let keep = tail_hold(&self.buf, &needles);
        let emit = self.buf.len() - keep;
        if emit > 0 {
            out.push(ResponseEvent::Content(self.buf[..emit].to_string()));
            self.buf.drain(..emit);
        }
        false
    }

    fn step_thinking(&mut self, out: &mut Vec<ResponseEvent>) -> bool {
        let Some(spec) = &self.parser.thinking else {
            self.mode = Mode::Text;
            return true;
        };
        if self.opening {
            let close = find_close(&self.buf, &spec.close);
            match (self.opener_len(&self.buf), close) {
                // Text after the match shows that the opener is complete.
                (Some(cut), _) if cut < self.buf.len() => {
                    self.buf.drain(..cut);
                    self.opening = false;
                }
                (None, Some((at, len))) => {
                    out.push(ResponseEvent::Content(self.buf[..at + len].to_string()));
                    self.buf.drain(..at + len);
                    self.opening = false;
                    self.mode = Mode::Text;
                    return true;
                }
                _ => return false,
            }
        }
        if let Some((at, len)) = find_close(&self.buf, &spec.close) {
            if at > 0 {
                out.push(ResponseEvent::Reasoning(self.buf[..at].to_string()));
            }
            self.buf.drain(..at + len);
            self.mode = Mode::Text;
            return true;
        }
        let needles: Vec<&str> = spec.close.iter().map(String::as_str).collect();
        let keep = tail_hold(&self.buf, &needles);
        let emit = self.buf.len() - keep;
        if emit > 0 {
            out.push(ResponseEvent::Reasoning(self.buf[..emit].to_string()));
            self.buf.drain(..emit);
        }
        false
    }

    fn step_tool(&mut self, out: &mut Vec<ResponseEvent>) -> bool {
        let Some(spec) = &self.parser.tool_calls else {
            self.mode = Mode::Text;
            return true;
        };
        let Some((at, len)) = find_close(&self.buf, &spec.close) else {
            return false;
        };
        let block = self.buf[..at].to_string();
        let raw_end = at + len;
        let event = match parse_tool(spec, &block) {
            Some(call) => ResponseEvent::ToolCall(call),
            None => ResponseEvent::Content(self.buf[..raw_end].to_string()),
        };
        out.push(event);
        self.buf.drain(..raw_end);
        self.mode = Mode::Text;
        true
    }

    /// Byte length of the thinking opener that starts `text`.
    fn opener_len(&self, text: &str) -> Option<usize> {
        let spec = self.parser.thinking.as_ref()?;
        let (_, rest) = spec.open.strip(text)?;
        Some(text.len() - rest.len())
    }

    fn find_hit(&self) -> Option<(usize, usize, Hit)> {
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
        if let Some((at, len)) = find_close(&self.buf, &self.parser.content_close) {
            pick(at, len, Hit::Close);
        }
        best
    }
}

fn find_close(text: &str, closes: &[String]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for c in closes {
        if let Some(at) = text.find(c.as_str())
            && best.is_none_or(|(b, _)| at < b)
        {
            best = Some((at, c.len()));
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

/// Extracts the leading literal text from a regex pattern.
fn literal_prefix(pattern: &str) -> String {
    let mut out = String::new();
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            let Some(&next) = chars.peek() else { break };
            if matches!(
                next,
                'w' | 'W' | 'd' | 'D' | 's' | 'S' | 'b' | 'B' | 'A' | 'z' | 'Z'
            ) {
                break;
            }
            let escaped = match next {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                other if other.is_ascii_punctuation() => other,
                _ => break,
            };
            chars.next();
            out.push(escaped);
            continue;
        }
        if matches!(
            c,
            '(' | ')' | '[' | ']' | '{' | '}' | '.' | '*' | '+' | '?' | '|' | '^' | '$'
        ) {
            break;
        }
        out.push(c);
    }
    out
}

fn parse_tool(spec: &BlockSpec, block: &str) -> Option<ToolCall> {
    let (captures, rest) = spec.open.strip(block)?;
    let parsed = spec.body.parse(rest)?;
    let shaped = match &spec.transform {
        Some(tmpl) => apply_transform(tmpl, &captures, &parsed),
        None => parsed,
    };
    to_call(shaped)
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

/// Parses JSON with optional custom string delimiters and bare keys.
fn parse_json(input: &str, args: &ContentArgs) -> Option<Value> {
    if args.string_delims.is_empty() && !args.unquoted_keys {
        return serde_json::from_str(input).ok();
    }
    let mut cursor = Cursor {
        src: input,
        pos: 0,
        args,
    };
    let value = cursor.value()?;
    cursor.skip_ws();
    cursor.rest().is_empty().then_some(value)
}

struct Cursor<'a> {
    src: &'a str,
    pos: usize,
    args: &'a ContentArgs,
}

impl<'a> Cursor<'a> {
    fn rest(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn skip_ws(&mut self) {
        let trimmed = self.rest().trim_start();
        self.pos = self.src.len() - trimmed.len();
    }

    fn eat(&mut self, ch: u8) -> bool {
        if self.rest().as_bytes().first() == Some(&ch) {
            self.pos += 1;
            return true;
        }
        false
    }

    fn value(&mut self) -> Option<Value> {
        self.skip_ws();
        if let Some(s) = self.delim_string() {
            return Some(Value::String(s));
        }
        let &first = self.rest().as_bytes().first()?;
        match first {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => self.quoted().map(Value::String),
            b't' if self.rest().starts_with("true") => {
                self.pos += 4;
                Some(Value::Bool(true))
            }
            b'f' if self.rest().starts_with("false") => {
                self.pos += 5;
                Some(Value::Bool(false))
            }
            b'n' if self.rest().starts_with("null") => {
                self.pos += 4;
                Some(Value::Null)
            }
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    fn delim_string(&mut self) -> Option<String> {
        for [open, close] in &self.args.string_delims {
            if let Some(after) = self.rest().strip_prefix(open.as_str()) {
                let end = after.find(close.as_str())?;
                let text = after[..end].to_string();
                self.pos += open.len() + end + close.len();
                return Some(text);
            }
        }
        None
    }

    fn quoted(&mut self) -> Option<String> {
        let bytes = self.rest().as_bytes();
        if bytes.first() != Some(&b'"') {
            return None;
        }
        let mut i = 1;
        while i < bytes.len() {
            match bytes[i] {
                b'\\' => i += 2,
                b'"' => {
                    let text: String = serde_json::from_str(&self.rest()[..=i]).ok()?;
                    self.pos += i + 1;
                    return Some(text);
                }
                _ => i += 1,
            }
        }
        None
    }

    fn key(&mut self) -> Option<String> {
        self.skip_ws();
        if let Some(s) = self.delim_string() {
            return Some(s);
        }
        if self.rest().starts_with('"') {
            return self.quoted();
        }
        if !self.args.unquoted_keys {
            return None;
        }
        let bytes = self.rest().as_bytes();
        let mut end = 0;
        while end < bytes.len() {
            match bytes[end] {
                b':' => break,
                b'{' | b'}' | b'[' | b']' | b',' => return None,
                _ => end += 1,
            }
        }
        let key = self.rest()[..end].trim();
        if key.is_empty() {
            return None;
        }
        let out = key.to_string();
        self.pos += end;
        Some(out)
    }

    fn object(&mut self) -> Option<Value> {
        if !self.eat(b'{') {
            return None;
        }
        let mut map = Map::new();
        self.skip_ws();
        if self.eat(b'}') {
            return Some(Value::Object(map));
        }
        loop {
            let k = self.key()?;
            self.skip_ws();
            if !self.eat(b':') {
                return None;
            }
            let v = self.value()?;
            map.insert(k, v);
            self.skip_ws();
            if self.eat(b',') {
                self.skip_ws();
                if self.eat(b'}') {
                    return Some(Value::Object(map));
                }
                continue;
            }
            if self.eat(b'}') {
                return Some(Value::Object(map));
            }
            return None;
        }
    }

    fn array(&mut self) -> Option<Value> {
        if !self.eat(b'[') {
            return None;
        }
        let mut items = Vec::new();
        self.skip_ws();
        if self.eat(b']') {
            return Some(Value::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_ws();
            if self.eat(b',') {
                self.skip_ws();
                if self.eat(b']') {
                    return Some(Value::Array(items));
                }
                continue;
            }
            if self.eat(b']') {
                return Some(Value::Array(items));
            }
            return None;
        }
    }

    fn number(&mut self) -> Option<Value> {
        let bytes = self.rest().as_bytes();
        let mut end = 0;
        while end < bytes.len()
            && matches!(bytes[end], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
        {
            end += 1;
        }
        let num: serde_json::Number = serde_json::from_str(&self.rest()[..end]).ok()?;
        self.pos += end;
        Some(Value::Number(num))
    }
}
