//! Relaxed JSON parser and incremental streaming normalizer.

use super::super::close::Closers;
use super::super::{BlockSpec, Body, ContentArgs, ResponseEvent};
use super::{StepOutcome, json_escape_inner, json_str, safe_value_emit};
use serde_json::{Map, Value};

/// Parses JSON with optional custom string delimiters and bare keys.
pub(in crate::response) fn parse_json(input: &str, args: &ContentArgs) -> Option<Value> {
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

/// Incremental normalizer state for delimited or bare-key JSON.
#[derive(Default)]
pub(in crate::response) struct RelaxedState {
    stack: Vec<bool>,
    expect_key: bool,
    pending_comma: bool,
    mode: RelaxedMode,
}

#[derive(Default)]
enum RelaxedMode {
    #[default]
    Normal,
    InDelim(usize),
    InQuoted {
        escaped: bool,
    },
}

/// Flushes a truncated relaxed-JSON call on EOF.
pub(super) fn finish_relaxed(state: &RelaxedState, rest: &str, out: &mut Vec<ResponseEvent>) {
    if let RelaxedMode::InDelim(_) = state.mode
        && !rest.is_empty()
    {
        out.push(ResponseEvent::ToolCallArgs(json_escape_inner(rest)));
    }
    out.push(ResponseEvent::ToolCallEnd);
}

/// Normalizes relaxed JSON bytes from `buf` into standard JSON chunks.
pub(super) fn step_relaxed(
    spec: &BlockSpec,
    state: &mut RelaxedState,
    buf: &mut String,
    out: &mut Vec<ResponseEvent>,
    eof: bool,
) -> StepOutcome {
    let Body::Json(args) = &spec.body else {
        unreachable!()
    };
    let mut emitted = String::new();
    let done = loop {
        match state.mode {
            RelaxedMode::InDelim(idx) => {
                let close_delim = &args.string_delims[idx][1];
                if let Some(at) = buf.find(close_delim.as_str()) {
                    emitted.push_str(&json_escape_inner(&buf[..at]));
                    emitted.push('"');
                    buf.drain(..at + close_delim.len());
                    state.mode = RelaxedMode::Normal;
                    continue;
                }
                if let Some((at, len)) = spec.close.find_for(buf, eof) {
                    emitted.push_str(&json_escape_inner(&buf[..at]));
                    emitted.push('"');
                    buf.drain(..at + len);
                    break true;
                }
                let emit = safe_value_emit(buf, &[close_delim.as_str()], &spec.close, false);
                if emit > 0 {
                    emitted.push_str(&json_escape_inner(&buf[..emit]));
                    buf.drain(..emit);
                }
                break false;
            }
            RelaxedMode::InQuoted { mut escaped } => {
                let bytes = buf.as_bytes();
                let mut closed = None;
                for (i, &b) in bytes.iter().enumerate() {
                    if escaped {
                        escaped = false;
                    } else if b == b'\\' {
                        escaped = true;
                    } else if b == b'"' {
                        closed = Some(i + 1);
                        break;
                    }
                }
                if let Some(cut) = closed {
                    emitted.push_str(&buf[..cut]);
                    buf.drain(..cut);
                    state.mode = RelaxedMode::Normal;
                    continue;
                }
                if let Some((at, len)) = spec.close.find_for(buf, eof) {
                    emitted.push_str(&buf[..at]);
                    buf.drain(..at + len);
                    break true;
                }
                let emit = buf.len() - spec.close.tail_hold(buf);
                if emit > 0 {
                    let mut esc = matches!(state.mode, RelaxedMode::InQuoted { escaped: true });
                    for &b in &buf.as_bytes()[..emit] {
                        if esc {
                            esc = false;
                        } else if b == b'\\' {
                            esc = true;
                        }
                    }
                    emitted.push_str(&buf[..emit]);
                    buf.drain(..emit);
                    state.mode = RelaxedMode::InQuoted { escaped: esc };
                }
                break false;
            }
            RelaxedMode::Normal => {
                let trimmed = buf.len() - buf.trim_start().len();
                if trimmed > 0 {
                    buf.drain(..trimmed);
                }
                if buf.is_empty() {
                    break false;
                }
                if let Some((0, len)) = spec.close.find_for(buf, eof) {
                    buf.drain(..len);
                    break true;
                }
                if let Some(idx) = match_delim_open(buf, args) {
                    flush_comma(state, &mut emitted);
                    emitted.push('"');
                    buf.drain(..args.string_delims[idx][0].len());
                    state.mode = RelaxedMode::InDelim(idx);
                    continue;
                }
                if !eof && prefix_of_any(buf, args, &spec.close) {
                    break false;
                }
                let first = buf.as_bytes()[0];
                match first {
                    b'{' => {
                        flush_comma(state, &mut emitted);
                        emitted.push('{');
                        buf.drain(..1);
                        state.stack.push(true);
                        state.expect_key = true;
                    }
                    b'[' => {
                        flush_comma(state, &mut emitted);
                        emitted.push('[');
                        buf.drain(..1);
                        state.stack.push(false);
                        state.expect_key = false;
                    }
                    b'}' => {
                        state.pending_comma = false;
                        state.expect_key = false;
                        state.stack.pop();
                        emitted.push('}');
                        buf.drain(..1);
                    }
                    b']' => {
                        state.pending_comma = false;
                        state.expect_key = false;
                        state.stack.pop();
                        emitted.push(']');
                        buf.drain(..1);
                    }
                    b',' => {
                        state.pending_comma = true;
                        state.expect_key = state.stack.last().copied().unwrap_or(false);
                        buf.drain(..1);
                    }
                    b':' => {
                        state.expect_key = false;
                        emitted.push(':');
                        buf.drain(..1);
                    }
                    b'"' => {
                        flush_comma(state, &mut emitted);
                        emitted.push('"');
                        buf.drain(..1);
                        state.mode = RelaxedMode::InQuoted { escaped: false };
                    }
                    _ if state.expect_key && args.unquoted_keys => {
                        let Some(colon) = buf.find(':') else {
                            break false;
                        };
                        let key = buf[..colon].trim();
                        flush_comma(state, &mut emitted);
                        emitted.push_str(&json_str(key));
                        emitted.push(':');
                        buf.drain(..colon + 1);
                        state.expect_key = false;
                    }
                    _ => {
                        let len = buf.bytes().take_while(|b| is_prim_byte(*b)).count();
                        if len == 0 {
                            if let Some((at, clen)) = spec.close.find_for(buf, eof) {
                                buf.drain(..at + clen);
                                break true;
                            }
                            break false;
                        }
                        flush_comma(state, &mut emitted);
                        emitted.push_str(&buf[..len]);
                        buf.drain(..len);
                    }
                }
            }
        }
    };
    if !emitted.is_empty() {
        out.push(ResponseEvent::ToolCallArgs(emitted));
    }
    if done {
        out.push(ResponseEvent::ToolCallEnd);
        StepOutcome::Done
    } else {
        StepOutcome::Wait
    }
}

fn flush_comma(state: &mut RelaxedState, out: &mut String) {
    if state.pending_comma {
        out.push(',');
        state.pending_comma = false;
    }
}

fn match_delim_open(buf: &str, args: &ContentArgs) -> Option<usize> {
    args.string_delims
        .iter()
        .position(|[open, _]| buf.starts_with(open.as_str()))
}

fn prefix_of_any(buf: &str, args: &ContentArgs, closes: &Closers) -> bool {
    let any_delim = args
        .string_delims
        .iter()
        .any(|[open, _]| open.starts_with(buf));
    let any_close = !buf.is_empty() && closes.tail_hold(buf) == buf.len();
    any_delim || any_close
}

fn is_prim_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'-' | b'+' | b'.')
}
