//! Incremental tool-call argument streaming for response blocks.

mod json;
#[cfg(test)]
pub(super) mod stream_test;

pub(super) use json::parse_json;

use super::close::{
    Closer, Closers, find_closers, find_closers_for, matching_paren, parse_literal_closers,
    unescape_lit,
};
use super::{BlockSpec, Body, Opener, ResponseEvent, parse_tool, tail_hold};
use json::{RelaxedState, finish_relaxed, step_relaxed};
use serde_json::Value;
use std::collections::BTreeMap;

/// Literal tag boundaries extracted from a `tags` body regex.
#[derive(Clone, Debug)]
pub(super) struct TagDelims {
    pub key_open: String,
    pub key_close: String,
    pub val_open: String,
    pub val_close: Vec<Closer>,
    pub trim_lead_nl: bool,
    pub trim_trail_nl: bool,
}

impl TagDelims {
    /// Extracts literal delimiters from a `key`/`value` regex pattern.
    pub(super) fn from_pattern(pattern: &str) -> Option<Self> {
        let rest = strip_inline_flags(pattern);
        let key_at = rest.find("(?P<key>")?;
        let key_open = unescape_lit(&rest[..key_at])?;
        let key_end = matching_paren(rest, key_at)?;
        let after_key = &rest[key_end + 1..];
        let val_at = after_key.find("(?P<value>")?;
        let mid = &after_key[..val_at];
        let val_end = matching_paren(after_key, val_at)?;
        let after_val = &after_key[val_end + 1..];
        let (key_close, val_open, trim_lead_nl) = match mid.split_once("\\s*") {
            Some((kc, vo)) => {
                let (vo, nl) = strip_opt_nl_end(vo);
                (unescape_lit(kc)?, unescape_lit(vo)?, nl)
            }
            None => {
                let (kc, nl) = strip_opt_nl_end(mid);
                (unescape_lit(kc)?, String::new(), nl)
            }
        };
        let (vc, trim_trail_nl) = strip_opt_nl_start(after_val);
        let val_close = parse_literal_closers(vc)?;
        if key_open.is_empty() || key_close.is_empty() || val_close.is_empty() {
            return None;
        }
        Some(Self {
            key_open,
            key_close,
            val_open,
            val_close,
            trim_lead_nl,
            trim_trail_nl,
        })
    }

    fn val_needles(&self) -> Vec<&str> {
        self.val_close.iter().map(|c| c.needle.as_str()).collect()
    }
}

/// Incremental parser state while inside `Mode::ToolCall`.
#[derive(Default)]
pub(super) enum ToolState {
    #[default]
    Start,
    /// Falls back to whole-block parsing on close.
    Buffer,
    /// Streams raw JSON body until the close tag.
    RawJson,
    /// Streams relaxed JSON with custom string delimiters or bare keys.
    RelaxedJson(RelaxedState),
    /// Streams tag-based key/value pairs as a JSON object.
    Tags(TagPhase),
    /// Streams the `arguments` JSON object from a literal-opened wrapper.
    JsonArgs {
        depth: usize,
        in_str: bool,
        escaped: bool,
        started: bool,
    },
    /// Consumes trailing wrapper bytes up to the block close tag.
    DrainClose,
}

pub(super) enum TagPhase {
    SeekKey {
        first: bool,
    },
    ReadKey {
        first: bool,
    },
    SeekValOpen {
        first: bool,
        key: String,
    },
    WaitValLead {
        first: bool,
        key: String,
    },
    DetectVal {
        first: bool,
        key: String,
        held: String,
    },
    StreamString,
}

/// Progress result of one tool-call streaming step.
pub(super) enum StepOutcome {
    Wait,
    Continue,
    Done,
}

impl ToolState {
    /// Whether `ToolCallStart` has already been emitted for this block.
    pub(super) fn started(&self) -> bool {
        !matches!(self, Self::Start | Self::Buffer)
    }

    /// Advances the active tool-call block using `buf`.
    pub(super) fn step(
        &mut self,
        spec: &BlockSpec,
        buf: &mut String,
        out: &mut Vec<ResponseEvent>,
        eof: bool,
    ) -> StepOutcome {
        match self {
            Self::Start => self.step_start(spec, buf, out, eof),
            Self::Buffer => step_buffer(spec, buf, out, eof),
            Self::RawJson => step_raw_json(spec, buf, out, eof),
            Self::RelaxedJson(state) => step_relaxed(spec, state, buf, out, eof),
            Self::Tags(_) => self.step_tags(spec, buf, out, eof),
            Self::JsonArgs { .. } => self.step_json_args(spec, buf, out, eof),
            Self::DrainClose => step_drain_close(spec, buf, eof),
        }
    }

    /// Flushes a truncated tool call when generation finishes.
    pub(super) fn finish(&mut self, spec: &BlockSpec, rest: &str, out: &mut Vec<ResponseEvent>) {
        match self {
            Self::Start | Self::Buffer => {
                if let Some(calls) = parse_tool(spec, rest) {
                    out.extend(calls.into_iter().map(ResponseEvent::ToolCall));
                } else {
                    out.push(ResponseEvent::Content(rest.to_string()));
                }
            }
            Self::RawJson => {
                let tail = rest.trim_end();
                if !tail.is_empty() {
                    out.push(ResponseEvent::ToolCallArgs(tail.to_string()));
                }
                out.push(ResponseEvent::ToolCallEnd);
            }
            Self::RelaxedJson(state) => finish_relaxed(state, rest, out),
            Self::Tags(phase) => {
                finish_tags(phase, spec, rest, out);
                out.push(ResponseEvent::ToolCallEnd);
            }
            Self::JsonArgs { .. } => {
                if !rest.is_empty() {
                    out.push(ResponseEvent::ToolCallArgs(rest.to_string()));
                }
                out.push(ResponseEvent::ToolCallEnd);
            }
            Self::DrainClose => {}
        }
        *self = Self::Start;
    }

    fn step_start(
        &mut self,
        spec: &BlockSpec,
        buf: &mut String,
        out: &mut Vec<ResponseEvent>,
        eof: bool,
    ) -> StepOutcome {
        if spec.close.needs_buffer() {
            *self = Self::Buffer;
            return StepOutcome::Continue;
        }
        match (&spec.open, &spec.body) {
            (
                Opener::Pattern { .. },
                Body::Tags {
                    delims: Some(_), ..
                },
            ) => self.start_pattern(spec, buf, out, false, eof),
            (Opener::Pattern { .. }, Body::Json(_)) => {
                self.start_pattern(spec, buf, out, true, eof)
            }
            (Opener::Literal(_), Body::Json(args))
                if spec.transform.is_none()
                    && args.string_delims.is_empty()
                    && !args.unquoted_keys =>
            {
                self.start_literal_json(spec, buf, out, eof)
            }
            _ => {
                *self = Self::Buffer;
                StepOutcome::Continue
            }
        }
    }

    fn start_pattern(
        &mut self,
        spec: &BlockSpec,
        buf: &mut String,
        out: &mut Vec<ResponseEvent>,
        expect_json: bool,
        eof: bool,
    ) -> StepOutcome {
        let Some((caps, rest)) = spec.open.strip(buf) else {
            let from = spec.open.min_close_offset(buf);
            return match spec.close.find_from_for(buf, from, eof) {
                Some(_) => {
                    *self = Self::Buffer;
                    StepOutcome::Continue
                }
                None => StepOutcome::Wait,
            };
        };
        if rest.is_empty() {
            return StepOutcome::Wait;
        }
        let cut = buf.len() - rest.len();
        if expect_json {
            let trimmed = rest.trim_start();
            let valid_start = trimmed
                .as_bytes()
                .first()
                .is_some_and(|b| is_json_start(*b));
            if !valid_start {
                if spec.close.find_from_for(buf, cut, eof).is_some() {
                    *self = Self::Buffer;
                    return StepOutcome::Continue;
                }
                return StepOutcome::Wait;
            }
        }
        let Some(tmpl) = &spec.transform else {
            *self = Self::Buffer;
            return StepOutcome::Continue;
        };
        let Some((id, kind, name)) = transform_meta(tmpl, &caps) else {
            *self = Self::Buffer;
            return StepOutcome::Continue;
        };
        buf.drain(..cut);
        out.push(ResponseEvent::ToolCallStart { id, kind, name });
        match &spec.body {
            Body::Tags { .. } => {
                out.push(ResponseEvent::ToolCallArgs("{".into()));
                *self = Self::Tags(TagPhase::SeekKey { first: true });
            }
            Body::Json(args) if args.string_delims.is_empty() && !args.unquoted_keys => {
                let trim = buf.len() - buf.trim_start().len();
                buf.drain(..trim);
                *self = Self::RawJson;
            }
            Body::Json(_) => {
                *self = Self::RelaxedJson(RelaxedState::default());
            }
            Body::Text => unreachable!(),
        }
        StepOutcome::Continue
    }

    fn start_literal_json(
        &mut self,
        spec: &BlockSpec,
        buf: &mut String,
        out: &mut Vec<ResponseEvent>,
        eof: bool,
    ) -> StepOutcome {
        let Some((_, rest)) = spec.open.strip(buf) else {
            *self = Self::Buffer;
            return StepOutcome::Continue;
        };
        let open_len = buf.len() - rest.len();
        match scan_json_header(rest) {
            HeaderScan::Wait => match spec.close.find_from_for(buf, open_len, eof) {
                Some(_) => {
                    *self = Self::Buffer;
                    StepOutcome::Continue
                }
                None => StepOutcome::Wait,
            },
            HeaderScan::Fallback => {
                *self = Self::Buffer;
                StepOutcome::Continue
            }
            HeaderScan::Ready {
                id,
                kind,
                name,
                args_offset,
            } => {
                buf.drain(..open_len + args_offset);
                out.push(ResponseEvent::ToolCallStart { id, kind, name });
                *self = Self::JsonArgs {
                    depth: 0,
                    in_str: false,
                    escaped: false,
                    started: false,
                };
                StepOutcome::Continue
            }
        }
    }

    fn step_tags(
        &mut self,
        spec: &BlockSpec,
        buf: &mut String,
        out: &mut Vec<ResponseEvent>,
        eof: bool,
    ) -> StepOutcome {
        let Body::Tags {
            delims: Some(delims),
            ..
        } = &spec.body
        else {
            unreachable!()
        };
        let Self::Tags(phase) = self else {
            unreachable!()
        };
        let safe_end = if eof {
            buf.len()
        } else {
            buf.len() - spec.close.tail_hold(buf)
        };
        match phase {
            TagPhase::SeekKey { first } => {
                let first = *first;
                let key_at = buf.find(delims.key_open.as_str());
                let close = spec.close.find_for(buf, eof);
                match (key_at, close) {
                    (Some(at), c)
                        if c.is_none_or(|(end, _)| at < end)
                            && (eof || c.is_some() || at < safe_end) =>
                    {
                        buf.drain(..at + delims.key_open.len());
                        *phase = TagPhase::ReadKey { first };
                        StepOutcome::Continue
                    }
                    (_, Some((at, len))) => {
                        buf.drain(..at + len);
                        out.push(ResponseEvent::ToolCallArgs("}".into()));
                        out.push(ResponseEvent::ToolCallEnd);
                        StepOutcome::Done
                    }
                    _ => {
                        let keep = tail_hold(buf, &[delims.key_open.as_str()])
                            .max(spec.close.tail_hold(buf));
                        buf.drain(..buf.len() - keep);
                        StepOutcome::Wait
                    }
                }
            }
            TagPhase::ReadKey { first } => {
                let first = *first;
                let close = spec.close.find_for(buf, eof);
                if let Some(at) = buf.find(delims.key_close.as_str())
                    && close.is_none_or(|(end, _)| at < end)
                    && (eof || close.is_some() || at < safe_end)
                {
                    let key = buf[..at].to_string();
                    buf.drain(..at + delims.key_close.len());
                    *phase = next_after_key(delims, first, key);
                    return StepOutcome::Continue;
                }
                if let Some((at, len)) = close {
                    buf.drain(..at + len);
                    out.push(ResponseEvent::ToolCallArgs("}".into()));
                    out.push(ResponseEvent::ToolCallEnd);
                    return StepOutcome::Done;
                }
                StepOutcome::Wait
            }
            TagPhase::SeekValOpen { first, key } => {
                let close = spec.close.find_for(buf, eof);
                if let Some(at) = buf.find(delims.val_open.as_str())
                    && close.is_none_or(|(end, _)| at < end)
                    && (eof || close.is_some() || at < safe_end)
                {
                    let first = *first;
                    let key = std::mem::take(key);
                    buf.drain(..at + delims.val_open.len());
                    *phase = next_after_val_open(delims, first, key);
                    return StepOutcome::Continue;
                }
                if let Some((at, len)) = close {
                    buf.drain(..at + len);
                    out.push(ResponseEvent::ToolCallArgs("}".into()));
                    out.push(ResponseEvent::ToolCallEnd);
                    return StepOutcome::Done;
                }
                StepOutcome::Wait
            }
            TagPhase::WaitValLead { first, key } => {
                if buf.is_empty() {
                    return StepOutcome::Wait;
                }
                if buf.starts_with('\n') {
                    buf.drain(..1);
                }
                let first = *first;
                let key = std::mem::take(key);
                *phase = TagPhase::DetectVal {
                    first,
                    key,
                    held: String::new(),
                };
                StepOutcome::Continue
            }
            TagPhase::DetectVal { first, key, held } => {
                let val_hit = find_closers_for(buf, &delims.val_close, eof);
                let blk_hit = spec.close.find_for(buf, eof);
                if let Some((at, len)) = val_hit
                    && blk_hit.is_none_or(|(b, _)| at <= b)
                    && (eof || blk_hit.is_some() || at < safe_end)
                {
                    held.push_str(&buf[..at]);
                    buf.drain(..at + len);
                    if delims.trim_trail_nl && held.ends_with('\n') {
                        held.pop();
                    }
                    let prefix = if *first { "" } else { "," };
                    let encoded = encode_tag_value(held);
                    out.push(ResponseEvent::ToolCallArgs(format!(
                        "{prefix}{}:{encoded}",
                        json_str(key)
                    )));
                    *phase = TagPhase::SeekKey { first: false };
                    return StepOutcome::Continue;
                }
                if let Some((at, len)) = blk_hit {
                    held.push_str(&buf[..at]);
                    buf.drain(..at + len);
                    let prefix = if *first { "" } else { "," };
                    let encoded = encode_tag_value(held);
                    out.push(ResponseEvent::ToolCallArgs(format!(
                        "{prefix}{}:{encoded}}}",
                        json_str(key)
                    )));
                    out.push(ResponseEvent::ToolCallEnd);
                    return StepOutcome::Done;
                }
                let val_needles = delims.val_needles();
                let emit = safe_value_emit(buf, &val_needles, &spec.close, delims.trim_trail_nl);
                if emit > 0 {
                    held.push_str(&buf[..emit]);
                    buf.drain(..emit);
                }
                if !could_be_json_prefix(held) {
                    let prefix = if *first { "" } else { "," };
                    let body = json_escape_inner(held);
                    out.push(ResponseEvent::ToolCallArgs(format!(
                        "{prefix}{}:\"{body}",
                        json_str(key)
                    )));
                    *phase = TagPhase::StreamString;
                    return StepOutcome::Continue;
                }
                StepOutcome::Wait
            }
            TagPhase::StreamString => {
                let val_hit = find_closers_for(buf, &delims.val_close, eof);
                let blk_hit = spec.close.find_for(buf, eof);
                if let Some((at, len)) = val_hit
                    && blk_hit.is_none_or(|(b, _)| at <= b)
                    && (eof || blk_hit.is_some() || at < safe_end)
                {
                    let mut chunk = &buf[..at];
                    if delims.trim_trail_nl {
                        chunk = chunk.strip_suffix('\n').unwrap_or(chunk);
                    }
                    let escaped = json_escape_inner(chunk);
                    buf.drain(..at + len);
                    out.push(ResponseEvent::ToolCallArgs(format!("{escaped}\"")));
                    *phase = TagPhase::SeekKey { first: false };
                    return StepOutcome::Continue;
                }
                if let Some((at, len)) = blk_hit {
                    let escaped = json_escape_inner(&buf[..at]);
                    buf.drain(..at + len);
                    out.push(ResponseEvent::ToolCallArgs(format!("{escaped}\"}}")));
                    out.push(ResponseEvent::ToolCallEnd);
                    return StepOutcome::Done;
                }
                let val_needles = delims.val_needles();
                let emit = safe_value_emit(buf, &val_needles, &spec.close, delims.trim_trail_nl);
                if emit > 0 {
                    let escaped = json_escape_inner(&buf[..emit]);
                    buf.drain(..emit);
                    out.push(ResponseEvent::ToolCallArgs(escaped));
                }
                StepOutcome::Wait
            }
        }
    }

    fn step_json_args(
        &mut self,
        spec: &BlockSpec,
        buf: &mut String,
        out: &mut Vec<ResponseEvent>,
        eof: bool,
    ) -> StepOutcome {
        let Self::JsonArgs {
            depth,
            in_str,
            escaped,
            started,
        } = self
        else {
            unreachable!()
        };
        let close = spec.close.find_for(buf, eof);
        let limit = match close {
            Some((at, _)) => at,
            None => buf.len() - spec.close.tail_hold(buf),
        };
        let bytes = buf.as_bytes();
        let mut end = None;
        for (i, &b) in bytes[..limit].iter().enumerate() {
            if *in_str {
                if *escaped {
                    *escaped = false;
                } else if b == b'\\' {
                    *escaped = true;
                } else if b == b'"' {
                    *in_str = false;
                }
                continue;
            }
            match b {
                b'"' => *in_str = true,
                b'{' => {
                    *depth += 1;
                    *started = true;
                }
                b'}' => {
                    *depth = depth.saturating_sub(1);
                    if *started && *depth == 0 {
                        end = Some(i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        if let Some(cut) = end {
            out.push(ResponseEvent::ToolCallArgs(buf[..cut].to_string()));
            out.push(ResponseEvent::ToolCallEnd);
            buf.drain(..cut);
            *self = Self::DrainClose;
            return StepOutcome::Continue;
        }
        if limit > 0 {
            out.push(ResponseEvent::ToolCallArgs(buf[..limit].to_string()));
            buf.drain(..limit);
        }
        if let Some((at, len)) = spec.close.find_for(buf, eof) {
            buf.drain(..at + len);
            out.push(ResponseEvent::ToolCallEnd);
            return StepOutcome::Done;
        }
        StepOutcome::Wait
    }
}

fn step_buffer(
    spec: &BlockSpec,
    buf: &mut String,
    out: &mut Vec<ResponseEvent>,
    eof: bool,
) -> StepOutcome {
    let from = match spec.open.strip(buf) {
        Some((_, rest)) => buf.len() - rest.len(),
        None => spec.open.min_close_offset(buf),
    };
    let Some((at, len)) = spec.close.find_from_for(buf, from, eof) else {
        return StepOutcome::Wait;
    };
    let block = buf[..at].to_string();
    let raw_end = at + len;
    match parse_tool(spec, &block) {
        Some(calls) => out.extend(calls.into_iter().map(ResponseEvent::ToolCall)),
        None => out.push(ResponseEvent::Content(buf[..raw_end].to_string())),
    }
    buf.drain(..raw_end);
    StepOutcome::Done
}

fn step_raw_json(
    spec: &BlockSpec,
    buf: &mut String,
    out: &mut Vec<ResponseEvent>,
    eof: bool,
) -> StepOutcome {
    if let Some((at, len)) = spec.close.find_for(buf, eof) {
        let tail = buf[..at].trim_end();
        if !tail.is_empty() {
            out.push(ResponseEvent::ToolCallArgs(tail.to_string()));
        }
        buf.drain(..at + len);
        out.push(ResponseEvent::ToolCallEnd);
        return StepOutcome::Done;
    }
    let keep = spec.close.tail_hold(buf);
    let emit = buf.len() - keep;
    if emit > 0 {
        out.push(ResponseEvent::ToolCallArgs(buf[..emit].to_string()));
        buf.drain(..emit);
    }
    StepOutcome::Wait
}

fn step_drain_close(spec: &BlockSpec, buf: &mut String, eof: bool) -> StepOutcome {
    if let Some((at, len)) = spec.close.find_for(buf, eof) {
        buf.drain(..at + len);
        return StepOutcome::Done;
    }
    let keep = spec.close.tail_hold(buf);
    buf.drain(..buf.len() - keep);
    StepOutcome::Wait
}

fn finish_tags(phase: &mut TagPhase, spec: &BlockSpec, rest: &str, out: &mut Vec<ResponseEvent>) {
    let Body::Tags {
        delims: Some(delims),
        ..
    } = &spec.body
    else {
        return;
    };
    match phase {
        TagPhase::StreamString => {
            let mut chunk = rest;
            if let Some((at, _)) = find_closers(chunk, &delims.val_close) {
                chunk = &chunk[..at];
            }
            if delims.trim_trail_nl {
                chunk = chunk.strip_suffix('\n').unwrap_or(chunk);
            }
            if !chunk.is_empty() {
                out.push(ResponseEvent::ToolCallArgs(json_escape_inner(chunk)));
            }
        }
        TagPhase::DetectVal { first, key, held } => {
            held.push_str(rest);
            if !held.is_empty() {
                let prefix = if *first { "" } else { "," };
                let encoded = encode_tag_value(held);
                out.push(ResponseEvent::ToolCallArgs(format!(
                    "{prefix}{}:{encoded}",
                    json_str(key)
                )));
            }
        }
        _ => {}
    }
}

fn next_after_key(delims: &TagDelims, first: bool, key: String) -> TagPhase {
    if delims.val_open.is_empty() {
        next_after_val_open(delims, first, key)
    } else {
        TagPhase::SeekValOpen { first, key }
    }
}

fn next_after_val_open(delims: &TagDelims, first: bool, key: String) -> TagPhase {
    if delims.trim_lead_nl {
        TagPhase::WaitValLead { first, key }
    } else {
        TagPhase::DetectVal {
            first,
            key,
            held: String::new(),
        }
    }
}

fn is_json_start(b: u8) -> bool {
    matches!(
        b,
        b'{' | b'[' | b'"' | b't' | b'f' | b'n' | b'-' | b'0'..=b'9'
    )
}

pub(super) fn safe_value_emit(
    buf: &str,
    val_closes: &[&str],
    block_close: &Closers,
    trim_nl: bool,
) -> usize {
    let keep = tail_hold(buf, val_closes).max(block_close.tail_hold(buf));
    let mut emit = buf.len() - keep;
    if trim_nl && emit > 0 && buf.as_bytes()[emit - 1] == b'\n' {
        emit -= 1;
    }
    emit
}

fn encode_tag_value(raw: &str) -> String {
    match serde_json::from_str::<Value>(raw) {
        Ok(val) => serde_json::to_string(&val).unwrap_or_else(|_| json_str(raw)),
        Err(_) => json_str(raw),
    }
}

pub(super) fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into())
}

pub(super) fn json_escape_inner(s: &str) -> String {
    let quoted = json_str(s);
    quoted[1..quoted.len() - 1].to_string()
}

/// Whether `s` could still become a valid JSON value.
fn could_be_json_prefix(s: &str) -> bool {
    let t = s.trim_start();
    let Some(&first) = t.as_bytes().first() else {
        return true;
    };
    match first {
        b'{' | b'[' | b'"' => true,
        b't' => {
            "true".starts_with(t) || t.strip_prefix("true").is_some_and(|r| r.trim().is_empty())
        }
        b'f' => {
            "false".starts_with(t) || t.strip_prefix("false").is_some_and(|r| r.trim().is_empty())
        }
        b'n' => {
            "null".starts_with(t) || t.strip_prefix("null").is_some_and(|r| r.trim().is_empty())
        }
        b'-' | b'0'..=b'9' => is_json_number_prefix(t),
        _ => false,
    }
}

/// Whether left-trimmed `t` can prefix a JSON number.
fn is_json_number_prefix(t: &str) -> bool {
    let b = t.as_bytes();
    let mut i = usize::from(b.first() == Some(&b'-'));
    if i == b.len() {
        return true;
    }
    match b[i] {
        b'0' => i += 1,
        b'1'..=b'9' => {
            i += 1;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        _ => return false,
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let frac = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == b.len() {
            return true;
        }
        if i == frac {
            return false;
        }
    }
    if i < b.len() && matches!(b[i], b'e' | b'E') {
        i += 1;
        if i < b.len() && matches!(b[i], b'+' | b'-') {
            i += 1;
        }
        let exp = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == b.len() {
            return true;
        }
        if i == exp {
            return false;
        }
    }
    b[i..].iter().all(u8::is_ascii_whitespace)
}

/// Extracts `(id, kind, name)` when `tmpl` maps `arguments` to `{content}`.
fn transform_meta(
    tmpl: &Value,
    caps: &BTreeMap<String, String>,
) -> Option<(Option<String>, String, String)> {
    let Value::Object(obj) = tmpl else {
        return None;
    };
    let func = match obj.get("function") {
        Some(Value::Object(inner)) => inner,
        None => obj,
        _ => return None,
    };
    if func.get("arguments")?.as_str()? != "{content}" {
        return None;
    }
    let name = resolve_field(func.get("name")?, caps)?;
    let id = obj.get("id").and_then(|v| resolve_field(v, caps));
    let kind = obj
        .get("type")
        .and_then(|v| resolve_field(v, caps))
        .unwrap_or_else(|| "function".into());
    Some((id, kind, name))
}

fn resolve_field(val: &Value, caps: &BTreeMap<String, String>) -> Option<String> {
    let s = val.as_str()?;
    if let Some(key) = s.strip_prefix('{').and_then(|v| v.strip_suffix('}')) {
        return caps.get(key).cloned();
    }
    Some(s.to_string())
}

enum HeaderScan {
    Wait,
    Fallback,
    Ready {
        id: Option<String>,
        kind: String,
        name: String,
        args_offset: usize,
    },
}

/// Scans a literal JSON object header up to `"arguments": {`.
fn scan_json_header(src: &str) -> HeaderScan {
    let bytes = src.as_bytes();
    let mut pos = skip_ws(bytes, 0);
    if pos >= bytes.len() {
        return HeaderScan::Wait;
    }
    if bytes[pos] != b'{' {
        return HeaderScan::Fallback;
    }
    pos += 1;
    let mut name: Option<String> = None;
    let mut id: Option<String> = None;
    let mut kind = String::from("function");
    loop {
        pos = skip_ws(bytes, pos);
        if pos >= bytes.len() {
            return HeaderScan::Wait;
        }
        if bytes[pos] == b'}' {
            return HeaderScan::Fallback;
        }
        let (key, next) = match scan_quoted(src, pos) {
            QuotedScan::Done(s, n) => (s, n),
            QuotedScan::Wait => return HeaderScan::Wait,
            QuotedScan::Invalid => return HeaderScan::Fallback,
        };
        pos = skip_ws(bytes, next);
        if pos >= bytes.len() {
            return HeaderScan::Wait;
        }
        if bytes[pos] != b':' {
            return HeaderScan::Fallback;
        }
        pos = skip_ws(bytes, pos + 1);
        if pos >= bytes.len() {
            return HeaderScan::Wait;
        }
        if key == "arguments" {
            if bytes[pos] == b'{'
                && let Some(name) = name
            {
                return HeaderScan::Ready {
                    id,
                    kind,
                    name,
                    args_offset: pos,
                };
            }
            return HeaderScan::Fallback;
        }
        let (val, next) = match scan_quoted(src, pos) {
            QuotedScan::Done(s, n) => (s, n),
            QuotedScan::Wait => return HeaderScan::Wait,
            QuotedScan::Invalid => return HeaderScan::Fallback,
        };
        match key.as_str() {
            "name" => name = Some(val),
            "id" => id = Some(val),
            "type" => kind = val,
            _ => return HeaderScan::Fallback,
        }
        pos = skip_ws(bytes, next);
        if pos >= bytes.len() {
            return HeaderScan::Wait;
        }
        if bytes[pos] == b',' {
            pos += 1;
        } else {
            return HeaderScan::Fallback;
        }
    }
}

enum QuotedScan {
    Done(String, usize),
    Wait,
    Invalid,
}

fn scan_quoted(src: &str, start: usize) -> QuotedScan {
    let bytes = src.as_bytes();
    if bytes.get(start) != Some(&b'"') {
        return QuotedScan::Invalid;
    }
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => {
                return match serde_json::from_str::<String>(&src[start..=i]) {
                    Ok(s) => QuotedScan::Done(s, i + 1),
                    Err(_) => QuotedScan::Invalid,
                };
            }
            _ => i += 1,
        }
    }
    QuotedScan::Wait
}

fn skip_ws(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
        pos += 1;
    }
    pos
}

fn strip_inline_flags(mut s: &str) -> &str {
    while let Some(rest) = s.strip_prefix("(?") {
        let Some(end) = rest.find(')') else { break };
        let flags = &rest[..end];
        if !flags.is_empty() && flags.bytes().all(|b| b.is_ascii_alphabetic() || b == b'-') {
            s = &rest[end + 1..];
        } else {
            break;
        }
    }
    s
}

fn strip_opt_nl_end(s: &str) -> (&str, bool) {
    match s.strip_suffix("\\n?") {
        Some(rest) => (rest, true),
        None => (s, false),
    }
}

fn strip_opt_nl_start(s: &str) -> (&str, bool) {
    match s.strip_prefix("\\n?") {
        Some(rest) => (rest, true),
        None => (s, false),
    }
}
