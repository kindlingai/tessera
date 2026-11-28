//! Block and tag closer matching with literal and lookahead support.

use fancy_regex::Regex;
use std::error::Error;

/// One literal closer: consuming or zero-width positive lookahead.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Closer {
    pub needle: String,
    pub consume: bool,
}

impl Closer {
    pub(super) fn consume(needle: String) -> Self {
        Self {
            needle,
            consume: true,
        }
    }

    pub(super) fn peek(needle: String) -> Self {
        Self {
            needle,
            consume: false,
        }
    }

    pub(super) fn consumed_len(&self) -> usize {
        if self.consume { self.needle.len() } else { 0 }
    }
}

/// Compiled close condition for a block or content field.
#[derive(Clone)]
pub(super) enum Closers {
    Literals(Vec<Closer>),
    Pattern {
        regex: Regex,
        needles: Vec<String>,
        consuming: Vec<String>,
        unanchored: bool,
    },
}

impl Closers {
    pub(super) fn empty() -> Self {
        Self::Literals(Vec::new())
    }

    pub(super) fn compile(
        close: Vec<String>,
        pattern: Option<String>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        if !close.is_empty() {
            return Ok(Self::Literals(
                close.into_iter().map(Closer::consume).collect(),
            ));
        }
        let Some(pat) = pattern else {
            return Ok(Self::empty());
        };
        if let Some(list) = parse_literal_closers(&pat) {
            return Ok(Self::Literals(list));
        }
        let regex = Regex::new(&pat)?;
        let mut needles = Vec::new();
        let mut consuming = Vec::new();
        let mut unanchored = false;
        for branch in split_alt(strip_group(&pat)) {
            let b = strip_group(branch);
            if let Some(inner) = strip_lookahead(b) {
                let p = literal_prefix(inner);
                if p.is_empty() {
                    unanchored = true;
                } else {
                    needles.push(p);
                }
            } else if b.starts_with("(?") {
                unanchored = true;
            } else {
                let p = literal_prefix(b);
                if p.is_empty() {
                    unanchored = true;
                } else {
                    needles.push(p);
                }
                if let Some(lit) = unescape_lit(b)
                    && !lit.is_empty()
                {
                    consuming.push(lit);
                }
            }
        }
        Ok(Self::Pattern {
            regex,
            needles,
            consuming,
            unanchored,
        })
    }

    pub(super) fn is_empty(&self) -> bool {
        matches!(self, Self::Literals(list) if list.is_empty())
    }

    pub(super) fn find(&self, text: &str) -> Option<(usize, usize)> {
        self.find_from(text, 0)
    }

    pub(super) fn find_from(&self, text: &str, from: usize) -> Option<(usize, usize)> {
        if from > text.len() {
            return None;
        }
        let slice = &text[from..];
        match self {
            Self::Literals(list) => find_closers(slice, list).map(|(at, len)| (from + at, len)),
            Self::Pattern { regex, .. } => regex
                .find(slice)
                .ok()
                .flatten()
                .map(|m| (from + m.start(), m.end() - m.start())),
        }
    }

    /// Trailing bytes held to match closers across chunk boundaries.
    pub(super) fn tail_hold(&self, buf: &str) -> usize {
        match self {
            Self::Literals(list) => {
                let needles: Vec<&str> = list.iter().map(|c| c.needle.as_str()).collect();
                super::tail_hold(buf, &needles)
            }
            Self::Pattern {
                needles,
                unanchored,
                ..
            } => {
                if *unanchored {
                    return buf.len();
                }
                let refs: Vec<&str> = needles.iter().map(String::as_str).collect();
                let mut hold = super::tail_hold(buf, &refs);
                for needle in needles {
                    if let Some(at) = buf.find(needle.as_str()) {
                        hold = hold.max(buf.len() - at);
                    }
                }
                hold
            }
        }
    }

    pub(super) fn consuming(&self) -> Vec<&str> {
        match self {
            Self::Literals(list) => list
                .iter()
                .filter(|c| c.consume)
                .map(|c| c.needle.as_str())
                .collect(),
            Self::Pattern { consuming, .. } => consuming.iter().map(String::as_str).collect(),
        }
    }
}

/// Finds the earliest match among literal and lookahead closers.
pub(super) fn find_closers(text: &str, closers: &[Closer]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for c in closers {
        if let Some(at) = text.find(c.needle.as_str()) {
            let len = c.consumed_len();
            let better = best.is_none_or(|(b_at, b_len)| at < b_at || (at == b_at && len > b_len));
            if better {
                best = Some((at, len));
            }
        }
    }
    best
}

/// Parses an alternation of literal and `(?=literal)` closer branches.
pub(super) fn parse_literal_closers(pat: &str) -> Option<Vec<Closer>> {
    let mut out = Vec::new();
    for branch in split_alt(strip_group(pat)) {
        let b = strip_group(branch);
        if let Some(inner) = strip_lookahead(b) {
            let lit = unescape_lit(inner)?;
            if lit.is_empty() {
                return None;
            }
            out.push(Closer::peek(lit));
        } else {
            let lit = unescape_lit(b)?;
            if lit.is_empty() {
                return None;
            }
            out.push(Closer::consume(lit));
        }
    }
    (!out.is_empty()).then_some(out)
}

fn strip_group(mut s: &str) -> &str {
    while s.starts_with("(?:") && matching_paren(s, 0) == Some(s.len() - 1) {
        s = &s[3..s.len() - 1];
    }
    s
}

fn strip_lookahead(s: &str) -> Option<&str> {
    if s.starts_with("(?=") && matching_paren(s, 0) == Some(s.len() - 1) {
        Some(&s[3..s.len() - 1])
    } else {
        None
    }
}

/// Splits `s` on top-level unescaped `|` outside groups and classes.
fn split_alt(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut depth = 0usize;
    let mut in_class = false;
    let mut chars = s.char_indices();
    while let Some((i, ch)) = chars.next() {
        match ch {
            '\\' => {
                let _ = chars.next();
            }
            '[' if !in_class => in_class = true,
            ']' if in_class => in_class = false,
            '(' if !in_class => depth += 1,
            ')' if !in_class => depth = depth.saturating_sub(1),
            '|' if depth == 0 && !in_class => {
                out.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

/// Byte offset of the `)` matching `(` at `open_at`.
pub(super) fn matching_paren(s: &str, open_at: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_class = false;
    let mut chars = s[open_at..].char_indices();
    while let Some((rel, ch)) = chars.next() {
        match ch {
            '\\' => {
                let _ = chars.next();
            }
            '[' if !in_class => in_class = true,
            ']' if in_class => in_class = false,
            '(' if !in_class => depth += 1,
            ')' if !in_class => {
                depth -= 1;
                if depth == 0 {
                    return Some(open_at + rel);
                }
            }
            _ => {}
        }
    }
    None
}

/// Unescapes a regex fragment containing only literal characters.
pub(super) fn unescape_lit(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            let next = chars.next()?;
            match next {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                other if other.is_ascii_punctuation() => out.push(other),
                _ => return None,
            }
            continue;
        }
        if matches!(
            c,
            '(' | ')' | '[' | ']' | '{' | '}' | '.' | '*' | '+' | '?' | '|' | '^' | '$'
        ) {
            return None;
        }
        out.push(c);
    }
    Some(out)
}

/// Extracts the leading literal text from a regex pattern.
pub(super) fn literal_prefix(pattern: &str) -> String {
    let mut out = String::new();
    let mut chars = pattern.chars().peekable();
    while let Some(c) = chars.next() {
        let ch = if c == '\\' {
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
            escaped
        } else {
            if matches!(
                c,
                '(' | ')' | '[' | ']' | '{' | '}' | '.' | '*' | '+' | '?' | '|' | '^' | '$'
            ) {
                break;
            }
            c
        };
        if matches!(chars.peek(), Some('?' | '*' | '{')) {
            break;
        }
        out.push(ch);
    }
    out
}
