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

/// Streaming settlement rule for regex closers.
#[derive(Clone)]
pub(super) enum StreamRule {
    AfterEnd,
    NotPrefixes(Vec<String>),
    TrailingChars(usize),
    DeferEof,
}

impl StreamRule {
    fn settled(&self, text: &str, end: usize) -> bool {
        let after = &text[end..];
        match self {
            Self::AfterEnd => !after.is_empty(),
            Self::NotPrefixes(guards) => {
                !after.is_empty() && !guards.iter().any(|g| g.starts_with(after))
            }
            Self::TrailingChars(need) => {
                !after.is_empty() && after.chars().nth(need.saturating_sub(1)).is_some()
            }
            Self::DeferEof => false,
        }
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
        rule: StreamRule,
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
        let (rule, needs_context) = stream_rule(&pat);
        let mut needles = Vec::new();
        let mut consuming = Vec::new();
        let mut unanchored = needs_context;
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
            rule,
        })
    }

    pub(super) fn is_empty(&self) -> bool {
        matches!(self, Self::Literals(list) if list.is_empty())
    }

    pub(super) fn defers_until_eof(&self) -> bool {
        matches!(
            self,
            Self::Pattern {
                rule: StreamRule::DeferEof,
                ..
            }
        )
    }

    pub(super) fn find_for(&self, text: &str, eof: bool) -> Option<(usize, usize)> {
        self.find_from_for(text, 0, eof)
    }

    pub(super) fn find_from_for(
        &self,
        text: &str,
        from: usize,
        eof: bool,
    ) -> Option<(usize, usize)> {
        if from > text.len() {
            return None;
        }
        let slice = &text[from..];
        match self {
            Self::Literals(list) => find_closers(slice, list).map(|(at, len)| (from + at, len)),
            Self::Pattern { regex, rule, .. } => {
                if !eof && matches!(rule, StreamRule::DeferEof) {
                    return None;
                }
                let m = regex.find(slice).ok().flatten()?;
                if !eof && !rule.settled(text, from + m.end()) {
                    return None;
                }
                Some((from + m.start(), m.end() - m.start()))
            }
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

/// Determines when a regex closer match settles mid-stream.
fn stream_rule(pat: &str) -> (StreamRule, bool) {
    if let Some(guards) = parse_neg_guards(pat) {
        return (StreamRule::NotPrefixes(guards), false);
    }
    let bytes = pat.as_bytes();
    let mut i = 0;
    let mut in_class = false;
    let mut max_la = 0usize;
    let mut needs_context = false;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                if !in_class && i + 1 < bytes.len() {
                    match bytes[i + 1] {
                        b'z' | b'Z' => return (StreamRule::DeferEof, needs_context),
                        b'b' | b'B' => {
                            max_la = max_la.max(1);
                            needs_context = true;
                        }
                        _ => {}
                    }
                }
                i += 2;
                continue;
            }
            b'[' if !in_class => in_class = true,
            b']' if in_class => in_class = false,
            b'$' if !in_class => return (StreamRule::DeferEof, needs_context),
            b'(' if !in_class && i + 2 < bytes.len() && bytes[i + 1] == b'?' => {
                let Some(end) = matching_paren(pat, i) else {
                    return (StreamRule::DeferEof, needs_context);
                };
                if bytes[i + 2] == b'<'
                    && i + 3 < bytes.len()
                    && matches!(bytes[i + 3], b'=' | b'!')
                {
                    needs_context = true;
                    if has_forward_assert(&pat[i + 4..end]) {
                        return (StreamRule::DeferEof, true);
                    }
                    i = end + 1;
                    continue;
                }
                if matches!(bytes[i + 2], b'=' | b'!') {
                    let Some(width) = max_match_chars(&pat[i + 3..end]) else {
                        return (StreamRule::DeferEof, needs_context);
                    };
                    max_la = max_la.max(width);
                    i = end + 1;
                    continue;
                }
                let head = &pat[i + 2..end];
                let flags = head.split_once(':').map_or(head, |(f, _)| f);
                if flags.contains('x') {
                    return (StreamRule::DeferEof, needs_context);
                }
            }
            _ => {}
        }
        i += 1;
    }
    let rule = match max_la {
        0 => StreamRule::AfterEnd,
        n => StreamRule::TrailingChars(n),
    };
    (rule, needs_context)
}

/// Extracts literal guards from trailing `(?!...)` closer branches.
fn parse_neg_guards(pat: &str) -> Option<Vec<String>> {
    let mut guards = Vec::new();
    for branch in split_alt(strip_group(pat)) {
        let b = strip_group(branch);
        if let Some(inner) = strip_lookahead(b) {
            if unescape_lit(inner)?.is_empty() {
                return None;
            }
            continue;
        }
        let bytes = b.as_bytes();
        let mut i = 0;
        let mut in_class = false;
        while i < bytes.len() {
            match bytes[i] {
                b'\\' => {
                    if !in_class
                        && i + 1 < bytes.len()
                        && matches!(bytes[i + 1], b'b' | b'B' | b'z' | b'Z')
                    {
                        return None;
                    }
                    i += 2;
                    continue;
                }
                b'[' if !in_class => in_class = true,
                b']' if in_class => in_class = false,
                b'$' if !in_class => return None,
                b'(' if !in_class && i + 1 < bytes.len() && bytes[i + 1] == b'?' => {
                    let end = matching_paren(b, i)?;
                    if i + 2 < bytes.len() && bytes[i + 2] == b'!' && end + 1 == b.len() {
                        for alt in split_alt(strip_group(&b[i + 3..end])) {
                            let lit = unescape_lit(strip_group(alt))?;
                            if lit.is_empty() {
                                return None;
                            }
                            guards.push(lit);
                        }
                        break;
                    }
                    if !(i + 2 < bytes.len() && bytes[i + 2] == b':') {
                        return None;
                    }
                    i += 3;
                    continue;
                }
                _ => {}
            }
            i += 1;
        }
    }
    (!guards.is_empty()).then_some(guards)
}

fn has_forward_assert(pat: &str) -> bool {
    let bytes = pat.as_bytes();
    let mut i = 0;
    let mut in_class = false;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                if !in_class
                    && i + 1 < bytes.len()
                    && matches!(bytes[i + 1], b'z' | b'Z' | b'b' | b'B')
                {
                    return true;
                }
                i += 2;
                continue;
            }
            b'[' if !in_class => in_class = true,
            b']' if in_class => in_class = false,
            b'$' if !in_class => return true,
            b'(' if !in_class
                && i + 2 < bytes.len()
                && bytes[i + 1] == b'?'
                && matches!(bytes[i + 2], b'=' | b'!') =>
            {
                return true;
            }
            _ => {}
        }
        i += 1;
    }
    false
}

/// Max chars matched by `pat`, or `None` if unbounded.
fn max_match_chars(pat: &str) -> Option<usize> {
    let mut best = 0usize;
    for branch in split_alt(pat) {
        best = best.max(branch_max_chars(branch)?);
    }
    Some(best)
}

fn branch_max_chars(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut total = 0usize;
    let mut last: Option<usize> = None;
    while i < bytes.len() {
        match bytes[i] {
            b'*' | b'+' => return None,
            b'?' => {
                last?;
                i += 1;
                if i < bytes.len() && bytes[i] == b'?' {
                    i += 1;
                }
                continue;
            }
            b'{' => {
                let (mult, next) = parse_quant_max(s, i)?;
                last = Some(last?.checked_mul(mult)?);
                i = next;
                if i < bytes.len() && bytes[i] == b'?' {
                    i += 1;
                }
                continue;
            }
            _ => {}
        }
        if let Some(prev) = last.take() {
            total = total.checked_add(prev)?;
        }
        match bytes[i] {
            b'$' => return None,
            b'^' => i += 1,
            b'\\' => {
                let (w, next) = escape_width(s, i)?;
                last = Some(w);
                i = next;
            }
            b'[' => {
                let next = skip_class(bytes, i)?;
                last = Some(1);
                i = next;
            }
            b'(' => {
                let end = matching_paren(s, i)?;
                let inner = &s[i + 1..end];
                let w = group_max_chars(inner)?;
                last = Some(w);
                i = end + 1;
            }
            _ => {
                let ch = s[i..].chars().next()?;
                last = Some(1);
                i += ch.len_utf8();
            }
        }
    }
    if let Some(prev) = last {
        total = total.checked_add(prev)?;
    }
    Some(total)
}

fn group_max_chars(inner: &str) -> Option<usize> {
    let Some(rest) = inner.strip_prefix('?') else {
        return max_match_chars(inner);
    };
    if let Some(body) = rest.strip_prefix(':') {
        return max_match_chars(body);
    }
    if let Some(body) = rest.strip_prefix('=').or_else(|| rest.strip_prefix('!')) {
        return max_match_chars(body);
    }
    if let Some(body) = rest.strip_prefix("<=").or_else(|| rest.strip_prefix("<!")) {
        return (!has_forward_assert(body)).then_some(0);
    }
    if let Some(after) = rest.strip_prefix("P<").or_else(|| rest.strip_prefix('<')) {
        let close = after.find('>')?;
        return max_match_chars(&after[close + 1..]);
    }
    let (flags, body) = match rest.split_once(':') {
        Some((f, b)) => (f, Some(b)),
        None => (rest, None),
    };
    if flags.contains('x') || !flags.bytes().all(|b| b.is_ascii_alphabetic() || b == b'-') {
        return None;
    }
    match body {
        Some(b) => max_match_chars(b),
        None => Some(0),
    }
}

fn escape_width(s: &str, at: usize) -> Option<(usize, usize)> {
    let bytes = s.as_bytes();
    let next = *bytes.get(at + 1)?;
    match next {
        b'z' | b'Z' | b'1'..=b'9' => None,
        b'A' => Some((0, at + 2)),
        b'b' | b'B' => Some((1, at + 2)),
        b'p' | b'P' => {
            if bytes.get(at + 2) == Some(&b'{') {
                let close = s[at + 3..].find('}')?;
                Some((1, at + 4 + close))
            } else {
                Some((1, at + 3))
            }
        }
        b'x' => {
            if bytes.get(at + 2) == Some(&b'{') {
                let close = s[at + 3..].find('}')?;
                Some((1, at + 4 + close))
            } else {
                (at + 4 <= bytes.len()).then_some((1, at + 4))
            }
        }
        b'u' => {
            if bytes.get(at + 2) == Some(&b'{') {
                let close = s[at + 3..].find('}')?;
                Some((1, at + 4 + close))
            } else {
                (at + 6 <= bytes.len()).then_some((1, at + 6))
            }
        }
        _ => {
            let ch = s[at + 1..].chars().next()?;
            Some((1, at + 1 + ch.len_utf8()))
        }
    }
}

fn skip_class(bytes: &[u8], start: usize) -> Option<usize> {
    let mut i = start + 1;
    if i < bytes.len() && bytes[i] == b'^' {
        i += 1;
    }
    if i < bytes.len() && bytes[i] == b']' {
        i += 1;
    }
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b']' => return Some(i + 1),
            _ => i += 1,
        }
    }
    None
}

fn parse_quant_max(s: &str, start: usize) -> Option<(usize, usize)> {
    let close = s[start + 1..].find('}')?;
    let body = &s[start + 1..start + 1 + close];
    let max = match body.split_once(',') {
        Some((_, "")) => return None,
        Some((_, hi)) => hi.parse::<usize>().ok()?,
        None => body.parse::<usize>().ok()?,
    };
    Some((max, start + 2 + close))
}
