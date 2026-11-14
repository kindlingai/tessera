//! Comment extraction for Rust and C-like kernel sources.

use crate::files::Source;
use crate::{LIMITS, Violation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Line,
    OuterDoc,
    InnerDoc,
    Block,
}

/// One comment as written, before grouping.
#[derive(Debug)]
pub struct Raw {
    pub kind: Kind,
    pub line: usize,
    pub end: usize,
    pub words: usize,
    /// Code precedes it on its first line.
    pub trailing: bool,
    /// Opens or closes a ``` code block.
    pub fence: bool,
    /// Holds a `;` outside backticks.
    pub semicolon: bool,
}

/// A comment as the rule counts it.
#[derive(Debug, PartialEq, Eq)]
pub struct Comment {
    pub line: usize,
    pub words: usize,
}

/// A `;` outside backtick spans.
pub fn prose_semicolon(body: &str) -> bool {
    let mut code = false;
    for c in body.chars() {
        match c {
            '`' => code = !code,
            ';' if !code => return true,
            _ => {}
        }
    }
    false
}

pub fn words(body: &str) -> usize {
    body.split_whitespace()
        .filter(|w| w.chars().any(char::is_alphanumeric))
        .count()
}

struct Scanner<'a> {
    b: &'a [u8],
    i: usize,
    line: usize,
    code: bool,
    rust: bool,
    out: Vec<Raw>,
    stripped: Vec<u8>,
}

impl Scanner<'_> {
    fn peek(&self, k: usize) -> u8 {
        self.b.get(self.i + k).copied().unwrap_or(0)
    }

    /// Consumes one byte, keeping code bytes when `keep` is set.
    fn bump(&mut self, keep: bool) {
        let c = self.b[self.i];
        if c == b'\n' {
            self.line += 1;
            self.code = false;
        }
        let kept = if keep || c == b'\n' { c } else { b' ' };
        self.stripped.push(kept);
        self.i += 1;
    }

    fn line_comment(&mut self) {
        let (line, trailing) = (self.line, self.code);
        let kind = match (self.peek(2), self.peek(3)) {
            (b'/', b'/') => Kind::Line,
            (b'/', _) => Kind::OuterDoc,
            (b'!', _) => Kind::InnerDoc,
            _ => Kind::Line,
        };
        let start = self.i;
        while self.i < self.b.len() && self.b[self.i] != b'\n' {
            self.bump(false);
        }
        let body = String::from_utf8_lossy(&self.b[start + 2..self.i]);
        let body = body.trim_start_matches(['/', '!']);
        self.out.push(Raw {
            kind,
            line,
            end: line,
            words: words(body),
            trailing,
            fence: body.trim_start().starts_with("```"),
            semicolon: prose_semicolon(body),
        });
    }

    fn block_comment(&mut self) {
        let (line, trailing, start) = (self.line, self.code, self.i);
        let mut depth = 0;
        while self.i < self.b.len() {
            if self.peek(0) == b'/' && self.peek(1) == b'*' {
                depth += 1;
                self.bump(false);
                self.bump(false);
            } else if self.peek(0) == b'*' && self.peek(1) == b'/' {
                depth -= 1;
                self.bump(false);
                self.bump(false);
                if depth == 0 || !self.rust {
                    break;
                }
            } else {
                self.bump(false);
            }
        }
        let body = String::from_utf8_lossy(&self.b[start..self.i]);
        let body = body.trim_start_matches("/*").trim_end_matches("*/");
        self.out.push(Raw {
            kind: Kind::Block,
            line,
            end: self.line,
            words: words(body),
            trailing,
            fence: false,
            semicolon: prose_semicolon(body),
        });
    }

    /// A quoted literal: skips escapes and keeps every byte.
    fn quoted(&mut self, q: u8) {
        self.bump(true);
        while self.i < self.b.len() && self.b[self.i] != q {
            if self.b[self.i] == b'\\' {
                self.bump(true);
            }
            if self.i < self.b.len() {
                self.bump(true);
            }
        }
        if self.i < self.b.len() {
            self.bump(true);
        }
    }

    /// r"..", r#".."#, br#".."#: returns false when not one.
    fn raw_string(&mut self) -> bool {
        let prev = if self.i == 0 {
            b' '
        } else {
            self.b[self.i - 1]
        };
        if prev.is_ascii_alphanumeric() || prev == b'_' {
            return false;
        }
        let mut k = usize::from(self.peek(0) == b'b');
        if self.peek(k) != b'r' {
            return false;
        }
        k += 1;
        let hashes = (0..).take_while(|&h| self.peek(k + h) == b'#').count();
        if self.peek(k + hashes) != b'"' {
            return false;
        }
        for _ in 0..k + hashes + 1 {
            self.bump(true);
        }
        let close: Vec<u8> = std::iter::once(b'"')
            .chain(std::iter::repeat_n(b'#', hashes))
            .collect();
        while self.i < self.b.len() && !self.b[self.i..].starts_with(&close) {
            self.bump(true);
        }
        for _ in 0..close.len().min(self.b.len() - self.i) {
            self.bump(true);
        }
        true
    }

    /// A char literal, or for Rust possibly a lifetime.
    fn quote(&mut self) {
        let is_char = !self.rust || self.peek(1) == b'\\' || self.peek(2) == b'\'';
        if is_char {
            self.quoted(b'\'');
        } else {
            self.bump(true);
        }
    }

    fn run(&mut self) {
        while self.i < self.b.len() {
            let (c, n) = (self.peek(0), self.peek(1));
            if c == b'/' && n == b'/' {
                self.line_comment();
            } else if c == b'/' && n == b'*' {
                self.block_comment();
            } else if c == b'"' {
                self.code = true;
                self.quoted(b'"');
            } else if c == b'\'' {
                self.code = true;
                self.quote();
            } else if self.rust && (c == b'r' || c == b'b') && self.raw_string() {
                self.code = true;
            } else {
                if !c.is_ascii_whitespace() {
                    self.code = true;
                }
                self.bump(true);
            }
        }
    }
}

fn scanner(text: &str, rust: bool) -> Scanner<'_> {
    let mut s = Scanner {
        b: text.as_bytes(),
        i: 0,
        line: 1,
        code: false,
        rust,
        out: Vec::new(),
        stripped: Vec::with_capacity(text.len()),
    };
    s.run();
    s
}

pub fn scan(text: &str, rust: bool) -> Vec<Raw> {
    scanner(text, rust).out
}

/// The text with every comment blanked out, line numbers kept.
pub fn strip(text: &str, rust: bool) -> String {
    String::from_utf8_lossy(&scanner(text, rust).stripped).into_owned()
}

/// Adjacent whole-line comments of one kind count as one.
pub fn group(raw: &[Raw]) -> Vec<Comment> {
    let mut out: Vec<Comment> = Vec::new();
    let mut prev: Option<&Raw> = None;
    for r in raw {
        let joins = prev.is_some_and(|p| {
            p.kind == r.kind
                && r.kind != Kind::Block
                && !p.trailing
                && !r.trailing
                && p.end + 1 == r.line
        });
        match out.last_mut() {
            Some(last) if joins => last.words += r.words,
            _ => out.push(Comment {
                line: r.line,
                words: r.words,
            }),
        }
        prev = Some(r);
    }
    out
}

/// Lines of comments with a `;` outside code spans and blocks.
pub fn semicolons(raw: &[Raw]) -> Vec<usize> {
    let mut block = false;
    let mut prev: Option<&Raw> = None;
    let mut out = Vec::new();
    for r in raw {
        // An unclosed block ends where its comment does.
        if prev.is_none_or(|p| p.kind != r.kind || p.end + 1 != r.line) {
            block = false;
        }
        if r.fence {
            block = !block;
        } else if r.semicolon && !block {
            out.push(r.line);
        }
        prev = Some(r);
    }
    out
}

/// The text after each `#` that starts a comment, by line.
pub fn hash_comments(text: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 && line.starts_with("#!") {
            continue;
        }
        let (mut quote, mut prev) = (None, ' ');
        for (at, c) in line.char_indices() {
            match (quote, c) {
                (None, '\'' | '"') => quote = Some(c),
                (Some(q), _) if c == q => quote = None,
                (None, '#') if prev.is_whitespace() => {
                    out.push((i + 1, &line[at + 1..]));
                    break;
                }
                _ => {}
            }
            prev = c;
        }
    }
    out
}

const SEMICOLON: &str = "`;` in prose: split the sentence, or quote code in backticks";

pub fn check_hash(src: &Source, out: &mut Vec<Violation>) {
    for (line, body) in hash_comments(&src.text) {
        if prose_semicolon(body) {
            out.push(src.violation(line, "comment-semicolon", SEMICOLON.into()));
        }
    }
}

pub fn check(src: &Source, rust: bool, out: &mut Vec<Violation>) {
    let raw = scan(&src.text, rust);
    for line in semicolons(&raw) {
        out.push(src.violation(line, "comment-semicolon", SEMICOLON.into()));
    }
    let max = LIMITS.max_comment_words;
    for c in group(&raw) {
        if c.words > max {
            out.push(src.violation(
                c.line,
                "comment-words",
                format!("comment of {} words (max {max})", c.words),
            ));
        }
    }
}
