//! A source file as flat tokens with byte ranges.

use std::ops::Range;

use proc_macro2::{Delimiter, Spacing, TokenStream, TokenTree};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Tok {
    Ident(String),
    /// A punctuation character, and whether the next one joins it.
    Punct(char, bool),
    Lit,
    Open,
    Close,
}

pub(crate) struct Token {
    pub tok: Tok,
    pub at: Range<usize>,
}

pub(crate) fn tokens(src: &str) -> Result<Vec<Token>, String> {
    let stream: TokenStream = src
        .parse()
        .map_err(|e: proc_macro2::LexError| e.to_string())?;
    let mut out = Vec::new();
    flatten(stream, &mut out);
    Ok(out)
}

fn flatten(stream: TokenStream, out: &mut Vec<Token>) {
    for tree in stream {
        let (tok, at) = match tree {
            TokenTree::Group(g) if g.delimiter() == Delimiter::None => {
                flatten(g.stream(), out);
                continue;
            }
            TokenTree::Group(g) => {
                out.push(Token {
                    tok: Tok::Open,
                    at: g.span_open().byte_range(),
                });
                flatten(g.stream(), out);
                (Tok::Close, g.span_close().byte_range())
            }
            TokenTree::Ident(i) => (Tok::Ident(i.to_string()), i.span().byte_range()),
            TokenTree::Punct(p) => (
                Tok::Punct(p.as_char(), p.spacing() == Spacing::Joint),
                p.span().byte_range(),
            ),
            TokenTree::Literal(l) => (Tok::Lit, l.span().byte_range()),
        };
        out.push(Token { tok, at });
    }
}

/// Reads tokens by index, with misses past either end.
pub(crate) struct Cursor<'a>(pub &'a [Token]);

impl Cursor<'_> {
    pub fn ident(&self, i: usize) -> Option<&str> {
        match self.0.get(i).map(|t| &t.tok) {
            Some(Tok::Ident(s)) => Some(s),
            _ => None,
        }
    }

    pub fn is(&self, i: usize, word: &str) -> bool {
        self.ident(i) == Some(word)
    }

    pub fn punct(&self, i: usize, c: char) -> bool {
        matches!(self.0.get(i).map(|t| &t.tok), Some(Tok::Punct(p, _)) if *p == c)
    }

    /// A `::` starting at `i`.
    pub fn colons(&self, i: usize) -> bool {
        matches!(self.0.get(i).map(|t| &t.tok), Some(Tok::Punct(':', true)))
            && self.punct(i + 1, ':')
    }

    pub fn tok(&self, i: usize) -> Option<&Tok> {
        self.0.get(i).map(|t| &t.tok)
    }

    /// Bytes from token `a` through token `b`.
    pub fn span(&self, a: usize, b: usize) -> Range<usize> {
        self.0[a].at.start..self.0[b].at.end
    }
}

/// Applies non-overlapping edits to `src`.
pub(crate) fn apply(src: &str, mut edits: Vec<(Range<usize>, String)>) -> Result<String, String> {
    edits.sort_by_key(|(at, _)| at.start);
    let mut out = String::with_capacity(src.len());
    let mut from = 0;
    for (at, text) in edits {
        if at.start < from {
            return Err(format!("edits overlap at byte {}", at.start));
        }
        out.push_str(&src[from..at.start]);
        out.push_str(&text);
        from = at.end;
    }
    out.push_str(&src[from..]);
    Ok(out)
}
