//! Path edits that move a crate into a merged crate.

use std::collections::BTreeMap;
use std::ops::Range;

use crate::tokens::{Cursor, Tok, apply, tokens};

/// How the paths in one crate's files change.
pub(crate) struct Moves<'a> {
    /// The module `crate::` now names, or empty.
    pub module: &'a str,
    /// Replacement paths, by crate name.
    pub names: &'a BTreeMap<String, String>,
    /// The crate's directory in its package, or empty.
    pub dir: &'a str,
    /// New paths for `include!` files, by their old path.
    pub include: &'a dyn Fn(&str) -> Option<String>,
}

type Edit = (Range<usize>, String);

pub(crate) fn rewrite(src: &str, moves: &Moves) -> Result<String, String> {
    let toks = tokens(src)?;
    let t = Cursor(&toks);
    let mut edits = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        if let Some(end) = alias(&t, i, moves) {
            edits.push((t.span(i, end), String::new()));
            i = end + 1;
            continue;
        }
        let edit = in_crate(&t, i, moves)
            .or_else(|| renamed(&t, i, moves))
            .or_else(|| in_dir(src, &t, i, moves))
            .or_else(|| included(src, &t, i, moves));
        edits.extend(edit);
        i += 1;
    }
    apply(src, edits)
}

/// The last token of an alias of a merged crate.
fn alias(t: &Cursor, i: usize, m: &Moves) -> Option<usize> {
    let named = |j: usize| t.ident(j).is_some_and(|n| m.names.contains_key(n));
    if t.is(i, "extern") && t.is(i + 1, "crate") && t.is(i + 2, "self") {
        return (t.is(i + 3, "as") && named(i + 4) && t.punct(i + 5, ';')).then_some(i + 5);
    }
    if t.is(i, "use") && t.is(i + 1, "crate") && t.is(i + 2, "as") {
        return (named(i + 3) && t.punct(i + 4, ';')).then_some(i + 4);
    }
    // `use name as _` links a crate. Modules need none.
    if t.is(i, "use") && named(i + 1) && t.is(i + 2, "as") && t.is(i + 3, "_") {
        return t.punct(i + 4, ';').then_some(i + 4);
    }
    if t.is(i, "self") && t.is(i + 1, "as") && named(i + 2) {
        return Some(if t.punct(i + 3, ',') { i + 3 } else { i + 2 });
    }
    None
}

/// Keywords that may come before a leading `::`.
const KEYWORDS: [&str; 16] = [
    "use", "as", "in", "for", "impl", "dyn", "where", "let", "mut", "ref", "return", "pub", "type",
    "match", "if", "move",
];

/// `crate::`, `$crate::` and `crate = crate` gain the module.
fn in_crate(t: &Cursor, i: usize, m: &Moves) -> Option<Edit> {
    if m.module.is_empty() || !t.is(i, "crate") {
        return None;
    }
    let value = i >= 2 && t.punct(i - 1, '=') && t.is(i - 2, "crate");
    let at = match (t.colons(i + 1), value) {
        (true, _) => t.0[i + 2].at.end,
        (false, true) => t.0[i].at.end,
        _ => return None,
    };
    let text = if value {
        format!("::{}", m.module)
    } else {
        format!("{}::", m.module)
    };
    Some((at..at, text))
}

/// A merged crate's name as the first path segment.
fn renamed(t: &Cursor, i: usize, m: &Moves) -> Option<Edit> {
    let to = m.names.get(t.ident(i)?)?;
    let alias = i > 0 && t.is(i - 1, "as") && !t.colons(i + 1);
    let field = i > 1 && t.punct(i - 1, '.') && !t.punct(i - 2, '.');
    if alias || field {
        return None;
    }
    if i < 2 || !t.colons(i - 2) {
        return Some((t.0[i].at.clone(), to.clone()));
    }
    // Later segments are not crate names, except after `#var`.
    let quoted = i >= 4 && t.punct(i - 4, '#');
    let segment = match t.tok(i.wrapping_sub(3)) {
        Some(Tok::Ident(w)) => !KEYWORDS.contains(&w.as_str()),
        Some(Tok::Punct('>', _)) => true,
        _ => false,
    };
    if i >= 3 && !quoted && segment {
        return None;
    }
    let text = if to.starts_with("crate") {
        to.clone()
    } else {
        format!("::{to}")
    };
    Some((t.span(i - 2, i), text))
}

/// Paths under `CARGO_MANIFEST_DIR` gain the crate's directory.
fn in_dir(src: &str, t: &Cursor, i: usize, m: &Moves) -> Option<Edit> {
    if m.dir.is_empty() {
        return None;
    }
    let lit = |j: usize| (t.tok(j) == Some(&Tok::Lit)).then(|| &src[t.0[j].at.clone()]);
    if let Some(text) = lit(i).filter(|s| s.contains("$CARGO_MANIFEST_DIR/")) {
        let to = format!("$CARGO_MANIFEST_DIR/{}/", m.dir);
        return Some((t.0[i].at.clone(), text.replace("$CARGO_MANIFEST_DIR/", &to)));
    }
    let env = t.is(i, "env") && t.punct(i + 1, '!') && t.tok(i + 2) == Some(&Tok::Open);
    if env && lit(i + 3) == Some("\"CARGO_MANIFEST_DIR\"") && t.tok(i + 4) == Some(&Tok::Close) {
        let text = format!("concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/{}\")", m.dir);
        return Some((t.span(i, i + 4), text));
    }
    None
}

/// `include!` files that move get their new path.
fn included(src: &str, t: &Cursor, i: usize, m: &Moves) -> Option<Edit> {
    let call = t.is(i, "include") && t.punct(i + 1, '!') && t.tok(i + 2) == Some(&Tok::Open);
    if !call || t.tok(i + 3) != Some(&Tok::Lit) {
        return None;
    }
    let at = t.0[i + 3].at.clone();
    let old = src[at.clone()].trim_matches('"');
    (m.include)(old).map(|new| (at, format!("{new:?}")))
}

/// Names of the `#[macro_export]` macros in `src`.
pub(crate) fn exports(src: &str) -> Result<Vec<String>, String> {
    let toks = tokens(src)?;
    let t = Cursor(&toks);
    let mut out = Vec::new();
    for i in 0..toks.len() {
        let marked =
            t.punct(i, '#') && t.tok(i + 1) == Some(&Tok::Open) && t.is(i + 2, "macro_export");
        if marked && t.is(i + 4, "macro_rules") && t.punct(i + 5, '!') {
            out.extend(t.ident(i + 6).map(String::from));
        }
    }
    Ok(out)
}
