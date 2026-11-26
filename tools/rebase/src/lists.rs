//! Conflicts where both sides only add list entries.

use std::collections::HashSet;

use crate::code::in_code;
use crate::hunk::{Hunk, Part, Refusal, parts};
use crate::names::{Side, join};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A whole `use` statement.
    Use,
    /// A name inside a `use` group.
    Item,
    Mod,
    /// A `Cargo.toml` dependency.
    Dep,
}

/// One entry with its attribute and comment lines.
type Entry<'s, 'a> = &'s [&'a str];

const NOT_LIST: &str = "a line is not a `use`, `mod` or dependency entry";
const EDITED: &str = "a side changes or removes an existing entry";

enum Joined<'a> {
    /// Lines of the two sides, as they were.
    Lines(Vec<&'a str>),
    /// New text for edited `use` lines.
    Text(String),
}
const NOT_GROUP: &str = "the lines are not inside a `use` group";
const NOT_TABLE: &str = "the lines are not inside a dependency table";
const NOT_CODE: &str = "the conflict is inside a string or a comment";

/// Resolves each hunk, or names the first refusal.
pub fn resolve(path: &str, text: &str) -> Result<String, Refusal> {
    let toml = path.rsplit('/').next() == Some("Cargo.toml");
    if !toml && !path.ends_with(".rs") {
        let why = "the file is not Rust source or `Cargo.toml`";
        return Err(Refusal { line: 1, why });
    }
    let mut out = String::new();
    let mut before: Vec<&str> = Vec::new();
    for part in parts(text)? {
        match part {
            Part::Line(line) => {
                out.push_str(line);
                before.push(line);
            }
            Part::Hunk(hunk) => {
                let line = hunk.line;
                if !toml && !in_code(&before) {
                    return Err(Refusal {
                        line,
                        why: NOT_CODE,
                    });
                }
                match union(&hunk, &before, toml).map_err(|why| Refusal { line, why })? {
                    Joined::Text(text) => out.push_str(&text),
                    Joined::Lines(lines) => {
                        out.push_str(&lines.concat());
                        before.extend(lines);
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Both sides' entries, with the base entries in place.
fn union<'a>(hunk: &Hunk<'a>, before: &[&str], toml: bool) -> Result<Joined<'a>, &'static str> {
    let (ours_kind, ours) = entries(&hunk.ours, toml).ok_or(NOT_LIST)?;
    let (base_kind, base) = entries(&hunk.base, toml).ok_or(NOT_LIST)?;
    let (theirs_kind, theirs) = entries(&hunk.theirs, toml).ok_or(NOT_LIST)?;
    let mut kind = None;
    for found in [ours_kind, base_kind, theirs_kind].into_iter().flatten() {
        kind = Some(both(kind, found).ok_or(NOT_LIST)?);
    }
    let kind = kind.ok_or(NOT_LIST)?;
    match kind {
        Kind::Item if !in_group(before) => return Err(NOT_GROUP),
        Kind::Dep if !in_table(before) => return Err(NOT_TABLE),
        _ => {}
    }
    let (Some(ours_gaps), Some(theirs_gaps)) = (gaps(&ours, &base), gaps(&theirs, &base)) else {
        // An edited `use` line can still keep each imported name.
        if !matches!(kind, Kind::Use | Kind::Item) {
            return Err(EDITED);
        }
        let items = kind == Kind::Item;
        let side = |entries| Side { entries, items };
        let text = join(&side(&ours), &side(&base), &side(&theirs));
        return text.map(Joined::Text).ok_or(EDITED);
    };
    let mut seen: HashSet<Entry> = ours.iter().copied().collect();
    let mut out = Vec::new();
    for (i, (ours, theirs)) in ours_gaps.iter().zip(&theirs_gaps).enumerate() {
        let adds = ours.iter().map(|entry| (entry, true));
        for (entry, ours) in adds.chain(theirs.iter().map(|entry| (entry, false))) {
            let blank = entry[0].trim().is_empty();
            let last: Option<&&str> = out.last().or(before.last());
            let doubled = blank && last.is_none_or(|line| line.trim().is_empty());
            if !doubled && (ours || blank || seen.insert(entry)) {
                out.extend_from_slice(entry);
            }
        }
        if let Some(entry) = base.get(i) {
            out.extend_from_slice(entry);
        }
    }
    Ok(Joined::Lines(out))
}

/// The kind of two entries. `use` and `mod` mix.
fn both(first: Option<Kind>, next: Kind) -> Option<Kind> {
    match (first, next) {
        (None, _) => Some(next),
        (Some(first), _) if first == next => Some(next),
        (Some(Kind::Use | Kind::Mod), Kind::Use | Kind::Mod) => Some(Kind::Mod),
        _ => None,
    }
}

/// A side's added entries around the base entries, else `None`.
fn gaps<'s, 'a>(side: &[Entry<'s, 'a>], base: &[Entry<'s, 'a>]) -> Option<Vec<Vec<Entry<'s, 'a>>>> {
    let mut out = vec![Vec::new()];
    for entry in side {
        if base.get(out.len() - 1) == Some(entry) {
            out.push(Vec::new());
        } else {
            out.last_mut()?.push(*entry);
        }
    }
    (out.len() == base.len() + 1).then_some(out)
}

/// Splits lines into entries of one kind.
fn entries<'s, 'a>(lines: &'s [&'a str], toml: bool) -> Option<(Option<Kind>, Vec<Entry<'s, 'a>>)> {
    let mut out = Vec::new();
    let mut found = None;
    let mut start = 0;
    let mut next = 0;
    while next < lines.len() {
        let line = lines[next].trim();
        next += 1;
        if line.is_empty() {
            if start + 1 != next {
                return None;
            }
        } else if prefix(line, toml) {
            continue;
        } else {
            let kind = kind(line, toml)?;
            found = Some(both(found, kind)?);
            if matches!(kind, Kind::Use | Kind::Item) {
                next = close(lines, next - 1, kind)?;
            }
        }
        out.push(&lines[start..next]);
        start = next;
    }
    (start == lines.len()).then_some((found, out))
}

/// An attribute or comment line before an entry.
pub fn prefix(line: &str, toml: bool) -> bool {
    if toml {
        return line.starts_with('#');
    }
    line.starts_with("//") || (line.starts_with("#[") && line.ends_with(']'))
}

/// The kind of entry that a line starts.
fn kind(line: &str, toml: bool) -> Option<Kind> {
    if toml {
        return dep_line(line).then_some(Kind::Dep);
    }
    let line = code_part(line);
    let rest = after_vis(line);
    if rest.starts_with("use ") {
        return Some(Kind::Use);
    }
    if let Some(name) = rest.strip_prefix("mod ") {
        let name = name.strip_suffix(';')?;
        let ident = !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        return ident.then_some(Kind::Mod);
    }
    (rest.len() == line.len()).then_some(Kind::Item)
}

/// The line without `pub` or `pub(...)`.
pub fn after_vis(line: &str) -> &str {
    let Some(rest) = line.strip_prefix("pub") else {
        return line;
    };
    let rest = match rest.strip_prefix('(') {
        Some(scope) => match scope.split_once(')') {
            Some((_, rest)) => rest,
            None => return line,
        },
        None => rest,
    };
    if rest.starts_with(' ') {
        rest.trim_start()
    } else {
        line
    }
}

/// The line without a trailing `//` comment.
fn code_part(line: &str) -> &str {
    line.split("//").next().unwrap_or_default().trim()
}

fn path_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || "_:{},* \t".contains(c)
}

/// The index after a `use` statement or group item.
fn close(lines: &[&str], first: usize, kind: Kind) -> Option<usize> {
    let end = if kind == Kind::Use { ';' } else { ',' };
    let mut depth = 0usize;
    for (i, line) in lines.iter().enumerate().skip(first) {
        let mut body = code_part(line);
        if i == first && kind == Kind::Use {
            body = after_vis(body).strip_prefix("use ")?;
        }
        for c in body.chars() {
            match c {
                '{' => depth += 1,
                '}' => depth = depth.checked_sub(1)?,
                _ => {}
            }
        }
        let closed = depth == 0 && body.ends_with(end);
        if closed {
            body = &body[..body.len() - 1];
        }
        if !body.chars().all(path_char) || (i == first && body.trim().is_empty()) {
            return None;
        }
        if closed {
            return Some(i + 1);
        }
    }
    None
}

/// A `key = value` line with a one-line value.
fn dep_line(line: &str) -> bool {
    let Some((key, value)) = line.split_once('=') else {
        return false;
    };
    let (key, value) = (key.trim(), value.trim());
    let mut depth = 0usize;
    for c in value.chars() {
        match c {
            '{' | '[' => depth += 1,
            '}' | ']' => match depth.checked_sub(1) {
                Some(less) => depth = less,
                None => return false,
            },
            _ => {}
        }
    }
    let key_ok = !key.is_empty()
        && key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_-.".contains(c));
    let quotes = value.matches('"').count();
    key_ok && !value.is_empty() && depth == 0 && quotes % 2 == 0 && !value.contains(['\'', '\\'])
}

/// True when the hunk sits in a `use` group.
fn in_group(before: &[&str]) -> bool {
    for line in before.iter().rev() {
        let line = line.trim();
        if line.is_empty() || prefix(line, false) {
            continue;
        }
        if kind(line, false) == Some(Kind::Item) && close(&[line], 0, Kind::Item) == Some(1) {
            continue;
        }
        let Some(body) = after_vis(line).strip_prefix("use ") else {
            return false;
        };
        let open = body
            .strip_suffix('{')
            .is_some_and(|path| !path.contains(['{', '}']));
        return open && body.chars().all(path_char);
    }
    false
}

/// True when the table above holds dependencies.
fn in_table(before: &[&str]) -> bool {
    for line in before.iter().rev() {
        let line = line.trim();
        if line.starts_with('[') {
            return !line.starts_with("[[") && line.ends_with("dependencies]");
        }
    }
    false
}
