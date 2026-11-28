//! A kernel and its includes, as one translation unit.

use crate::{Backend, EDITION, HEADERS, Import};
use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt::{self, Write};

/// The engine's choices for one compile.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub backend: Backend,
    /// Some kept pages are not on the device.
    pub remote_kv: bool,
    /// Record timing marks.
    pub marks: bool,
}

impl Options {
    /// Kept pages on the device, and no timing marks.
    pub const fn plain(backend: Backend) -> Self {
        Self {
            backend,
            remote_kv: false,
            marks: false,
        }
    }
}

/// The text to compile, and what it imports.
#[derive(Debug)]
pub struct Flat {
    pub text: String,
    /// Sorted, without repeats.
    pub imports: Vec<Import>,
}

/// Local files in include order, each after what it includes.
struct Files<'a> {
    load: &'a dyn Fn(&str) -> Option<String>,
    done: Vec<(String, Cow<'a, str>, Scan)>,
    open: Vec<String>,
}

impl<'a> Files<'a> {
    fn visit(&mut self, name: &str, text: Cow<'a, str>) -> Result<(), Error> {
        let whole = |problem| Error {
            file: name.into(),
            line: 0,
            problem,
        };
        let found = scan(name, &text)?;
        self.open.push(name.into());
        for path in &found.locals {
            if self.open.contains(path) {
                return Err(whole(Problem::Cycle(path.clone())));
            }
            if self.done.iter().any(|(n, ..)| n == path) {
                continue;
            }
            let text = (self.load)(path).ok_or_else(|| whole(Problem::Missing(path.clone())))?;
            self.visit(path, Cow::Owned(text))?;
        }
        self.open.pop();
        self.done.push((name.into(), text, found));
        Ok(())
    }
}

fn line(out: &mut String, n: usize, name: &str) {
    let _ = writeln!(out, "#line {n} \"{name}\"");
}

/// Appends `text` with sorted `includes` line numbers blanked out.
fn append_stripped(out: &mut String, text: &str, includes: &[usize]) {
    if !text.contains('\r') {
        let Some(&last) = includes.last() else {
            out.push_str(text);
            if !text.is_empty() && !text.ends_with('\n') {
                out.push('\n');
            }
            return;
        };
        let mut inc = 0;
        let mut pos = 0;
        for n in 0..=last {
            let rest = &text[pos..];
            let (slice, next) = match rest.find('\n') {
                Some(nl) => (&rest[..nl], pos + nl + 1),
                None => (rest, text.len()),
            };
            if inc < includes.len() && includes[inc] == n {
                inc += 1;
            } else {
                out.push_str(slice);
            }
            out.push('\n');
            pos = next;
            if pos >= text.len() {
                return;
            }
        }
        let tail = &text[pos..];
        out.push_str(tail);
        if !tail.is_empty() && !tail.ends_with('\n') {
            out.push('\n');
        }
        return;
    }
    let mut inc = 0;
    for (n, l) in text.lines().enumerate() {
        if inc < includes.len() && includes[inc] == n {
            inc += 1;
        } else {
            out.push_str(l);
        }
        out.push('\n');
    }
}

/// Flattens kernel `name`. `load` reads other recipe files.
pub fn flatten(
    name: &str,
    text: &str,
    load: &dyn Fn(&str) -> Option<String>,
    options: &Options,
) -> Result<Flat, Error> {
    let whole = |problem| Error {
        file: name.into(),
        line: 0,
        problem,
    };
    let backend = options.backend;
    if !Backend::of(name).contains(&backend) {
        return Err(whole(Problem::Extension(backend)));
    }
    let mut files = Files {
        load,
        done: Vec::new(),
        open: Vec::new(),
    };
    files.visit(name, Cow::Borrowed(text))?;
    let imports: BTreeSet<Import> = files
        .done
        .iter()
        .flat_map(|(.., s)| s.imports.iter().copied())
        .collect();
    let imports: Vec<Import> = imports.into_iter().collect();
    if imports.is_empty() {
        return Err(whole(Problem::NoImports));
    }
    let no_metal = |i: &&Import| i.persistent() || **i == Import::Trellis;
    if let Some(i) = imports
        .iter()
        .find(no_metal)
        .filter(|_| backend == Backend::Metal)
    {
        return Err(whole(Problem::Unsupported(backend, i.name())));
    }
    let collective = imports.iter().any(|i| i.collective());
    if options.remote_kv && !(collective && imports.contains(&Import::Kept)) {
        return Err(whole(Problem::RemoteKv));
    }
    let headers: Vec<_> = HEADERS
        .iter()
        .filter(|h| (h.needed)(&imports, backend))
        .collect();
    let texts: Vec<_> = headers.iter().map(|h| (h.name, (h.text)())).collect();
    let cap: usize = 256
        + texts
            .iter()
            .map(|(n, t)| t.len() + n.len() + 16)
            .sum::<usize>()
        + files
            .done
            .iter()
            .map(|(f, t, _)| t.len() + f.len() + 16)
            .sum::<usize>();
    let mut out = String::with_capacity(cap);
    let mut define = |name: &str| {
        let _ = writeln!(out, "#define {name} 1");
    };
    define(backend.define());
    if options.remote_kv {
        define("TESSERA_REMOTE_KV");
    }
    if options.marks {
        define("TESSERA_MARKS");
    }
    for cap in headers.iter().filter_map(|h| h.cap) {
        define(cap);
    }
    for (name, body) in &texts {
        line(&mut out, 1, name);
        out.push_str(body);
    }
    for (file, text, found) in &files.done {
        line(&mut out, 1, file);
        append_stripped(&mut out, text, &found.includes);
    }
    Ok(Flat { text: out, imports })
}

/// The fault, without the place.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum Problem {
    #[error("<{0}> names no edition: write <tessera/{EDITION}/...>")]
    Bare(String),
    #[error("edition {0}, but this engine reads edition {EDITION}")]
    Edition(String),
    #[error("no import named {0}")]
    Unknown(String),
    #[error("<{0}> is not a tessera import")]
    Foreign(String),
    #[error("{0:?} must be a path inside the recipe")]
    Path(String),
    #[error("an include must name a literal path")]
    Operand,
    #[error("includes belong in the block at the top")]
    Late,
    #[error("no file {0:?} in the recipe")]
    Missing(String),
    #[error("{0:?} includes itself")]
    Cycle(String),
    #[error("a kernel must import at least one tessera header")]
    NoImports,
    #[error("this extension does not build for {0:?}")]
    Extension(crate::Backend),
    #[error("{0:?} cannot import {1}")]
    Unsupported(crate::Backend, &'static str),
    #[error("remote kept rows need kept and a collective")]
    RemoteKv,
}

/// A problem, and the file and line it is on.
#[derive(Debug, PartialEq)]
pub struct Error {
    pub file: String,
    /// One-based, or zero for the whole file.
    pub line: usize,
    pub problem: Problem,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.file, self.line, self.problem)
    }
}

impl std::error::Error for Error {}

/// One file's imports, local includes, and include lines.
#[derive(Debug, Default)]
pub(crate) struct Scan {
    pub imports: Vec<Import>,
    pub locals: Vec<String>,
    /// Zero-based lines the flattener blanks.
    pub includes: Vec<usize>,
}

/// Next `//` or `/*` in `s`, as `(index, is_block)`.
fn next_comment(s: &str) -> Option<(usize, bool)> {
    let bytes = s.as_bytes();
    let mut at = 0;
    while let Some(rel) = s[at..].find('/') {
        let i = at + rel;
        match bytes.get(i + 1) {
            Some(b'/') => return Some((i, false)),
            Some(b'*') => return Some((i, true)),
            _ => at = i + 1,
        }
    }
    None
}

/// The part of `line` outside comments, borrowing when unsplit.
fn code<'a>(line: &'a str, open: &mut bool) -> Cow<'a, str> {
    let mut rest = line;
    if *open {
        let Some(end) = rest.find("*/") else {
            return Cow::Borrowed("");
        };
        rest = &rest[end + 2..];
        *open = false;
    }
    let Some((pos, block)) = next_comment(rest) else {
        return Cow::Borrowed(rest);
    };
    if !block {
        return Cow::Borrowed(&rest[..pos]);
    }
    *open = true;
    let after = &rest[pos + 2..];
    let Some(end) = after.find("*/") else {
        return Cow::Borrowed(&rest[..pos]);
    };
    *open = false;
    let mut out = String::with_capacity(rest.len());
    out.push_str(&rest[..pos]);
    out.push(' ');
    rest = &after[end + 2..];
    loop {
        if *open {
            let Some(end) = rest.find("*/") else {
                return Cow::Owned(out);
            };
            rest = &rest[end + 2..];
            *open = false;
        }
        match next_comment(rest) {
            Some((p, false)) => {
                out.push_str(&rest[..p]);
                return Cow::Owned(out);
            }
            Some((p, true)) => {
                out.push_str(&rest[..p]);
                out.push(' ');
                rest = &rest[p + 2..];
                *open = true;
            }
            None => {
                out.push_str(rest);
                return Cow::Owned(out);
            }
        }
    }
}

/// The directive after `#`, if `code` is one.
fn directive(code: &str) -> Option<&str> {
    Some(code.trim_start().strip_prefix('#')?.trim_start())
}

fn local(path: &str) -> Result<String, Problem> {
    let bad = path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path.contains(':')
        || path
            .split('/')
            .any(|p| p == ".." || p == "." || p.is_empty());
    if bad {
        return Err(Problem::Path(path.into()));
    }
    Ok(path.into())
}

fn tessera(target: &str) -> Result<Import, Problem> {
    let mut parts = target.split('/');
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some("tessera"), Some(_), None, None) => Err(Problem::Bare(target.into())),
        (Some("tessera"), Some(edition), Some(name), None)
            if edition.parse::<u32>() == Ok(EDITION) =>
        {
            Import::parse(name).ok_or_else(|| Problem::Unknown(name.into()))
        }
        (Some("tessera"), Some(edition), Some(_), None) => Err(Problem::Edition(edition.into())),
        _ => Err(Problem::Foreign(target.into())),
    }
}

impl Scan {
    /// Reads one `#include` operand in the leading block.
    fn include(&mut self, operand: &str) -> Result<(), Problem> {
        let operand = operand.trim();
        if operand.ends_with('\\') {
            return Err(Problem::Operand);
        }
        if let Some(target) = operand.strip_prefix('<').and_then(|o| o.strip_suffix('>')) {
            self.imports.push(tessera(target)?);
        } else if let Some(path) = operand.strip_prefix('"').and_then(|o| o.strip_suffix('"')) {
            self.locals.push(local(path)?);
        } else {
            return Err(Problem::Operand);
        }
        Ok(())
    }
}

/// Scans `text`. Includes go only in the top block.
pub(crate) fn scan(file: &str, text: &str) -> Result<Scan, Error> {
    let at = |line: usize, problem| Error {
        file: file.into(),
        line: line + 1,
        problem,
    };
    let has_probe = text.contains("__has_include");
    let mut scan = Scan::default();
    let mut open = false;
    let mut leading = true;
    for (n, line) in text.lines().enumerate() {
        let code = code(line, &mut open);
        let code = code.trim();
        if code.is_empty() {
            continue;
        }
        let late = (has_probe && code.contains("__has_include")) || code.starts_with("%:");
        let kind = directive(code);
        let named = |word: &str| kind.is_some_and(|d| d.starts_with(word));
        if leading && named("include") && !late {
            let operand = &kind.unwrap_or_default()["include".len()..];
            scan.include(operand).map_err(|p| at(n, p))?;
            scan.includes.push(n);
            continue;
        }
        leading = false;
        if late || named("include") || named("import") || named("embed") {
            return Err(at(n, Problem::Late));
        }
    }
    Ok(scan)
}
