//! A kernel and its includes, as one translation unit.

use crate::{Backend, EDITION, HEADERS, Import};
use std::collections::BTreeSet;
use std::fmt;

/// The engine's choices for one compile.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub backend: Backend,
    /// Some kept pages are not on the device.
    pub remote_kv: bool,
    /// Record timing marks.
    pub marks: bool,
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
    done: Vec<(String, String, Scan)>,
    open: Vec<String>,
}

impl Files<'_> {
    fn visit(&mut self, name: &str, text: String) -> Result<(), Error> {
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
            self.visit(path, text)?;
        }
        self.open.pop();
        self.done.push((name.into(), text, found));
        Ok(())
    }
}

fn line(out: &mut String, n: usize, name: &str) {
    out.push_str(&format!("#line {n} \"{name}\"\n"));
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
    files.visit(name, text.into())?;
    let imports: BTreeSet<Import> = files
        .done
        .iter()
        .flat_map(|(.., s)| s.imports.clone())
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
    let mut out = String::new();
    let mut define = |name: &str| out.push_str(&format!("#define {name} 1\n"));
    define(backend.define());
    if options.remote_kv {
        define("TESSERA_REMOTE_KV");
    }
    if options.marks {
        define("TESSERA_MARKS");
    }
    let headers = HEADERS.iter().filter(|h| (h.needed)(&imports, backend));
    let headers: Vec<_> = headers.collect();
    for cap in headers.iter().filter_map(|h| h.cap) {
        define(cap);
    }
    for h in headers {
        line(&mut out, 1, h.name);
        out.push_str((h.text)());
    }
    for (file, text, found) in &files.done {
        line(&mut out, 1, file);
        for (n, l) in text.lines().enumerate() {
            if !found.includes.contains(&n) {
                out.push_str(l);
            }
            out.push('\n');
        }
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

/// The part of `line` outside comments.
fn code(line: &str, open: &mut bool) -> String {
    let mut out = String::new();
    let mut rest = line;
    loop {
        if *open {
            let Some(end) = rest.find("*/") else {
                return out;
            };
            rest = &rest[end + 2..];
            *open = false;
        }
        let line_at = rest.find("//");
        let block_at = rest.find("/*");
        match (line_at, block_at) {
            (Some(l), Some(b)) if l < b => return out + &rest[..l],
            (_, Some(b)) => {
                out.push_str(&rest[..b]);
                out.push(' ');
                rest = &rest[b + 2..];
                *open = true;
            }
            (Some(l), None) => return out + &rest[..l],
            (None, None) => return out + rest,
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
    let parts: Vec<&str> = target.split('/').collect();
    match parts[..] {
        ["tessera", _] => Err(Problem::Bare(target.into())),
        ["tessera", edition, name] if edition == EDITION.to_string() => {
            Import::parse(name).ok_or_else(|| Problem::Unknown(name.into()))
        }
        ["tessera", edition, _] => Err(Problem::Edition(edition.into())),
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
    let mut scan = Scan::default();
    let mut open = false;
    let mut leading = true;
    for (n, line) in text.lines().enumerate() {
        let code = code(line, &mut open);
        let code = code.trim();
        if code.is_empty() {
            continue;
        }
        let late = code.contains("__has_include") || code.starts_with("%:");
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
