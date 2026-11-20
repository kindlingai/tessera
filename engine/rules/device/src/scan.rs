//! The leading include block of one kernel file.

use crate::{EDITION, Import};
use std::fmt;

/// What is wrong, without the place.
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
