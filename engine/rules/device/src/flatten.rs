//! A kernel and everything it includes, as one translation unit.

use crate::scan::{Problem, Scan, scan};
use crate::{Backend, Error, HEADERS, Import};
use std::collections::BTreeSet;

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
    let persistent = imports.iter().find(|i| i.persistent());
    if let Some(i) = persistent.filter(|_| backend == Backend::Metal) {
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
        out.push_str(h.text);
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
