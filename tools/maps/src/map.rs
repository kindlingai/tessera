//! The map format. Every string is a reference, never text.

use std::collections::BTreeMap;

use serde::Deserialize;
use toml::Spanned;

use crate::{Fault, LIMITS};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Sheet {
    task: BTreeMap<String, Task>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Task {
    pub steps: Vec<Spanned<Step>>,
    #[serde(default)]
    pub each: Vec<Spanned<Each>>,
}

/// The closed verb list. Each verb's relation lives in `verbs.rs`.
#[derive(Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Verb {
    Define,
    Register,
    Implement,
    List,
    Place,
    Test,
    Run,
}

#[derive(Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Test,
    Build,
    Bin,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    #[serde(rename = "do")]
    pub verb: Verb,
    pub with: Option<String>,
    pub like: Option<String>,
    pub at: Option<String>,
    pub via: Option<String>,
    pub pkg: Option<String>,
    pub mode: Option<Mode>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Each {
    pub of: String,
    pub has: String,
}

/// Field string kinds: symbol, file, glob or package.
#[derive(Clone, Copy, PartialEq)]
pub enum Class {
    Symbol,
    Path,
    Glob,
    Package,
}

impl Verb {
    /// Required fields, then optional ones.
    fn fields(self) -> (&'static [&'static str], &'static [&'static str]) {
        match self {
            Verb::Define | Verb::Register | Verb::Implement => (&["with", "like"], &[]),
            Verb::List | Verb::Place => (&["at", "like"], &[]),
            Verb::Test => (&["with", "like"], &["via"]),
            Verb::Run => (&["pkg", "mode"], &[]),
        }
    }

    pub fn name(self) -> String {
        format!("{self:?}").to_lowercase()
    }
}

impl Step {
    /// The string fields that are set, with their classes.
    pub fn refs(&self) -> Vec<(&'static str, &str, Class)> {
        let like = |s: &str| match s.contains("::") {
            true => Class::Symbol,
            false => Class::Path,
        };
        let at = match self.verb {
            Verb::Place => Class::Glob,
            _ => Class::Path,
        };
        let fields = [
            ("with", &self.with, Class::Symbol),
            ("at", &self.at, at),
            ("like", &self.like, Class::Path),
            ("via", &self.via, Class::Symbol),
            ("pkg", &self.pkg, Class::Package),
        ];
        let mut out = Vec::new();
        for (name, value, class) in fields {
            if let Some(v) = value {
                let class = if name == "like" { like(v) } else { class };
                out.push((name, v.as_str(), class));
            }
        }
        out
    }

    fn shape(&self) -> Result<(), String> {
        let (need, may) = self.verb.fields();
        let mut set: Vec<&str> = self.refs().iter().map(|r| r.0).collect();
        if self.mode.is_some() {
            set.push("mode");
        }
        for f in need {
            if !set.contains(f) {
                return Err(format!("needs `{f}`"));
            }
        }
        match set.iter().find(|f| !need.contains(f) && !may.contains(f)) {
            Some(f) => Err(format!("does not take `{f}`")),
            None => Ok(()),
        }
    }
}

/// One task, from one map file.
pub struct Map {
    pub path: String,
    pub id: String,
    pub line: usize,
    pub task: Task,
}

pub fn line_of(text: &str, offset: usize) -> usize {
    text[..offset.min(text.len())].matches('\n').count() + 1
}

fn segment_ok(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
}

/// Kebab-case, within the lint.toml limits.
fn id_fault(id: &str) -> Option<String> {
    let parts: Vec<&str> = id.split('-').collect();
    let (segments, chars) = (LIMITS.max_name_segments, LIMITS.max_map_id_chars);
    let word = |p: &&str| !p.is_empty() && p.chars().all(|c| c.is_ascii_lowercase());
    if !parts.iter().all(word) {
        return Some(format!("id `{id}` is not kebab-case words"));
    }
    if parts.len() > segments || id.len() > chars {
        return Some(format!(
            "id `{id}` is over the limit ({segments} segments, {chars} characters)"
        ));
    }
    None
}

fn ident_ok(s: &str) -> bool {
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Whether `value` has the form of `class`. Resolution comes later.
fn class_ok(value: &str, class: Class) -> bool {
    let file_part = |p: &str, glob: bool| {
        !p.is_empty()
            && p != ".."
            && p != "."
            && p.chars()
                .all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c) || (glob && c == '*'))
    };
    match class {
        Class::Symbol => {
            let body = value.strip_suffix('!').unwrap_or(value);
            let segs: Vec<&str> = body.split("::").collect();
            segs.len() > 1 && segs.iter().all(|s| ident_ok(s))
        }
        Class::Path | Class::Glob => value.split('/').all(|p| file_part(p, class == Class::Glob)),
        Class::Package => value.split('-').all(segment_ok),
    }
}

fn fault(path: &str, line: usize, message: String) -> Fault {
    Fault {
        path: path.into(),
        line,
        message,
    }
}

/// One file's tasks. Format faults go to `out`.
pub fn read(path: &str, text: &str, out: &mut Vec<Fault>) -> Vec<Map> {
    if let Some(at) = text.find('#') {
        let why = "a `#` comment. Maps hold references, not text".into();
        out.push(fault(path, line_of(text, at), why));
        return Vec::new();
    }
    let stem = path
        .rsplit('/')
        .next()
        .unwrap_or(path)
        .trim_end_matches(".toml");
    let sheet: Sheet = match toml::from_str(text) {
        Ok(s) => s,
        Err(e) => {
            let line = e.span().map_or(1, |s| line_of(text, s.start));
            out.push(fault(path, line, e.message().to_string()));
            return Vec::new();
        }
    };
    if sheet.task.len() != 1 || !sheet.task.contains_key(stem) {
        let why = format!("a map file holds one task, `[task.{stem}]`");
        out.push(fault(path, 1, why));
        return Vec::new();
    }
    let mut maps = Vec::new();
    for (id, task) in sheet.task {
        let line = text
            .find(&format!("[task.{id}]"))
            .map_or(1, |at| line_of(text, at));
        let map = Map {
            path: path.into(),
            id,
            line,
            task,
        };
        if map.form(text, out) {
            maps.push(map);
        }
    }
    maps
}

impl Map {
    pub fn fault(&self, line: usize, message: String) -> Fault {
        fault(&self.path, line, format!("task {}: {message}", self.id))
    }

    /// Checks the id, limits, field shapes and string classes.
    fn form(&self, text: &str, out: &mut Vec<Fault>) -> bool {
        let before = out.len();
        if let Some(why) = id_fault(&self.id) {
            out.push(self.fault(self.line, why));
        }
        let max = LIMITS.max_map_steps;
        let (steps, each) = (&self.task.steps, &self.task.each);
        if steps.is_empty() || steps.len() > max || each.len() > max {
            out.push(self.fault(
                self.line,
                format!("1 to {max} steps, at most {max} invariants"),
            ));
        }
        for (n, step) in steps.iter().enumerate() {
            let line = line_of(text, step.span().start);
            let s = step.get_ref();
            let head = format!("step {} ({})", n + 1, s.verb.name());
            if let Err(why) = s.shape() {
                out.push(self.fault(line, format!("{head} {why}")));
            }
            for (field, value, class) in s.refs() {
                if !class_ok(value, class) {
                    out.push(self.fault(line, format!("{head} `{field}` is not a reference")));
                }
            }
        }
        for e in each {
            let line = line_of(text, e.span().start);
            for value in [&e.get_ref().of, &e.get_ref().has] {
                if !class_ok(value, Class::Symbol) {
                    out.push(self.fault(line, format!("each `{value}` is not a symbol")));
                }
            }
        }
        out.len() == before
    }
}
