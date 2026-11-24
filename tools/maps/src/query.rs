//! Queries over the maps: one task, citing tasks, registries, digests.

use std::fmt::Write;

use crate::Atlas;
use crate::code::glob;
use crate::map::{Class, Map};
use crate::registry::Target;

/// FNV-1a, 64 bits. Stable across builds and platforms.
fn fnv(text: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn place(items: &Result<Vec<crate::Item>, String>) -> String {
    match items {
        Ok(items) => {
            let at: Vec<String> = items
                .iter()
                .map(|it| format!("{}:{}", it.file, it.line))
                .collect();
            at.join(" ")
        }
        Err(why) => format!("unresolved: {why}"),
    }
}

impl Atlas {
    fn task(&self, id: &str) -> Result<&Map, String> {
        self.maps
            .iter()
            .find(|m| m.id == id)
            .ok_or(format!("no task `{id}`"))
    }

    /// Each step of task `id`, its references at `file:line`.
    pub fn show(&self, id: &str) -> Result<String, String> {
        let map = self.task(id)?;
        let mut out = format!("{} {}:{}\n", map.id, map.path, map.line);
        for (n, step) in map.task.steps.iter().enumerate() {
            let s = step.get_ref();
            let mode = s.mode.map(|m| format!(" mode={m:?}")).unwrap_or_default();
            let _ = writeln!(out, "{:>2} {}{}", n + 1, s.verb.name(), mode.to_lowercase());
            for (field, value, got) in self.refs(s) {
                let _ = writeln!(out, "     {field:<4} {value}  {}", place(&got));
            }
        }
        for e in &map.task.each {
            let e = e.get_ref();
            let _ = writeln!(out, "  each {} has {}", e.of, e.has);
        }
        Ok(out)
    }

    /// Tasks and steps whose references land in `path`.
    pub fn cites(&self, path: &str) -> String {
        let mut out = String::new();
        for map in &self.maps {
            for (n, step) in map.task.steps.iter().enumerate() {
                for (field, value, got) in self.refs(step.get_ref()) {
                    let lands = got
                        .as_ref()
                        .is_ok_and(|items| items.iter().any(|it| it.file == path));
                    if lands || glob(&value, path) {
                        let line = self.line(map, step.span().start);
                        let verb = step.get_ref().verb.name();
                        let _ = writeln!(
                            out,
                            "{}:{line}: {} step {} {verb} {field}",
                            map.path,
                            map.id,
                            n + 1
                        );
                    }
                }
            }
            for e in &map.task.each {
                let e = e.get_ref();
                for value in [&e.of, &e.has] {
                    let got = self
                        .book()
                        .resolve(value, Class::Symbol)
                        .unwrap_or_default();
                    if got.iter().any(|it| it.file == path) {
                        let line = self.line(map, 0).max(map.line);
                        let _ = writeln!(out, "{}:{line}: {} each {value}", map.path, map.id);
                    }
                }
            }
        }
        out
    }

    /// Each registry, its registering macros and its registrants.
    pub fn inventory(&self) -> String {
        let mut out = String::new();
        for (r, reg) in self.inv.registries.iter().enumerate() {
            let _ = writeln!(out, "{} {}:{}", reg.path.join("::"), reg.file, reg.line);
            for m in &self.inv.makers {
                let open = m.target == Target::Open;
                if m.target == Target::Fixed(r)
                    || (open
                        && reg
                            .members
                            .iter()
                            .any(|x| self.inv.makers[x.maker].name == m.name))
                {
                    let _ = writeln!(out, "  macro {}! {}:{}", m.name, m.file, m.line);
                }
            }
            for x in &reg.members {
                let _ = writeln!(out, "  {} {}:{}", x.key, x.file, x.line);
            }
        }
        out
    }

    /// One digest per task over its steps and resolved files.
    pub fn digest(&self) -> String {
        let mut out = String::new();
        for map in &self.maps {
            let mut text = map.id.clone();
            for step in &map.task.steps {
                let s = step.get_ref();
                let _ = write!(text, "|{}:{:?}", s.verb.name(), s.mode);
                for (field, value, got) in self.refs(s) {
                    let mut files: Vec<String> = got
                        .unwrap_or_default()
                        .into_iter()
                        .map(|it| it.file)
                        .collect();
                    files.sort();
                    let _ = write!(text, ",{field}={value}>{}", files.join("+"));
                }
            }
            for e in &map.task.each {
                let _ = write!(text, "|each:{}:{}", e.get_ref().of, e.get_ref().has);
            }
            let _ = writeln!(out, "{} {:016x}", map.id, fnv(&text));
        }
        out
    }
}
