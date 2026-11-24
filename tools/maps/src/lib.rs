//! Task maps: ordered verb steps over checked code references.

mod calls;
mod code;
mod map;
mod query;
mod registry;
mod verbs;

#[cfg(test)]
mod verbs_test;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::LazyLock;

pub use code::listed;
use code::{Item, Repo};
use map::{Map, Step};
use registry::Inventory;
use verbs::Book;

/// The lint.toml limits maps read.
#[derive(serde::Deserialize)]
struct Limits {
    max_name_segments: usize,
    max_map_id_chars: usize,
    max_map_steps: usize,
}

static LIMITS: LazyLock<Limits> =
    LazyLock::new(|| toml::from_str(include_str!("../../../lint.toml")).expect("lint.toml"));

/// Map files live here, outside the protected paths.
const DIR: &str = "maps/";

pub struct Fault {
    pub path: String,
    pub line: usize,
    pub message: String,
}

/// The maps, the workspace they cite and its registries.
pub struct Atlas {
    repo: Repo,
    inv: Inventory,
    maps: Vec<Map>,
    texts: BTreeMap<String, String>,
    faults: Vec<Fault>,
}

/// A reference as written, and what it resolved to.
type Ref = (&'static str, String, Result<Vec<Item>, String>);

impl Atlas {
    pub fn open(root: &Path) -> Result<Atlas, String> {
        let repo = Repo::open(root)?;
        let (mut maps, mut texts, mut faults) = (Vec::new(), BTreeMap::new(), Vec::new());
        for path in repo.files.iter().filter(|f| f.starts_with(DIR)) {
            let line = 1;
            if !path.ends_with(".toml") || path[DIR.len()..].contains('/') {
                let message = format!("only {DIR}<id>.toml files belong here");
                faults.push(Fault {
                    path: path.clone(),
                    line,
                    message,
                });
                continue;
            }
            let text =
                std::fs::read_to_string(root.join(path)).map_err(|e| format!("{path}: {e}"))?;
            maps.extend(map::read(path, &text, &mut faults));
            texts.insert(path.clone(), text);
        }
        let inv = registry::derive(&repo);
        Ok(Atlas {
            repo,
            inv,
            maps,
            texts,
            faults,
        })
    }

    fn book(&self) -> Book<'_> {
        Book {
            repo: &self.repo,
            inv: &self.inv,
        }
    }

    fn line(&self, map: &Map, offset: usize) -> usize {
        map::line_of(&self.texts[&map.path], offset)
    }

    fn refs(&self, step: &Step) -> Vec<Ref> {
        let book = self.book();
        let mut out = Vec::new();
        for (field, value, class) in step.refs() {
            out.push((field, value.to_string(), book.resolve(value, class)));
        }
        out
    }

    /// Every fault: format, resolution, relation and invariant.
    pub fn check(mut self) -> Vec<Fault> {
        let mut out = std::mem::take(&mut self.faults);
        let book = self.book();
        let mut seen: BTreeMap<&str, &str> = BTreeMap::new();
        for map in &self.maps {
            if let Some(first) = seen.insert(&map.id, &map.path) {
                out.push(map.fault(map.line, format!("is also in {first}")));
            }
            for (n, step) in map.task.steps.iter().enumerate() {
                let line = self.line(map, step.span().start);
                let head = format!("step {} ({})", n + 1, step.get_ref().verb.name());
                let mut like = None;
                let mut resolved = true;
                for (field, _, got) in self.refs(step.get_ref()) {
                    match got {
                        Ok(items) if field == "like" => like = items.into_iter().next(),
                        Ok(_) => {}
                        Err(why) => {
                            resolved = false;
                            out.push(map.fault(line, format!("{head} {field}: {why}")));
                        }
                    }
                }
                if let (true, Err(why)) = (resolved, book.holds(step.get_ref(), like.as_ref())) {
                    out.push(map.fault(line, format!("{head} does not hold: {why}")));
                }
            }
            for e in &map.task.each {
                let line = self.line(map, e.span().start);
                if let Err(why) = book.each(e.get_ref()) {
                    out.push(map.fault(
                        line,
                        format!("each {} does not hold: {why}", e.get_ref().of),
                    ));
                }
            }
        }
        out
    }
}

/// Map faults in the repository at `root`.
pub fn check(root: &Path) -> Result<Vec<Fault>, String> {
    Ok(Atlas::open(root)?.check())
}
