//! Pairs a model's config keys with the values files offer.

use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::BTreeSet;
use std::fmt;

/// One source of config values, by name.
pub struct Offered {
    pub name: String,
    pub root: Value,
}

impl Offered {
    /// The JSON text `json`, named `name`.
    pub fn parse(name: &str, json: &str) -> Result<Self, String> {
        let root = serde_json::from_str(json).map_err(|e| format!("{name}: {e}"))?;
        Ok(Self {
            name: name.into(),
            root,
        })
    }
}

/// Why the keys and the offers do not pair off.
#[derive(Debug, Default, PartialEq)]
pub struct Unresolved {
    /// Keys no offer holds.
    pub missing: Vec<String>,
    /// Keys one offer holds under aliases that differ.
    pub ambiguous: Vec<(String, Vec<String>)>,
    /// Keys whose value the model cannot take, and why.
    pub misfit: Vec<(String, String)>,
}

impl fmt::Display for Unresolved {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for path in &self.missing {
            writeln!(f, "{path}: no file offers it")?;
        }
        for (path, held) in &self.ambiguous {
            writeln!(f, "{path}: differs across {}", held.join(", "))?;
        }
        for (path, why) in &self.misfit {
            writeln!(f, "{path}: {why}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Unresolved {}

/// The keys a model reads. Earlier offers win.
pub struct Keys<'a> {
    offers: &'a [Offered],
    /// Paths read, by offer.
    read: BTreeSet<(usize, String)>,
    unresolved: Unresolved,
}

impl<'a> Keys<'a> {
    pub fn new(offers: &'a [Offered]) -> Self {
        Self {
            offers,
            read: BTreeSet::new(),
            unresolved: Unresolved::default(),
        }
    }

    /// The first of `paths` an offer holds, which must exist.
    pub fn need<T: DeserializeOwned + Default>(&mut self, paths: &[&str]) -> T {
        match self.find(paths) {
            Some(Some(value)) => value,
            Some(None) => T::default(),
            None => {
                self.unresolved.missing.push(paths[0].into());
                T::default()
            }
        }
    }

    /// The first of `paths` an offer holds, else `default`.
    pub fn or<T: DeserializeOwned>(&mut self, paths: &[&str], default: T) -> T {
        self.find(paths).flatten().unwrap_or(default)
    }

    /// Checks a value the graph fixes, if offered.
    pub fn fixed(&mut self, path: &str, want: Value) {
        if let Some(Some(held)) = self.find::<Value>(&[path])
            && held != want
        {
            self.misfit(path, format!("the graph fixes {want}, not {held}"));
        }
    }

    /// Records a value the caller found the model cannot take.
    pub fn misfit(&mut self, path: &str, why: impl Into<String>) {
        self.unresolved.misfit.push((path.into(), why.into()));
    }

    /// Values beside read ones that nothing read, else the failures.
    pub fn finish(self) -> Result<Vec<String>, Unresolved> {
        let u = self.unresolved;
        if !(u.missing.is_empty() && u.ambiguous.is_empty() && u.misfit.is_empty()) {
            return Err(u);
        }
        let mut unread = Vec::new();
        for (i, offer) in self.offers.iter().enumerate() {
            let read: Vec<&str> = (self.read.iter())
                .filter(|(at, _)| *at == i)
                .map(|(_, path)| path.as_str())
                .collect();
            let parents: BTreeSet<&str> = read.iter().map(|p| parent(p)).collect();
            for at in parents {
                let Some(Value::Object(map)) = lookup(&offer.root, at) else {
                    continue;
                };
                for key in map.keys() {
                    let path = join(at, key);
                    let under = format!("{path}.");
                    if !read.iter().any(|r| *r == path || r.starts_with(&under)) {
                        unread.push(format!("{}: {path}", offer.name));
                    }
                }
            }
        }
        Ok(unread)
    }

    /// `None` when absent. `Some(None)` on a misfit.
    fn find<T: DeserializeOwned>(&mut self, paths: &[&str]) -> Option<Option<T>> {
        let offers = self.offers;
        for (i, offer) in offers.iter().enumerate() {
            let mut held = Vec::new();
            for path in paths {
                if let Some(value) = lookup(&offer.root, path) {
                    held.push((*path, value));
                }
            }
            let Some(&(first, value)) = held.first() else {
                continue;
            };
            for (path, _) in &held {
                self.read.insert((i, path.to_string()));
            }
            if held.iter().any(|(_, v)| *v != value) {
                let names = held.iter().map(|(p, _)| format!("{}: {p}", offer.name));
                self.unresolved
                    .ambiguous
                    .push((first.into(), names.collect()));
            }
            return Some(match T::deserialize(value) {
                Ok(value) => Some(value),
                Err(e) => {
                    self.misfit(first, format!("{}: {e}", offer.name));
                    None
                }
            });
        }
        None
    }
}

/// The value at dotted `path`, the root at `""`.
fn lookup<'v>(root: &'v Value, path: &str) -> Option<&'v Value> {
    if path.is_empty() {
        return Some(root);
    }
    let mut at = root;
    for key in path.split('.') {
        at = at.as_object()?.get(key)?;
    }
    Some(at)
}

fn parent(path: &str) -> &str {
    path.rsplit_once('.').map_or("", |(p, _)| p)
}

fn join(parent: &str, key: &str) -> String {
    match parent {
        "" => key.into(),
        _ => format!("{parent}.{key}"),
    }
}
