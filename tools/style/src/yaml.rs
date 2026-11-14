//! The YAML in rule files: block maps, lists and scalars.

use std::collections::BTreeMap;

pub enum Yaml {
    Str(String),
    List(Vec<String>),
    Map(BTreeMap<String, Yaml>),
}

impl Yaml {
    pub fn get(&self, key: &str) -> Option<&Yaml> {
        match self {
            Yaml::Map(m) => m.get(key),
            _ => None,
        }
    }

    pub fn str(&self) -> Option<&str> {
        match self {
            Yaml::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn list(&self) -> &[String] {
        match self {
            Yaml::List(l) => l,
            _ => &[],
        }
    }
}

/// A plain or single-quoted scalar. `''` is a quote inside quotes.
fn scalar(s: &str) -> String {
    let s = s.trim();
    match s.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')) {
        Some(inner) => inner.replace("''", "'"),
        None => s.to_string(),
    }
}

fn indent(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// The map or list at `depth`, from `lines[*at]`.
fn block(lines: &[&str], at: &mut usize, depth: usize) -> Yaml {
    if lines
        .get(*at)
        .is_some_and(|l| l.trim_start().starts_with("- "))
    {
        let mut items = Vec::new();
        while let Some(l) = lines.get(*at).filter(|l| indent(l) == depth) {
            let Some(item) = l.trim_start().strip_prefix("- ") else {
                break;
            };
            items.push(scalar(item));
            *at += 1;
        }
        return Yaml::List(items);
    }
    let mut map = BTreeMap::new();
    while let Some(l) = lines.get(*at).filter(|l| indent(l) == depth) {
        let (key, rest) = split_key(l.trim_start());
        *at += 1;
        let value = match rest {
            Some(v) => Yaml::Str(scalar(v)),
            None => match lines.get(*at) {
                Some(next) if indent(next) > depth => block(lines, at, indent(next)),
                _ => Yaml::Str(String::new()),
            },
        };
        map.insert(key, value);
    }
    Yaml::Map(map)
}

/// `key: value` or `key:`. A key may be single-quoted.
fn split_key(line: &str) -> (String, Option<&str>) {
    let end = match line.strip_prefix('\'') {
        Some(rest) => rest.find("':").map_or(line.len(), |i| i + 2),
        None => line.find(':').unwrap_or(line.len()),
    };
    let key = scalar(&line[..end]);
    let rest = line.get(end + 1..).map(str::trim).filter(|r| !r.is_empty());
    (key, rest)
}

pub fn parse(text: &str) -> Yaml {
    let lines: Vec<&str> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .collect();
    block(&lines, &mut 0, 0)
}
