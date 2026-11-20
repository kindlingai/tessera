//! `Cargo.toml` for a published crate, from its members' manifests.

use std::collections::{BTreeMap, BTreeSet};

use toml::{Table, Value};

use crate::Result;
use crate::dist::Release;
use crate::meta::{Dep, Meta};
use crate::plan::{Kind, Out};

/// The published crate that holds each workspace package.
pub(crate) type Owners<'a> = BTreeMap<usize, &'a str>;

/// One dependency, merged over its naming members.
struct Want {
    req: String,
    path: Option<String>,
    package: Option<String>,
    features: BTreeSet<String>,
    default: bool,
    optional: bool,
}

pub(crate) fn manifest(meta: &Meta, out: &Out, owners: &Owners, rel: &Release) -> Result<Table> {
    let version = rel.version.as_str();
    let first = &meta.packages[out.members[0].pkg];
    let mut package = Table::new();
    package.insert("name".into(), out.name.clone().into());
    package.insert("version".into(), version.into());
    package.insert("edition".into(), first.edition.clone().into());
    package.insert(
        "license".into(),
        first.license.clone().unwrap_or_default().into(),
    );
    package.insert("description".into(), out.description.clone().into());
    if let Some(repo) = &rel.repository {
        package.insert("repository".into(), repo.clone().into());
    }
    let mut doc = Table::new();
    doc.insert("package".into(), package.into());
    let mut target = Table::new();
    if out.kind == Kind::Bin {
        let bins = [crate::plan::CLI_ALIAS, out.name.as_str()].map(|name| {
            let mut target = Table::new();
            target.insert("name".into(), name.into());
            target.insert("path".into(), "src/main.rs".into());
            Value::from(target)
        });
        doc.insert("bin".into(), Value::Array(bins.into()));
    } else {
        target.insert("path".into(), "src/lib.rs".into());
        target.insert("proc-macro".into(), (out.kind == Kind::Macro).into());
        doc.insert("lib".into(), target.into());
    }
    let all = dependencies(meta, out, owners, version)?;
    let optional: BTreeSet<String> = all
        .values()
        .flat_map(|deps| {
            deps.iter()
                .filter(|(_, w)| w.optional)
                .map(|(k, _)| k.clone())
        })
        .collect();
    for (scope, deps) in all {
        let table: Table = deps.into_iter().map(|(k, w)| (k, entry(w))).collect();
        match scope {
            None => doc.insert("dependencies".into(), table.into()),
            Some(cfg) => {
                let targets = doc.entry("target").or_insert_with(|| Table::new().into());
                let mut inner = Table::new();
                inner.insert("dependencies".into(), table.into());
                targets
                    .as_table_mut()
                    .expect("a table")
                    .insert(cfg, inner.into())
            }
        };
    }
    // `dep:` names only dependencies that stay optional.
    let kept = |e: &String| e.strip_prefix("dep:").is_none_or(|d| optional.contains(d));
    let features: Table = features(meta, out, owners)
        .into_iter()
        .map(|(f, e)| {
            (
                f,
                Value::Array(e.into_iter().filter(kept).map(Value::from).collect()),
            )
        })
        .collect();
    doc.insert("features".into(), features.into());
    Ok(doc)
}

type Scoped = BTreeMap<Option<String>, BTreeMap<String, Want>>;

fn dependencies(meta: &Meta, out: &Out, owners: &Owners, version: &str) -> Result<Scoped> {
    let mut all = Scoped::new();
    for m in &out.members {
        let pkg = &meta.packages[m.pkg];
        for d in pkg.normal() {
            let (key, want) = match meta.packages.iter().position(|q| q.name == d.name) {
                Some(q) if owners.get(&q) == Some(&out.name.as_str()) => continue,
                Some(q) => {
                    let other = owners.get(&q).ok_or(format!(
                        "{} needs {}, which is in no crate",
                        pkg.name, d.name
                    ))?;
                    (other.to_string(), linked(d, other, version))
                }
                None => (d.key().to_string(), external(d)),
            };
            let scope = all.entry(d.target.clone()).or_default();
            match scope.get_mut(&key) {
                Some(have) => join(have, want),
                None => drop(scope.insert(key, want)),
            }
        }
    }
    Ok(all)
}

fn linked(d: &Dep, other: &str, version: &str) -> Want {
    let mut w = external(d);
    w.req = version.into();
    w.path = Some(format!("../{other}"));
    w.package = None;
    w
}

fn external(d: &Dep) -> Want {
    Want {
        req: d.req.clone(),
        path: None,
        package: d.rename.as_ref().map(|_| d.name.clone()),
        features: d.features.iter().cloned().collect(),
        default: d.uses_default_features,
        optional: d.optional,
    }
}

fn join(have: &mut Want, want: Want) {
    if floor(&want.req) > floor(&have.req) {
        have.req = want.req;
    }
    have.features.extend(want.features);
    have.default |= want.default;
    have.optional &= want.optional;
}

/// The lowest version a requirement takes, as numbers.
fn floor(req: &str) -> Vec<u64> {
    let digits = req.trim_start_matches(|c: char| !c.is_ascii_digit());
    digits.split('.').map(|n| n.parse().unwrap_or(0)).collect()
}

fn entry(w: Want) -> Value {
    let mut t = Table::new();
    t.insert("version".into(), w.req.into());
    if let Some(path) = w.path {
        t.insert("path".into(), path.into());
    }
    if let Some(package) = w.package {
        t.insert("package".into(), package.into());
    }
    if !w.features.is_empty() {
        t.insert(
            "features".into(),
            Value::Array(w.features.into_iter().map(Value::from).collect()),
        );
    }
    if !w.default {
        t.insert("default-features".into(), false.into());
    }
    if w.optional {
        t.insert("optional".into(), true.into());
    }
    t.into()
}

/// Member features by name, with merged links made local.
fn features(meta: &Meta, out: &Out, owners: &Owners) -> BTreeMap<String, BTreeSet<String>> {
    let mut all: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for m in &out.members {
        let pkg = &meta.packages[m.pkg];
        for (name, entries) in &pkg.features {
            let set = all.entry(name.clone()).or_default();
            for e in entries {
                let local = feature(meta, m.pkg, e, out, owners);
                set.extend(local.filter(|l| l != name));
            }
        }
    }
    all
}

fn feature(meta: &Meta, p: usize, e: &str, out: &Out, owners: &Owners) -> Option<String> {
    let Some((dep, feat)) = e.split_once('/') else {
        return Some(e.into());
    };
    let key = dep.trim_end_matches('?');
    let pkg = &meta.packages[p];
    let d = pkg.dependencies.iter().find(|d| d.key() == key)?;
    let Some(q) = meta.packages.iter().position(|q| q.name == d.name) else {
        return Some(e.into());
    };
    match owners.get(&q) {
        Some(o) if *o == out.name => Some(feat.into()),
        Some(o) => Some(format!("{o}{}/{feat}", &dep[key.len()..])),
        None => None,
    }
}
