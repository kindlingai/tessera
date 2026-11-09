//! The function name rule, judged across a whole crate.

use std::collections::BTreeMap;

use crate::files::owner;
use crate::{LIMITS, Violation};

/// A Rust function or kernel entry point, with its location.
pub struct Name {
    pub path: String,
    pub line: usize,
    pub name: String,
    pub what: &'static str,
}

/// Snake-case pieces, ignoring leading or doubled underscores.
pub fn pieces(name: &str) -> Vec<&str> {
    let name = name.strip_prefix("r#").unwrap_or(name);
    name.split('_').filter(|s| !s.is_empty()).collect()
}

/// Another crate name shares its first or last piece.
fn in_family(name: &str, crate_names: &[&str]) -> bool {
    let p = pieces(name);
    let (Some(first), Some(last)) = (p.first(), p.last()) else {
        return false;
    };
    crate_names.iter().filter(|&&o| o != name).any(|o| {
        let q = pieces(o);
        q.first() == Some(first) || q.last() == Some(last)
    })
}

/// Why `name` breaks the rule, if it does.
pub fn fault(name: &str, crate_names: &[&str]) -> Option<String> {
    let n = pieces(name).len();
    let (base, family) = (LIMITS.max_name_segments, LIMITS.max_family_segments);
    if n <= base {
        return None;
    }
    if n <= family && in_family(name, crate_names) {
        return None;
    }
    Some(if n > family {
        format!("{n} segments (max {family})")
    } else {
        format!("{n} segments: no other function in the crate shares its first or last")
    })
}

pub fn check(dirs: &[String], names: &[Name], out: &mut Vec<Violation>) {
    let mut by_crate: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for n in names {
        let dir = owner(dirs, &n.path).unwrap_or("");
        by_crate.entry(dir).or_default().push(&n.name);
    }
    for n in names {
        let crate_names = &by_crate[owner(dirs, &n.path).unwrap_or("")];
        if let Some(why) = fault(&n.name, crate_names) {
            out.push(Violation {
                path: n.path.clone(),
                line: n.line,
                rule: "name-segments",
                message: format!("{} `{}` has {why}", n.what, n.name),
            });
        }
    }
}
