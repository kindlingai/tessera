//! Crates depend on their own layer or lower.

use super::workspace::Pkg;
use crate::Violation;

/// An upward dependency that stays, and why.
#[derive(serde::Deserialize)]
pub struct Allow {
    pub from: String,
    pub to: String,
    pub reason: String,
}

/// Index of the layer whose directory prefix holds `dir`.
fn layer_of(layers: &[Vec<String>], dir: &str) -> Option<usize> {
    layers
        .iter()
        .position(|prefixes| prefixes.iter().any(|p| dir.starts_with(p.as_str())))
}

pub fn check(pkgs: &[Pkg], layers: &[Vec<String>], allow: &[Allow], out: &mut Vec<Violation>) {
    let mut used = vec![false; allow.len()];
    for p in pkgs {
        let Some(mine) = layer_of(layers, &p.dir) else {
            let message = format!("{} is in no layer: add its directory to `layers`", p.dir);
            out.push(Violation {
                path: p.manifest_path(),
                line: 1,
                rule: "layers",
                message,
            });
            continue;
        };
        for d in &p.deps {
            let Some(dep) = pkgs.iter().find(|q| q.name == d.name) else {
                continue;
            };
            if layer_of(layers, &dep.dir).is_none_or(|theirs| theirs <= mine) {
                continue;
            }
            if let Some(i) = allow
                .iter()
                .position(|a| a.from == p.name && a.to == d.name)
            {
                used[i] = true;
                continue;
            }
            let message = format!(
                "`{}` ({}) {} `{}` ({}), which is in a higher layer",
                p.name,
                p.dir,
                d.kind.verb(),
                dep.name,
                dep.dir
            );
            out.push(Violation {
                path: p.manifest_path(),
                line: p.dep_line(&d.name, d.kind),
                rule: "layers",
                message,
            });
        }
    }
    for (a, used) in allow.iter().zip(used) {
        if !used {
            out.push(Violation {
                path: "lint.toml".into(),
                line: 1,
                rule: "layers",
                message: format!("`{}` to `{}` is no longer upward: remove it", a.from, a.to),
            });
        }
    }
}
