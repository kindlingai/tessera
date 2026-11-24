//! Registries from `gather!` and `scatter!`, their macros and members.

use crate::calls::{all_idents, static_name, top_idents, top_lits};
use crate::code::{Item, Repo};

pub struct Registry {
    pub path: Vec<String>,
    pub file: String,
    pub line: usize,
    pub members: Vec<Member>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Target {
    Fixed(usize),
    /// The registry comes from each call's arguments.
    Open,
}

pub struct Maker {
    pub name: String,
    pub krate: String,
    pub file: String,
    pub line: usize,
    pub export: bool,
    pub target: Target,
    pub wraps: Option<usize>,
    /// The wrapped call repeats, one registrant per string literal.
    pub fan: bool,
}

pub struct Member {
    pub key: String,
    pub file: String,
    pub line: usize,
    pub maker: usize,
}

#[derive(Default)]
pub struct Inventory {
    pub registries: Vec<Registry>,
    pub makers: Vec<Maker>,
}

fn screaming(s: &str) -> bool {
    s.len() > 1
        && s.chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

fn ident_of(repo: &Repo, file: &str) -> String {
    repo.owner(file)
        .map(|k| k.ident.clone())
        .unwrap_or_default()
}

/// A macro defined in `krate` is callable from `file`.
fn reaches(repo: &Repo, file: &str, krate: &str, export: bool) -> bool {
    let same = ident_of(repo, file) == krate;
    same || (export && repo.text(file).is_some_and(|t| t.contains(krate)))
}

pub fn derive(repo: &Repo) -> Inventory {
    let mut inv = Inventory::default();
    inv.gathers(repo);
    let defs: Vec<&str> = repo.holding("macro_rules").collect();
    while inv.makers_in(repo, &defs) {}
    for at in 0..inv.makers.len() {
        inv.members_of(repo, at);
    }
    inv
}

impl Inventory {
    fn add(&mut self, repo: &Repo, file: &str, line: usize, name: String) {
        let Some(mut path) = repo.module_of(file) else {
            return;
        };
        path.push(name);
        if !self.registries.iter().any(|r| r.path == path) {
            let (file, members) = (file.to_string(), Vec::new());
            self.registries.push(Registry {
                path,
                file,
                line,
                members,
            });
        }
    }

    /// Registries from `gather!`, direct or inside a declaring macro.
    fn gathers(&mut self, repo: &Repo) {
        let mut declarers = Vec::new();
        for file in repo.holding("gather!") {
            let Some(scan) = repo.scan(file) else {
                continue;
            };
            for call in scan.calls.iter().filter(|c| c.name == "gather") {
                match (static_name(&call.args), call.def) {
                    (Some(n), _) if !n.starts_with('$') => self.add(repo, file, call.line, n),
                    (Some(_), Some(d)) => {
                        let def = &scan.defs[d];
                        declarers.push((def.name.clone(), ident_of(repo, file), def.export));
                    }
                    _ => {}
                }
            }
        }
        for (name, krate, export) in declarers {
            let needle = format!("{name}!");
            for file in repo.holding(&needle) {
                let Some(scan) = repo
                    .scan(file)
                    .filter(|_| reaches(repo, file, &krate, export))
                else {
                    continue;
                };
                for call in &scan.calls {
                    if call.name != name || call.def.is_some() {
                        continue;
                    }
                    if let Some(id) = top_idents(&call.args).into_iter().find(|i| screaming(i)) {
                        self.add(repo, file, call.line, id);
                    }
                }
            }
        }
    }

    /// The maker a call to `name` in `file` reaches.
    pub fn maker_for(&self, repo: &Repo, file: &str, name: &str) -> Option<usize> {
        let krate = ident_of(repo, file);
        let named = |m: &&Maker| m.name == name;
        let all: Vec<usize> = (0..self.makers.len())
            .filter(|&i| named(&&self.makers[i]))
            .collect();
        let local = all.iter().find(|&&i| self.makers[i].krate == krate);
        local
            .or_else(|| {
                all.iter().find(|&&i| {
                    let m = &self.makers[i];
                    reaches(repo, file, &m.krate, m.export)
                })
            })
            .copied()
    }

    /// The registry an identifier in `args` names, own crate first.
    fn named_in(&self, args: &proc_macro2::TokenStream, krate: &str) -> Option<usize> {
        let mut ids = Vec::new();
        all_idents(args, &mut ids);
        let hits: Vec<usize> = (0..self.registries.len())
            .filter(|&r| {
                ids.iter()
                    .any(|i| self.registries[r].path.last() == Some(i))
            })
            .collect();
        let local = hits.iter().find(|&&r| self.registries[r].path[0] == krate);
        local.or(hits.first()).copied()
    }

    /// Adds scattering or wrapping macros. True if any.
    fn makers_in(&mut self, repo: &Repo, defs: &[&str]) -> bool {
        let mut grew = false;
        for &file in defs {
            let Some(scan) = repo.scan(file) else {
                continue;
            };
            let krate = ident_of(repo, file);
            for call in &scan.calls {
                let Some(d) = call.def else {
                    continue;
                };
                let def = &scan.defs[d];
                let known = |m: &Maker| m.file == file && m.line == def.line;
                if self.makers.iter().any(known) {
                    continue;
                }
                let inner = match call.name.as_str() {
                    "scatter" => None,
                    other => match self.maker_for(repo, file, other) {
                        Some(i) => Some(i),
                        None => continue,
                    },
                };
                let named = self.named_in(&call.args, &krate).map(Target::Fixed);
                let target = match inner.map(|i| self.makers[i].target) {
                    Some(Target::Fixed(r)) => Target::Fixed(r),
                    _ => named.unwrap_or(Target::Open),
                };
                self.makers.push(Maker {
                    name: def.name.clone(),
                    krate: krate.clone(),
                    file: file.to_string(),
                    line: def.line,
                    export: def.export,
                    target,
                    wraps: inner,
                    fan: call.fan,
                });
                grew = true;
            }
        }
        grew
    }

    fn members_of(&mut self, repo: &Repo, at: usize) {
        let name = self.makers[at].name.clone();
        for file in repo.holding(&format!("{name}!")) {
            let Some(scan) = repo.scan(file) else {
                continue;
            };
            for call in scan
                .calls
                .iter()
                .filter(|c| c.name == name && c.def.is_none())
            {
                if self.maker_for(repo, file, &name) != Some(at) {
                    continue;
                }
                let krate = ident_of(repo, file);
                let reg = match self.makers[at].target {
                    Target::Fixed(r) => Some(r),
                    Target::Open => self.named_in(&call.args, &krate),
                };
                let Some(reg) = reg else {
                    continue;
                };
                let m = &self.makers[at];
                let lits: Vec<String> = top_lits(&call.args)
                    .into_iter()
                    .filter(|l| !l.is_empty())
                    .collect();
                let fan = m.wraps.is_some() && m.fan;
                let mut keys: Vec<String> = lits
                    .into_iter()
                    .take(if fan { usize::MAX } else { 1 })
                    .collect();
                if keys.is_empty() {
                    keys.extend(top_idents(&call.args).into_iter().take(1));
                }
                for key in keys {
                    let (file, line) = (file.to_string(), call.line);
                    self.registries[reg].members.push(Member {
                        key,
                        file,
                        line,
                        maker: at,
                    });
                }
            }
        }
    }

    /// The registry a symbol path names.
    pub fn find(&self, items: &[Item], path: &str) -> Option<usize> {
        let at = |r: &Registry| {
            items
                .iter()
                .any(|it| it.file == r.file && r.path.last() == Some(&it.name))
        };
        let by_path = |r: &Registry| r.path.join("::") == path;
        self.registries
            .iter()
            .position(at)
            .or_else(|| self.registries.iter().position(by_path))
    }

    /// The maker a resolved macro item is.
    pub fn maker_at(&self, it: &Item) -> Option<usize> {
        self.makers
            .iter()
            .position(|m| m.file == it.file && m.line == it.line && m.name == it.name)
    }

    /// Whether maker `a` is `b` or wraps it.
    pub fn leads_to(&self, mut a: usize, b: usize) -> bool {
        loop {
            if a == b {
                return true;
            }
            match self.makers[a].wraps {
                Some(w) => a = w,
                None => return false,
            }
        }
    }
}
