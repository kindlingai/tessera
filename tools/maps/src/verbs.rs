//! Verb relations: the test behind each verb.

use std::collections::BTreeSet;

use crate::calls::{top_idents, top_lits};
use crate::code::{Item, Kind, Repo, glob};
use crate::map::{Class, Each, Mode, Step, Verb};
use crate::registry::Inventory;

pub(crate) struct Book<'a> {
    pub repo: &'a Repo,
    pub inv: &'a Inventory,
}

/// Lines of one file a claim is read from.
struct Scope {
    file: String,
    start: usize,
    end: usize,
}

impl Scope {
    fn of(it: &Item) -> Scope {
        let whole = matches!(it.kind, Kind::File | Kind::Mod);
        let (start, end) = if whole {
            (0, usize::MAX)
        } else {
            (it.start, it.end)
        };
        let file = it.file.clone();
        Scope { file, start, end }
    }

    fn holds(&self, line: usize) -> bool {
        (self.start..=self.end).contains(&line)
    }
}

fn file_item(file: &str) -> Item {
    let (name, kind, file) = (file.to_string(), Kind::File, file.to_string());
    Item {
        name,
        start: 1,
        kind,
        file,
        line: 1,
        end: 1,
    }
}

/// `a/b/../c` as `a/c`.
fn normal(path: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            p => out.push(p),
        }
    }
    out.join("/")
}

fn last_of(path: &str) -> &str {
    path.trim_end_matches('!')
        .rsplit("::")
        .next()
        .unwrap_or(path)
}

impl Book<'_> {
    /// The items a reference names, or what is missing.
    pub fn resolve(&self, value: &str, class: Class) -> Result<Vec<Item>, String> {
        let files = &self.repo.files;
        let found: Vec<Item> = match class {
            Class::Symbol => {
                let mut items = self.repo.resolve(value);
                if items.is_empty()
                    && let Some(r) = self.inv.find(&[], value)
                {
                    let r = &self.inv.registries[r];
                    let mut it = file_item(&r.file);
                    (it.kind, it.start, it.line, it.end) = (Kind::Item, r.line, r.line, r.line);
                    items.push(it);
                }
                items
            }
            Class::Path => files
                .iter()
                .filter(|f| *f == value)
                .map(|f| file_item(f))
                .collect(),
            Class::Glob => files
                .iter()
                .filter(|f| glob(value, f))
                .map(|f| file_item(f))
                .collect(),
            Class::Package => self
                .repo
                .package(value)
                .map(|k| file_item(&format!("{}Cargo.toml", k.dir)))
                .into_iter()
                .collect(),
        };
        match (found.is_empty(), class) {
            (false, _) => Ok(found),
            (true, Class::Symbol) => Err(format!("`{value}` names no item in the workspace")),
            (true, Class::Package) => Err(format!("`{value}` is not a workspace package")),
            (true, _) => Err(format!("no file at `{value}`")),
        }
    }

    /// Whether lines in `scope` name `path`, qualified or imported.
    fn mentions(&self, scope: &Scope, path: &str) -> bool {
        let segs: Vec<&str> = path.trim_end_matches('!').split("::").collect();
        let (krate, name) = (segs[0], last_of(path));
        let Some(scan) = self.repo.scan(&scope.file) else {
            return false;
        };
        let inside = |l: usize| scope.holds(l);
        let called = scan.calls.iter().any(|c| inside(c.line) && c.name == name);
        if path.ends_with('!') && !called {
            return false;
        }
        let bare = |n: &str| scan.idents.iter().any(|(l, s)| inside(*l) && s == n);
        let here = self
            .repo
            .resolve(path)
            .iter()
            .any(|it| it.file == scope.file);
        if here && bare(name) {
            return true;
        }
        let same = self
            .repo
            .owner(&scope.file)
            .is_some_and(|k| k.ident == krate);
        let from =
            |first: &str| first == krate || (same && ["crate", "self", "super"].contains(&first));
        let module = (segs.len() > 2).then(|| segs[segs.len() - 2]);
        for (l, chain) in &scan.chains {
            let n = chain.len();
            if !inside(*l) || chain[n - 1] != name {
                continue;
            }
            if from(&chain[0]) || module == Some(chain[n - 2].as_str()) {
                return true;
            }
        }
        for (local, raw) in self.repo.uses_in(&scope.file) {
            if !raw.first().is_some_and(|f| from(f)) {
                continue;
            }
            let leaf = raw.last().is_some_and(|l| l == name);
            if (local == "*" && bare(name)) || (leaf && bare(&local)) {
                return true;
            }
        }
        false
    }

    /// Tests the step's relation against any `like` item.
    pub fn holds(&self, step: &Step, likes: &[Item]) -> Result<(), String> {
        if likes.is_empty() {
            return self.holds_at(step, None);
        }
        let mut why = Ok(());
        for like in likes {
            why = self.holds_at(step, Some(Scope::of(like)));
            if why.is_ok() {
                break;
            }
        }
        why
    }

    fn holds_at(&self, step: &Step, scope: Option<Scope>) -> Result<(), String> {
        let field = |f: &Option<String>| f.clone().unwrap_or_default();
        let (with, at) = (field(&step.with), field(&step.at));
        let shown = field(&step.like);
        let fail = |what: String| Err(format!("`{shown}` {what}"));
        match (step.verb, scope) {
            (Verb::Define, Some(s)) if !self.mentions(&s, &with) => {
                fail(format!("does not use `{with}`"))
            }
            (Verb::Register, Some(s)) => self.registers(&s, &with).or_else(fail),
            (Verb::Implement, Some(s)) if !self.implements(&s, &with) => {
                fail(format!("does not implement `{with}`"))
            }
            (Verb::List, Some(s)) if !self.lists(&at, &s.file) => Err(format!(
                "`{at}` does not declare the module in `{}`",
                s.file
            )),
            (Verb::Place, Some(s)) if !self.places(&s, &at) => {
                fail(format!("names no file under `{at}`"))
            }
            (Verb::Test, Some(s)) => self.tests(&s, &with, step.via.as_deref()).or_else(fail),
            (Verb::Run, _) => self.runs(&field(&step.pkg), step.mode),
            _ => Ok(()),
        }
    }

    fn registers(&self, s: &Scope, with: &str) -> Result<(), String> {
        let items = self.repo.resolve(with);
        let Some(maker) = items.iter().find_map(|it| self.inv.maker_at(it)) else {
            return Err(format!("is cited, but `{with}` is not a registering macro"));
        };
        let scan = self.repo.scan(&s.file).ok_or("is not Rust")?;
        for call in &scan.calls {
            if call.def.is_some() || !s.holds(call.line) {
                continue;
            }
            let reached = self.inv.maker_for(self.repo, &s.file, &call.name);
            if reached.is_some_and(|m| self.inv.leads_to(m, maker)) {
                return Ok(());
            }
        }
        Err(format!("calls neither `{with}` nor a wrapper of `{with}`"))
    }

    fn implements(&self, s: &Scope, with: &str) -> bool {
        let Some(scan) = self.repo.scan(&s.file) else {
            return false;
        };
        let name = last_of(with);
        let resolved = self.repo.resolve(with);
        let derive = resolved.iter().any(|it| it.kind == Kind::Derive);
        let impls = scan.impls.iter().any(|(l, t)| s.holds(*l) && t == name);
        let derives = derive && scan.derives.iter().any(|(l, t)| s.holds(*l) && t == name);
        (impls || derives) && self.mentions(s, with)
    }

    /// A `mod` in `at`, or a local `mod $` macro call.
    fn lists(&self, at: &str, file: &str) -> bool {
        let stem = file.trim_end_matches("/mod.rs").trim_end_matches(".rs");
        let stem = stem.rsplit('/').next().unwrap_or(stem);
        if self.repo.kid_file(at, stem, &[]).as_deref() != Some(file) {
            return false;
        }
        if let Some(tree) = self.repo.tree(at) {
            for i in &tree.items {
                if let syn::Item::Mod(m) = i
                    && m.content.is_none()
                    && m.ident == stem
                {
                    return true;
                }
            }
        }
        let (Some(scan), Some(text)) = (self.repo.scan(at), self.repo.text(at)) else {
            return false;
        };
        if !text.contains("mod $") {
            return false;
        }
        for call in &scan.calls {
            let local = scan.defs.iter().any(|d| d.name == call.name);
            if local && call.def.is_none() && top_idents(&call.args).iter().any(|i| i == stem) {
                return true;
            }
        }
        false
    }

    /// A string literal in scope names a file `at` matches.
    fn places(&self, s: &Scope, at: &str) -> bool {
        let Some(scan) = self.repo.scan(&s.file) else {
            return false;
        };
        let dir = s.file.rsplit_once('/').map_or("", |d| d.0);
        let krate = self.repo.owner(&s.file).map_or("", |k| k.dir.as_str());
        for (l, lit) in &scan.lits {
            if !s.holds(*l) {
                continue;
            }
            for t in [format!("{dir}/{lit}"), format!("{krate}{lit}"), lit.clone()] {
                let t = normal(&t);
                if glob(at, &t) && self.repo.files.contains(&t) {
                    return true;
                }
            }
        }
        false
    }

    fn tests(&self, s: &Scope, with: &str, via: Option<&str>) -> Result<(), String> {
        let stem = s.file.trim_end_matches(".rs");
        if !s.file.contains("/tests/") && !stem.ends_with("_test") {
            return Err("is not a test file".into());
        }
        if !self.mentions(s, with) {
            return Err(format!("does not use `{with}`"));
        }
        let scan = self.repo.scan(&s.file).ok_or("is not Rust")?;
        match via {
            Some(v) if !v.ends_with('!') => Err(format!("cites `{v}`, which is not a macro")),
            Some(v) if !self.mentions(s, v) => Err(format!("does not call `{v}`")),
            None if !scan.tests.iter().any(|l| s.holds(*l)) => Err("has no `#[test]`".into()),
            _ => Ok(()),
        }
    }

    fn runs(&self, pkg: &str, mode: Option<Mode>) -> Result<(), String> {
        let k = self
            .repo
            .package(pkg)
            .ok_or(format!("`{pkg}` is not a package"))?;
        let has = match mode {
            Some(Mode::Test) => self.repo.has_tests(k),
            Some(Mode::Build) => k.root.is_some(),
            Some(Mode::Bin) => k.bin,
            None => true,
        };
        match has {
            true => Ok(()),
            false => Err(format!("`{pkg}` has no target for {mode:?}")),
        }
    }

    /// Registrants of `of` with no `has`, by key.
    pub fn each(&self, e: &Each) -> Result<(), String> {
        let of = self.repo.resolve(&e.of);
        let reg = self
            .inv
            .find(&of, &e.of)
            .ok_or(format!("`{}` is not a registry", e.of))?;
        let has = self.resolve(&e.has, Class::Symbol)?;
        let keys = self.keys(&e.has, &has);
        let mut missing: Vec<String> = Vec::new();
        for m in &self.inv.registries[reg].members {
            if !keys.contains(&m.key) {
                missing.push(format!("{} ({}:{})", m.key, m.file, m.line));
            }
        }
        match missing.is_empty() {
            true => Ok(()),
            false => Err(format!("no `{}` for {}", e.has, missing.join(", "))),
        }
    }

    /// Keys `has` holds: registrants, call arguments or literal fields.
    fn keys(&self, has: &str, items: &[Item]) -> BTreeSet<String> {
        if let Some(r) = self.inv.find(items, has) {
            return self.inv.registries[r]
                .members
                .iter()
                .map(|m| m.key.clone())
                .collect();
        }
        let (name, mac) = (last_of(has), has.ends_with('!'));
        let mut out = BTreeSet::new();
        for file in self.repo.holding(name) {
            let whole = Scope {
                file: file.to_string(),
                start: 0,
                end: usize::MAX,
            };
            let Some(scan) = self.repo.scan(file).filter(|_| self.mentions(&whole, has)) else {
                continue;
            };
            let mut add = |args: &proc_macro2::TokenStream| {
                out.extend(top_lits(args));
                out.extend(top_idents(args));
            };
            match mac {
                true => scan
                    .calls
                    .iter()
                    .filter(|c| c.name == name)
                    .for_each(|c| add(&c.args)),
                false => scan
                    .bodies
                    .iter()
                    .filter(|b| b.name == name)
                    .for_each(|b| add(&b.args)),
            }
        }
        out
    }
}
