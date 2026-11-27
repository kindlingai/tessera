//! Kernel files that replace planned kernels, read again on change.

use crate::Error;
use crate::kernels::{fixed_text, group};
use crate::run::{Outcome, Reload};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs::Metadata;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;
use std::time::SystemTime;
use tessera_codegen::Dialect;
use tessera_gpukit::backend::Backend;
use tessera_gpukit::launch::THREADS_PER_BLOCK;
use tessera_lower::{Shape, Source, Step};

/// Flattens a raw recipe kernel with `<tessera/...>` imports.
pub type Flattener =
    fn(Backend, &str, &str, &dyn Fn(&str) -> Option<String>) -> Result<String, String>;

static FLATTENER: OnceLock<Flattener> = OnceLock::new();

const RESIDENT_LAUNCH_ERR: &str = "launch.toml overrides are unsupported on resident kernels";

/// Registers the `<tessera/...>` kernel flattener for overrides.
pub fn set_flattener(flattener: Flattener) {
    let _ = FLATTENER.set(flattener);
}

/// Optional launch and compiler overrides from `<entry>.launch.toml`.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LaunchOverride {
    pub threads: Option<u32>,
    pub dynamic_shared: Option<u32>,
    #[serde(default)]
    pub nvrtc_flags: Vec<String>,
}

impl LaunchOverride {
    fn parse(text: &str) -> Result<Self, String> {
        let parsed: Self = toml::from_str(text).map_err(|e| format!("launch.toml: {e}"))?;
        if parsed.threads == Some(0) {
            return Err("launch.toml: threads must be > 0".into());
        }
        Ok(parsed)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.threads.is_none() && self.dynamic_shared.is_none() && self.nvrtc_flags.is_empty()
    }

    pub(crate) fn flags(&self) -> Vec<&str> {
        self.nvrtc_flags.iter().map(String::as_str).collect()
    }
}

/// Applies optional launch overrides to `shape` and `shared`.
pub(crate) fn apply_launch(
    shape: Shape,
    shared: u32,
    over: Option<&LaunchOverride>,
) -> (Shape, u32) {
    let Some(over) = over else {
        return (shape, shared);
    };
    let shape = match shape {
        Shape::Grid {
            across,
            down,
            threads,
        } => Shape::Grid {
            across,
            down,
            threads: over.threads.map_or(threads, |t| t as usize),
        },
        Shape::Line(n) if n > 0 && (over.threads.is_some() || over.dynamic_shared.is_some()) => {
            let t = over
                .threads
                .unwrap_or_else(|| THREADS_PER_BLOCK.min(n as u32));
            Shape::Grid {
                across: n.div_ceil(t as usize),
                down: 1,
                threads: t as usize,
            }
        }
        other => other,
    };
    (shape, over.dynamic_shared.unwrap_or(shared))
}

/// A file's change time, length and inode.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Version {
    modified: SystemTime,
    len: u64,
    inode: u64,
}

impl Version {
    fn of(path: &Path) -> Option<Self> {
        let meta = std::fs::metadata(path).ok()?;
        Some(Self {
            modified: meta.modified().ok()?,
            len: meta.len(),
            inode: inode(&meta),
        })
    }
}

#[cfg(unix)]
fn inode(meta: &Metadata) -> u64 {
    std::os::unix::fs::MetadataExt::ino(meta)
}

#[cfg(not(unix))]
fn inode(_: &Metadata) -> u64 {
    0
}

/// The name a kernel file takes, before its extension.
pub(crate) fn entry(source: &Source) -> &str {
    source.entry()
}

/// The text a GPU runner compiles for `step` on `backend`.
pub fn text(step: &Step, backend: Backend) -> String {
    let source = step.kernel.source();
    match (&source, fixed_text(&source, backend)) {
        (Source::Made(body), _) => body.text().bounded(Dialect::of(backend), Some(group(step))),
        (_, fixed) => fixed.unwrap_or_default().0.to_string(),
    }
}

/// Prepared source and launch sidecar for one override entry.
pub(crate) struct PreparedOverride {
    pub source: String,
    pub launch: Option<LaunchOverride>,
}

/// Watched header paths with their last observed versions.
type Deps = Vec<(PathBuf, Option<Version>)>;

/// File versions observed during one scan.
pub(crate) struct Observed {
    source: Option<Version>,
    launch: Option<Version>,
    deps: Deps,
}

/// A file change detected on disk before committing state.
pub(crate) enum Pending {
    /// File matches planned text with no active override.
    Unchanged { entry: String, seen: Observed },
    /// Active override was deleted or reverted to planned text.
    Revert {
        entry: String,
        path: PathBuf,
        seen: Observed,
    },
    /// Override source, headers or sidecar failed validation.
    Failed {
        entry: String,
        path: PathBuf,
        seen: Observed,
        why: String,
    },
    /// File or sidecar changed and needs compilation.
    Compile {
        entry: String,
        path: PathBuf,
        seen: Observed,
        text: String,
        launch: Option<LaunchOverride>,
    },
}

/// Active compiled override for one entry.
struct Active<K> {
    kernel: K,
    text: String,
    launch: Option<LaunchOverride>,
}

/// A compiled or reverted override change waiting to commit.
pub(crate) struct Ready<K> {
    entry: String,
    seen: Observed,
    active: Option<Option<Active<K>>>,
    reload: Option<Reload>,
}

impl<K> Ready<K> {
    pub(crate) fn unchanged(entry: String, seen: Observed) -> Self {
        Self {
            entry,
            seen,
            active: None,
            reload: None,
        }
    }

    pub(crate) fn reverted(entry: String, path: PathBuf, seen: Observed) -> Self {
        Self {
            entry,
            seen,
            active: Some(None),
            reload: Some(Reload {
                path,
                outcome: Outcome::Planned,
            }),
        }
    }

    pub(crate) fn loaded(
        entry: String,
        path: PathBuf,
        seen: Observed,
        kernel: K,
        text: String,
        launch: Option<LaunchOverride>,
    ) -> Self {
        Self {
            entry,
            seen,
            active: Some(Some(Active {
                kernel,
                text,
                launch,
            })),
            reload: Some(Reload {
                path,
                outcome: Outcome::Loaded,
            }),
        }
    }

    pub(crate) fn failed(entry: String, path: PathBuf, seen: Observed, why: String) -> Self {
        Self {
            entry,
            seen,
            active: None,
            reload: Some(Reload {
                path,
                outcome: Outcome::Failed(why),
            }),
        }
    }
}

/// A file's last seen versions, headers and active kernel.
struct File<K> {
    source: Option<Version>,
    launch: Option<Version>,
    deps: Deps,
    active: Option<Active<K>>,
}

/// Kernel files in one directory.
pub(crate) struct Overrides<K> {
    dir: PathBuf,
    backend: Backend,
    dialect: Dialect,
    files: HashMap<String, File<K>>,
}

/// Loads one entry override and optional sidecar for offline checks.
pub(crate) fn load_entry(
    dir: &Path,
    backend: Backend,
    entry: &str,
    resident: bool,
    planned: impl FnOnce() -> String,
) -> Result<Option<PreparedOverride>, String> {
    prepare_entry(dir, backend, Dialect::of(backend), entry, resident, planned).0
}

/// Loads one standalone file with local headers and sidecar.
pub(crate) fn load_file(
    path: &Path,
    backend: Backend,
) -> Result<Result<(String, Option<LaunchOverride>), String>, Error> {
    let raw =
        std::fs::read_to_string(path).map_err(|e| Error::KernelFile(path.to_path_buf(), e))?;
    Ok(prepare_file(path, backend, &raw))
}

fn prepare_file(
    path: &Path,
    backend: Backend,
    raw: &str,
) -> Result<(String, Option<LaunchOverride>), String> {
    let root = path.parent().unwrap_or(Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("kernel");
    let dialect = Dialect::of(backend);
    let source = prepare_source(root, backend, dialect, stem, raw).0?;
    let sidecar = root.join(format!("{stem}.launch.toml"));
    let launch = match std::fs::read_to_string(&sidecar) {
        Ok(text) => {
            let parsed = LaunchOverride::parse(&text)?;
            (!parsed.is_empty()).then_some(parsed)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(format!("launch.toml: {e}")),
    };
    Ok((source, launch))
}

fn prepare_entry(
    dir: &Path,
    backend: Backend,
    dialect: Dialect,
    entry: &str,
    resident: bool,
    planned: impl FnOnce() -> String,
) -> (Result<Option<PreparedOverride>, String>, Observed) {
    let path = dir.join(format!("{entry}.{}", dialect.extension()));
    let sidecar = dir.join(format!("{entry}.launch.toml"));
    let now_src = Version::of(&path);
    let now_cfg = Version::of(&sidecar);
    let mut deps = Vec::new();
    let prepared = match now_src {
        Some(_) => match std::fs::read_to_string(&path) {
            Ok(raw) => {
                let (res, found) = prepare_source(dir, backend, dialect, entry, &raw);
                deps = found;
                Some(res)
            }
            Err(e) => Some(Err(e.to_string())),
        },
        None => None,
    };
    let seen = Observed {
        source: now_src,
        launch: now_cfg,
        deps,
    };
    let launch_ov = match now_cfg {
        Some(_) => match std::fs::read_to_string(&sidecar)
            .map_err(|e| format!("launch.toml: {e}"))
            .and_then(|t| LaunchOverride::parse(&t))
        {
            Ok(ov) => (!ov.is_empty()).then_some(ov),
            Err(msg) => return (Err(msg), seen),
        },
        None => None,
    };
    if resident && launch_ov.is_some() {
        return (Err(RESIDENT_LAUNCH_ERR.into()), seen);
    }
    let expanded = match prepared {
        Some(Ok(text)) => Some(text),
        Some(Err(msg)) => return (Err(msg), seen),
        None => None,
    };
    let base = planned();
    let src_over = expanded.filter(|t| *t != base);
    let out = match (src_over, launch_ov) {
        (Some(source), launch) => Some(PreparedOverride { source, launch }),
        (None, Some(launch)) => Some(PreparedOverride {
            source: base,
            launch: Some(launch),
        }),
        (None, None) => None,
    };
    (Ok(out), seen)
}

impl<K> Overrides<K> {
    pub(crate) fn new(dir: PathBuf, backend: Backend) -> Self {
        Self {
            dir,
            backend,
            dialect: Dialect::of(backend),
            files: HashMap::new(),
        }
    }

    pub(crate) fn path(&self, entry: &str) -> PathBuf {
        self.dir
            .join(format!("{entry}.{}", self.dialect.extension()))
    }

    #[cfg(test)]
    pub(crate) fn launch_path(&self, entry: &str) -> PathBuf {
        self.dir.join(format!("{entry}.launch.toml"))
    }

    /// Inspects `entry`'s file, headers and sidecar without mutating state.
    pub(crate) fn scan(
        &self,
        entry: &str,
        resident: bool,
        planned: impl FnOnce() -> String,
    ) -> Option<Pending> {
        let path = self.path(entry);
        let sidecar = self.dir.join(format!("{entry}.launch.toml"));
        let now_src = Version::of(&path);
        let now_cfg = Version::of(&sidecar);
        let prev = self.files.get(entry);
        let same = match prev {
            Some(f) => {
                f.source == now_src
                    && f.launch == now_cfg
                    && f.deps.iter().all(|(p, v)| Version::of(p) == *v)
            }
            None => now_src.is_none() && now_cfg.is_none(),
        };
        if same {
            return None;
        }
        let (res, seen) = prepare_entry(
            &self.dir,
            self.backend,
            self.dialect,
            entry,
            resident,
            planned,
        );
        Some(match res {
            Err(why) => Pending::Failed {
                entry: entry.to_string(),
                path,
                seen,
                why,
            },
            Ok(None) if prev.is_some_and(|f| f.active.is_some()) => Pending::Revert {
                entry: entry.to_string(),
                path,
                seen,
            },
            Ok(None) => Pending::Unchanged {
                entry: entry.to_string(),
                seen,
            },
            Ok(Some(prep)) => Pending::Compile {
                entry: entry.to_string(),
                path,
                seen,
                text: prep.source,
                launch: prep.launch,
            },
        })
    }

    /// Commits one prepared change into recorded state.
    pub(crate) fn commit(&mut self, ready: Ready<K>) -> Option<Reload> {
        let file = self.files.entry(ready.entry).or_insert(File {
            source: None,
            launch: None,
            deps: Vec::new(),
            active: None,
        });
        file.source = ready.seen.source;
        file.launch = ready.seen.launch;
        file.deps = ready.seen.deps;
        if let Some(active) = ready.active {
            file.active = active;
        }
        ready.reload
    }

    /// Reads `entry`'s file, headers or sidecar if changed.
    #[cfg(test)]
    pub(crate) fn check(
        &mut self,
        entry: &str,
        resident: bool,
        planned: impl FnOnce() -> String,
        compile: impl FnOnce(&str, &[&str]) -> Result<K, Error>,
    ) -> Option<Reload> {
        let ready = match self.scan(entry, resident, planned)? {
            Pending::Unchanged { entry, seen } => Ready::unchanged(entry, seen),
            Pending::Revert { entry, path, seen } => Ready::reverted(entry, path, seen),
            Pending::Failed {
                entry,
                path,
                seen,
                why,
            } => Ready::failed(entry, path, seen, why),
            Pending::Compile {
                entry,
                path,
                seen,
                text,
                launch,
            } => {
                let flags = launch.as_ref().map_or_else(Vec::new, LaunchOverride::flags);
                match compile(&text, &flags) {
                    Ok(kernel) => Ready::loaded(entry, path, seen, kernel, text, launch),
                    Err(e) => Ready::failed(entry, path, seen, e.to_string()),
                }
            }
        };
        self.commit(ready)
    }

    pub(crate) fn kernel(&self, entry: &str) -> Option<&K> {
        Some(&self.files.get(entry)?.active.as_ref()?.kernel)
    }

    /// The text of `entry`'s loaded file.
    pub(crate) fn text(&self, entry: &str) -> Option<&str> {
        Some(&self.files.get(entry)?.active.as_ref()?.text)
    }

    /// Launch override for `entry`'s loaded file, if any.
    pub(crate) fn launch(&self, entry: &str) -> Option<&LaunchOverride> {
        self.files.get(entry)?.active.as_ref()?.launch.as_ref()
    }

    /// Writes `text` as `entry`'s file, unless one exists.
    pub(crate) fn write(&self, entry: &str, text: &str) -> Result<Option<PathBuf>, Error> {
        let path = self.path(entry);
        if path.exists() {
            return Ok(None);
        }
        let failed = |e: std::io::Error| Error::KernelFile(path.clone(), e);
        std::fs::create_dir_all(&self.dir).map_err(failed)?;
        std::fs::write(&path, text).map_err(failed)?;
        Ok(Some(path))
    }
}

/// Strips `//` and `/* */` comments from one line.
fn strip_comments(line: &str, open: &mut bool) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let (mut i, mut start) = (0, 0);
    while i < bytes.len() {
        if *open {
            if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                *open = false;
                i += 2;
                start = i;
            } else {
                i += 1;
            }
            continue;
        }
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            out.push_str(&line[start..i]);
            return out;
        }
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            out.push_str(&line[start..i]);
            out.push(' ');
            *open = true;
            i += 2;
            continue;
        }
        i += 1;
    }
    if !*open {
        out.push_str(&line[start..]);
    }
    out
}

/// Parses `#include "..."` or `#include <...>` on one code line.
fn parse_include(code: &str) -> Option<(bool, &str)> {
    let dir = code.trim_start().strip_prefix('#')?.trim_start();
    let rest = dir.strip_prefix("include")?;
    if !rest.starts_with(|c: char| c.is_whitespace() || c == '"' || c == '<') {
        return None;
    }
    let op = rest.trim();
    if let Some(p) = op.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        return Some((false, p));
    }
    if let Some(p) = op.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
        return Some((true, p));
    }
    None
}

/// Validates a relative header path inside the overrides directory.
fn safe_rel(raw: &str) -> Result<PathBuf, String> {
    if raw.is_empty() || raw.starts_with('/') || raw.contains('\\') || raw.contains(':') {
        return Err(format!("include path {raw:?} escapes override directory"));
    }
    let mut out = PathBuf::new();
    for comp in Path::new(raw).components() {
        match comp {
            Component::Normal(p) => out.push(p),
            Component::CurDir => {}
            _ => return Err(format!("include path {raw:?} escapes override directory")),
        }
    }
    if out.as_os_str().is_empty() {
        return Err(format!("include path {raw:?} is empty"));
    }
    Ok(out)
}

/// Expands local includes and flattens root or transitive `<tessera/...>` imports.
fn prepare_source(
    root: &Path,
    backend: Backend,
    dialect: Dialect,
    entry: &str,
    raw: &str,
) -> (Result<String, String>, Deps) {
    let file_name = format!("{entry}.{}", dialect.extension());
    let mut expander = Expander {
        root,
        deps: Vec::new(),
        once: HashSet::new(),
        open: Vec::new(),
        tessera: Vec::new(),
        expanded_local: false,
    };
    let root_path = root.join(&file_name);
    if has_pragma_once(raw) {
        expander.once.insert(root_path.clone());
    }
    expander.open.push(root_path);
    let res = expander.expand(&file_name, Path::new(""), raw, false);
    let expanded = match res {
        Ok(text) => text,
        Err(e) => return (Err(e), expander.deps),
    };
    if expander.tessera.is_empty() {
        return (Ok(expanded), expander.deps);
    }
    let Some(flattener) = FLATTENER.get() else {
        return (
            Err("override uses <tessera/...> includes, but no flattener is registered".into()),
            expander.deps,
        );
    };
    let input = if expander.expanded_local {
        let mut hoisted = String::with_capacity(expanded.len() + 128);
        for imp in &expander.tessera {
            hoisted.push_str(&format!("#include <{imp}>\n"));
        }
        hoisted.push_str(&format!("#line 1 \"{file_name}\"\n"));
        hoisted.push_str(&expanded);
        hoisted
    } else {
        raw.to_string()
    };
    let deps = RefCell::new(expander.deps);
    let load = |rel: &str| -> Option<String> {
        let clean = safe_rel(rel).ok()?;
        let full = root.join(&clean);
        let ver = Version::of(&full);
        let mut list = deps.borrow_mut();
        if !list.iter().any(|(p, _)| *p == full) {
            list.push((full.clone(), ver));
        }
        std::fs::read_to_string(&full).ok()
    };
    let flat = flattener(backend, &file_name, &input, &load);
    (flat, deps.into_inner())
}

struct Expander<'a> {
    root: &'a Path,
    deps: Deps,
    once: HashSet<PathBuf>,
    open: Vec<PathBuf>,
    tessera: Vec<String>,
    expanded_local: bool,
}

impl Expander<'_> {
    fn track(&mut self, path: &Path) -> Option<Version> {
        let ver = Version::of(path);
        if !self.deps.iter().any(|(p, _)| p == path) {
            self.deps.push((path.to_path_buf(), ver));
        }
        ver
    }

    fn resolve(&self, parent: &Path, rel: &Path) -> PathBuf {
        let joined = self.root.join(parent).join(rel);
        if joined.exists() {
            return joined;
        }
        let fallback = self.root.join(rel);
        if fallback.exists() {
            return fallback;
        }
        joined
    }

    fn record_tessera(&mut self, target: &str) {
        if !self.tessera.iter().any(|t| t == target) {
            self.tessera.push(target.to_string());
        }
    }

    fn expand(
        &mut self,
        label: &str,
        parent: &Path,
        text: &str,
        in_header: bool,
    ) -> Result<String, String> {
        let mut open = false;
        let mut has_directive = false;
        for line in text.lines() {
            let code = strip_comments(line, &mut open);
            match parse_include(&code) {
                Some((false, _)) => has_directive = true,
                Some((true, target)) if target.starts_with("tessera/") => has_directive = true,
                _ => {}
            }
        }
        if !has_directive && !in_header {
            return Ok(text.to_string());
        }
        let mut out = String::with_capacity(text.len());
        let mut open = false;
        for (idx, line) in text.lines().enumerate() {
            let code = strip_comments(line, &mut open);
            let trimmed = code.trim();
            if in_header && is_pragma_once(trimmed) {
                out.push('\n');
                continue;
            }
            if let Some((true, target)) = parse_include(trimmed)
                && target.starts_with("tessera/")
            {
                self.record_tessera(target);
                out.push('\n');
                continue;
            }
            if let Some((false, raw_path)) = parse_include(trimmed) {
                self.expanded_local = true;
                let rel = safe_rel(raw_path)?;
                let full = self.resolve(parent, &rel);
                self.track(&full);
                if self.once.contains(&full) {
                    out.push('\n');
                    continue;
                }
                if self.open.contains(&full) {
                    return Err(format!("include cycle on {raw_path:?}"));
                }
                let body = std::fs::read_to_string(&full)
                    .map_err(|e| format!("missing header {raw_path:?}: {e}"))?;
                if has_pragma_once(&body) {
                    self.once.insert(full.clone());
                }
                let next_parent = full
                    .strip_prefix(self.root)
                    .ok()
                    .and_then(Path::parent)
                    .unwrap_or(Path::new(""))
                    .to_path_buf();
                self.open.push(full.clone());
                let nested = self.expand(raw_path, &next_parent, &body, true)?;
                self.open.pop();
                out.push_str(&format!("#line 1 \"{raw_path}\"\n"));
                out.push_str(&nested);
                if !nested.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str(&format!("#line {} \"{label}\"\n", idx + 2));
                continue;
            }
            out.push_str(line);
            out.push('\n');
        }
        if !text.ends_with('\n') && out.ends_with('\n') {
            out.pop();
        }
        Ok(out)
    }
}

fn is_pragma_once(code: &str) -> bool {
    code.strip_prefix('#')
        .map(str::trim_start)
        .and_then(|d| d.strip_prefix("pragma"))
        .is_some_and(|rest| rest.trim() == "once")
}

fn has_pragma_once(text: &str) -> bool {
    let mut open = false;
    text.lines()
        .any(|l| is_pragma_once(strip_comments(l, &mut open).trim()))
}
