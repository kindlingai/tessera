//! Fails on near-copies of code, using EmbeddingGemma 2 embeddings.

mod cache;
mod far;
mod fetch;
mod pairs;
mod report;
mod windows;

#[cfg(test)]
mod cache_test;
#[cfg(test)]
mod far_test;
#[cfg(test)]
mod pairs_test;
#[cfg(test)]
mod report_test;
#[cfg(test)]
mod windows_test;

use anyhow::{Context, Result, bail};
use cache::{Cache, key};
use pairs::{Allow, near};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use windows::{Window, cut};

const USAGE: &str = "usage: tessera-simcheck fetch <model_dir>
       tessera-simcheck check <model_dir> [cache_dir [json_file [base_ref]]] [--budget <n>] [--warm] [--staged]
       tessera-simcheck far <model_dir> [cache_dir]
       tessera-simcheck place <model_dir> <cache_dir> [paths_file]

`check` writes its findings to <json_file> when given. It embeds every
uncached window of files changed since <base_ref>, then at most <n> other
uncached windows, and compares only embedded windows. `--warm` never fails.
It also fails on allow.toml pairs that allow nothing. `--staged` compares only
pairs that hold a window of a staged file, and skips the allow.toml check.
With <base_ref>, it also reports new files nearer another crate.
`far` lists crates whose code leans toward each other.
`place` lists each file's lean toward another crate.";

/// The repository's lint.toml keys this tool reads.
#[derive(serde::Deserialize)]
struct Settings {
    similar_window: usize,
    similar_spacing: usize,
    max_similarity: f32,
    max_lean: f32,
    extensions: Vec<String>,
    similar_allow: String,
}

/// The repository root, and its lint.toml.
fn settings() -> Result<(PathBuf, Settings)> {
    let root = PathBuf::from(git(Path::new("."), &["rev-parse", "--show-toplevel"])?.trim());
    let path = root.join("lint.toml");
    let text = std::fs::read_to_string(&path).with_context(|| path.display().to_string())?;
    Ok((root, toml::from_str(&text)?))
}

/// Texts per call to the model, and per progress line.
const BATCH: usize = 64;

fn git(root: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git").args(args).current_dir(root).output()?;
    if !out.status.success() {
        bail!("git {args:?} failed");
    }
    Ok(String::from_utf8(out.stdout)?)
}

/// Windows of every tracked or new source file under `root`.
fn all_windows(root: &Path, s: &Settings) -> Result<Vec<Window>> {
    let listed = git(
        root,
        &["ls-files", "--cached", "--others", "--exclude-standard"],
    )?;
    let mut out = Vec::new();
    for path in listed.lines() {
        let ext = Path::new(path).extension().and_then(|e| e.to_str());
        if !ext.is_some_and(|e| s.extensions.iter().any(|x| x == e)) {
            continue;
        }
        // Deleted but still in the index.
        let Ok(text) = std::fs::read_to_string(root.join(path)) else {
            continue;
        };
        // Both sides of a conflict match. tessera-lint reports the file.
        if tessera_shape::conflict(&text).is_some() {
            continue;
        }
        out.extend(cut(path, &text, s.similar_window, s.similar_spacing));
    }
    Ok(out)
}

/// Embeds `changed` files, then `budget` other windows.
fn embed(
    model: &Path,
    wins: &[Window],
    cache: &mut Cache,
    changed: &HashSet<String>,
    budget: Option<usize>,
) -> Result<Vec<Option<Vec<f32>>>> {
    let mut required = Vec::new();
    let mut optional = Vec::new();
    for w in wins {
        let k = key(&w.text);
        if cache.get(k).is_some() {
            continue;
        }
        match changed.contains(&w.path) {
            true => required.push((k, w.text.as_str())),
            false => optional.push((k, w.text.as_str())),
        }
    }
    // Each run starts where the last run stopped.
    optional.sort_unstable();
    optional.dedup();
    optional.truncate(budget.unwrap_or(usize::MAX));
    let mut missing: Vec<&str> = required.iter().chain(&optional).map(|(_, t)| *t).collect();
    missing.dedup();
    eprintln!("{} windows, {} to embed", wins.len(), missing.len());
    if !missing.is_empty() {
        let m = eg2cpu::Model::load(model).context("loading the model; run `fetch` first")?;
        let start = std::time::Instant::now();
        for (i, chunk) in missing.chunks(BATCH).enumerate() {
            for (t, out) in chunk.iter().zip(m.embed_batch(chunk)?) {
                cache.insert(key(t), out.embedding);
            }
            let done = (i * BATCH + chunk.len()).min(missing.len());
            let pct = done * 100 / missing.len();
            let secs = start.elapsed().as_secs();
            eprintln!("embedded {done}/{} ({pct}%) in {secs}s", missing.len());
        }
    }
    Ok(wins
        .iter()
        .map(|w| cache.get(key(&w.text)).cloned())
        .collect())
}

/// Files staged, or changed between `base` and HEAD.
fn changed_files(root: &Path, base: Option<&str>, staged: bool) -> Result<HashSet<String>> {
    let range;
    let which = match (staged, base) {
        (true, _) => "--cached",
        (false, Some(base)) => {
            range = format!("{base}...HEAD");
            &range
        }
        (false, None) => return Ok(HashSet::new()),
    };
    let diff = git(root, &["diff", "--name-only", which])?;
    Ok(diff.lines().map(String::from).collect())
}

/// A `check` run's embedding scope, and if it fails.
#[derive(Default)]
struct Mode {
    /// Uncached windows to embed outside changed files. `None` embeds all.
    budget: Option<usize>,
    /// Never fails. Builds on main set it.
    warm: bool,
    /// Compares only pairs that hold a staged file's window.
    staged: bool,
}

/// Changed files count from the `base` ref.
fn check(
    model: &Path,
    cache_file: Option<&Path>,
    (json, base): (Option<&Path>, Option<&str>),
    mode: &Mode,
) -> Result<bool> {
    let (root, s) = settings()?;
    let allow =
        std::fs::read_to_string(root.join(&s.similar_allow)).context(s.similar_allow.clone())?;
    let allow = Allow::parse(&allow)?;
    let wins = all_windows(&root, &s)?;
    let mut cache = Cache::open(cache_file)?;
    let changed = changed_files(&root, base, mode.staged)?;
    let found = embed(model, &wins, &mut cache, &changed, mode.budget)?;
    if let Some(f) = cache_file {
        let keys: Vec<u64> = wins.iter().map(|w| key(&w.text)).collect();
        cache.save(f, &keys)?;
    }
    // Compare only the embedded windows.
    let total = wins.len();
    let (mut kept, mut embs) = (Vec::new(), Vec::new());
    for (w, e) in wins.into_iter().zip(found) {
        if let Some(e) = e {
            kept.push(w);
            embs.push(e);
        }
    }
    let wins = kept;
    let coverage = (wins.len() < total).then_some((wins.len(), total));
    if let Some((done, all)) = coverage {
        println!("coverage: {done} of {all} windows embedded");
    }
    let focus: Option<Vec<bool>> = mode
        .staged
        .then(|| wins.iter().map(|w| changed.contains(&w.path)).collect());
    let limits = (s.similar_window, s.max_similarity);
    let (hits, used) = near(&wins, &embs, limits, &allow, focus.as_deref());
    // Allowed pairs may hold unembedded or unfocused windows.
    let stale = match (coverage, &focus) {
        (None, None) => allow.stale(&used),
        _ => Vec::new(),
    };
    let leans = new_leans(&root, &s, (&wins, &embs), base)?;
    for l in &leans {
        println!(
            "{}: lean: {:.3} to {}, {:.3} to its own {}",
            l.path, l.near, l.other, l.own, l.home
        );
    }
    for h in &hits {
        let (a, b) = (&wins[h.a], &wins[h.b]);
        println!(
            "{}:{}: similar: {:.3} to {}:{} ({} window pairs at or above {})",
            a.path, a.line, h.score, b.path, b.line, h.count, s.max_similarity
        );
    }
    for (a, b) in &stale {
        println!(
            "{}: stale: {a} and {b} are not near-copies",
            s.similar_allow
        );
    }
    if let Some(path) = json {
        let r = report::report(
            (&hits, &stale, &leans),
            &wins,
            (&root, &s.similar_allow),
            (limits, coverage),
        )?;
        r.write(path).map_err(anyhow::Error::msg)?;
    }
    Ok(mode.warm || (hits.is_empty() && stale.is_empty()))
}

/// Crates this many directory steps apart count as far.
const FAR_STEPS: usize = 2;

/// Crate directories, relative to `root`.
fn crates(root: &Path) -> Result<Vec<String>> {
    let listed = git(
        root,
        &["ls-files", "--cached", "--others", "--exclude-standard"],
    )?;
    Ok(listed
        .lines()
        .filter_map(|p| p.strip_suffix("Cargo.toml"))
        .map(|d| d.trim_end_matches('/').to_string())
        .collect())
}

/// Proc-macro crates must stay apart, so files never move in.
fn movable(root: &Path, crates: &[String]) -> Vec<bool> {
    crates
        .iter()
        .map(|c| {
            let toml = std::fs::read_to_string(root.join(c).join("Cargo.toml"));
            !toml.is_ok_and(|t| t.contains("proc-macro = true"))
        })
        .collect()
}

/// Files new since `base` that read like another crate.
fn new_leans(
    root: &Path,
    s: &Settings,
    scan: (&[Window], &[Vec<f32>]),
    base: Option<&str>,
) -> Result<Vec<far::Lean>> {
    let (wins, embs) = scan;
    let Some(base) = base else {
        return Ok(Vec::new());
    };
    let range = format!("{base}...HEAD");
    let diff = git(root, &["diff", "--name-only", "--diff-filter=A", &range])?;
    let added: Vec<String> = diff.lines().map(String::from).collect();
    let keep: Vec<usize> = (0..wins.len())
        .filter(|&i| !far::test(&wins[i].path))
        .collect();
    let wins: Vec<Window> = keep.iter().map(|&i| wins[i].clone()).collect();
    let embs: Vec<Vec<f32>> = keep.iter().map(|&i| embs[i].clone()).collect();
    let crates = crates(root)?;
    let movable = movable(root, &crates);
    Ok(far::leans(
        &wins,
        &embs,
        &crates,
        &movable,
        (&added, s.max_lean),
    ))
}

/// Non-test windows, their embeddings, and the crates.
type Tree = (Vec<Window>, Vec<Vec<f32>>, Vec<String>);

/// Embeds the tree.
fn tree(model: &Path, cache_file: Option<&Path>) -> Result<Tree> {
    let (root, s) = settings()?;
    let wins = all_windows(&root, &s)?;
    let mut cache = Cache::open(cache_file)?;
    // `far` and `place` need every window.
    let found = embed(model, &wins, &mut cache, &HashSet::new(), None)?;
    let embs: Vec<Vec<f32>> = found.into_iter().map(|e| e.expect("embedded")).collect();
    if let Some(f) = cache_file {
        let keys: Vec<u64> = wins.iter().map(|w| key(&w.text)).collect();
        cache.save(f, &keys)?;
    }
    let keep: Vec<usize> = (0..wins.len())
        .filter(|&i| !far::test(&wins[i].path))
        .collect();
    let embs: Vec<Vec<f32>> = keep.iter().map(|&i| embs[i].clone()).collect();
    let wins: Vec<Window> = wins.into_iter().filter(|w| !far::test(&w.path)).collect();
    Ok((wins, embs, crates(&root)?))
}

/// Prints each file's lean, or only those in `only`.
fn place(model: &Path, cache_file: &Path, only: Option<&Path>) -> Result<bool> {
    let (wins, embs, crates) = tree(model, Some(cache_file))?;
    let only: Option<Vec<String>> = match only {
        Some(f) => Some(
            std::fs::read_to_string(f)?
                .lines()
                .map(String::from)
                .collect(),
        ),
        None => None,
    };
    for f in far::files(&wins, &embs, &crates, &vec![true; crates.len()]) {
        if only.as_ref().is_some_and(|o| !o.contains(&f.path)) {
            continue;
        }
        let (o, n, h, c) = (f.own, f.near, &crates[f.home], &crates[f.other]);
        println!(
            "{:+.3}\t{}\t{o:.3}\t{n:.3}\t{}\t{h}\t{c}",
            n - o,
            f.windows,
            f.path
        );
    }
    Ok(true)
}

/// Prints crate pairs whose code leans toward each other.
fn far(model: &Path, cache_file: Option<&Path>) -> Result<bool> {
    let (wins, embs, crates) = tree(model, cache_file)?;
    let files = far::files(&wins, &embs, &crates, &vec![true; crates.len()]);
    println!("pull   windows a->b  b->a  steps  crates");
    for p in far::pairs(&wins, &embs, &crates) {
        if p.steps < FAR_STEPS {
            continue;
        }
        println!(
            "{:.2}  {:>4}/{:<4}  {:>4}  {:>4}  {:>5}  {} | {}",
            far::pull(&p),
            p.sizes.0,
            p.sizes.1,
            p.leans.0,
            p.leans.1,
            p.steps,
            crates[p.a],
            crates[p.b]
        );
        let pair =
            |f: &&far::File| (f.home, f.other) == (p.a, p.b) || (f.home, f.other) == (p.b, p.a);
        for f in files.iter().filter(|f| f.near > f.own).filter(pair) {
            println!(
                "        {:+.3}  {} -> {}",
                f.near - f.own,
                f.path,
                crates[f.other]
            );
        }
    }
    Ok(true)
}

/// Parses `check` paths and its flags.
fn check_args<'a>(args: &[&'a str]) -> Option<(Vec<&'a str>, Mode)> {
    let (mut paths, mut mode) = (Vec::new(), Mode::default());
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match *a {
            "--budget" => mode.budget = Some(it.next()?.parse().ok()?),
            "--warm" => mode.warm = true,
            "--staged" => mode.staged = true,
            p => paths.push(p),
        }
    }
    (paths.len() <= 3).then_some((paths, mode))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let run = match args[..] {
        ["fetch", dir] => fetch::fetch(Path::new(dir)).map(|()| true),
        ["check", dir, ref rest @ ..] => match check_args(rest) {
            Some((paths, mode)) => {
                let path = |i: usize| paths.get(i).map(Path::new);
                check(
                    Path::new(dir),
                    path(0),
                    (path(1), paths.get(2).copied()),
                    &mode,
                )
            }
            None => {
                eprintln!("{USAGE}");
                return ExitCode::from(2);
            }
        },
        ["far", dir] => far(Path::new(dir), None),
        ["far", dir, cache] => far(Path::new(dir), Some(Path::new(cache))),
        ["place", dir, cache] => place(Path::new(dir), Path::new(cache), None),
        ["place", dir, cache, only] => {
            place(Path::new(dir), Path::new(cache), Some(Path::new(only)))
        }
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => {
            eprintln!("tessera-simcheck: failed. Share the code, or update allow.toml.");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("tessera-simcheck: {e:#}");
            ExitCode::from(2)
        }
    }
}
