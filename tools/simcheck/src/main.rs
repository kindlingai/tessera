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
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use windows::{Window, cut};

const USAGE: &str = "usage: tessera-simcheck fetch <model_dir>
       tessera-simcheck check <model_dir> [cache_file [json_file [base_ref]]]
       tessera-simcheck far <model_dir> [cache_file]
       tessera-simcheck place <model_dir> <cache_file> [paths_file]

`check` writes its findings to <json_file> when given.
It also fails on allow.toml pairs that allow nothing.
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
        out.extend(cut(path, &text, s.similar_window, s.similar_spacing));
    }
    Ok(out)
}

/// Embeds uncached windows, then returns every embedding in order.
fn embed(model: &Path, wins: &[Window], cache: &mut Cache) -> Result<Vec<Vec<f32>>> {
    let missing: Vec<&str> = wins
        .iter()
        .map(|w| w.text.as_str())
        .filter(|t| cache.get(key(t)).is_none())
        .collect();
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
        .map(|w| cache.get(key(&w.text)).expect("embedded").clone())
        .collect())
}

/// `base` names the ref new files are counted from.
fn check(
    model: &Path,
    cache_file: Option<&Path>,
    json: Option<&Path>,
    base: Option<&str>,
) -> Result<bool> {
    let s: Settings = toml::from_str(include_str!("../../../lint.toml"))?;
    let allow = Allow::parse(include_str!("../allow.toml"))?;
    let root = PathBuf::from(git(Path::new("."), &["rev-parse", "--show-toplevel"])?.trim());
    let wins = all_windows(&root, &s)?;
    let mut cache = match cache_file {
        Some(f) => Cache::load(f)?,
        None => Cache::default(),
    };
    let embs = embed(model, &wins, &mut cache)?;
    if let Some(f) = cache_file {
        let keys: Vec<u64> = wins.iter().map(|w| key(&w.text)).collect();
        cache.save(f, &keys)?;
    }
    let (hits, used) = near(&wins, &embs, s.similar_window, s.max_similarity, &allow);
    let stale = allow.stale(&used);
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
        println!("tools/simcheck/allow.toml: stale: {a} and {b} are not near-copies");
    }
    if let Some(path) = json {
        let limits = (s.similar_window, s.max_similarity);
        let r = report::report((&hits, &stale, &leans), &wins, &root, limits)?;
        r.write(path).map_err(anyhow::Error::msg)?;
    }
    Ok(hits.is_empty() && stale.is_empty())
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

/// Embeds the tree. `keep_all` keeps cache entries of other trees.
fn tree(model: &Path, cache_file: Option<&Path>, keep_all: bool) -> Result<Tree> {
    let s: Settings = toml::from_str(include_str!("../../../lint.toml"))?;
    let root = PathBuf::from(git(Path::new("."), &["rev-parse", "--show-toplevel"])?.trim());
    let wins = all_windows(&root, &s)?;
    let mut cache = match cache_file {
        Some(f) => Cache::load(f)?,
        None => Cache::default(),
    };
    let embs = embed(model, &wins, &mut cache)?;
    if let Some(f) = cache_file {
        let mut keys: Vec<u64> = wins.iter().map(|w| key(&w.text)).collect();
        if keep_all {
            keys.extend(cache.keys());
        }
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
    let (wins, embs, crates) = tree(model, Some(cache_file), true)?;
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
    let (wins, embs, crates) = tree(model, cache_file, false)?;
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

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let run = match args[..] {
        ["fetch", dir] => fetch::fetch(Path::new(dir)).map(|()| true),
        ["check", dir] => check(Path::new(dir), None, None, None),
        ["check", dir, cache] => check(Path::new(dir), Some(Path::new(cache)), None, None),
        ["check", dir, cache, json] => check(
            Path::new(dir),
            Some(Path::new(cache)),
            Some(Path::new(json)),
            None,
        ),
        ["check", dir, cache, json, base] => check(
            Path::new(dir),
            Some(Path::new(cache)),
            Some(Path::new(json)),
            Some(base),
        ),
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
