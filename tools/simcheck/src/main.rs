//! Fails on near-copies of code, using EmbeddingGemma 2 embeddings.

mod cache;
mod fetch;
mod pairs;
mod report;
mod windows;

#[cfg(test)]
mod cache_test;
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
       tessera-simcheck check <model_dir> [cache_file [report_file [link_base]]]

`check` links report hunks under <link_base> when given.
It also fails on allow.toml pairs that allow nothing.";

/// The repository's lint.toml keys this tool reads.
#[derive(serde::Deserialize)]
struct Settings {
    similar_window: usize,
    similar_spacing: usize,
    max_similarity: f32,
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

fn check(
    model: &Path,
    cache_file: Option<&Path>,
    report: Option<&Path>,
    links: Option<&str>,
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
    if let Some(r) = report {
        let limits = (s.similar_window, s.max_similarity);
        let md = report::markdown(&hits, &stale, &wins, &root, limits, links)?;
        std::fs::write(r, md)?;
    }
    Ok(hits.is_empty() && stale.is_empty())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    let run = match args[..] {
        ["fetch", dir] => fetch::fetch(Path::new(dir)).map(|()| true),
        ["check", dir] => check(Path::new(dir), None, None, None),
        ["check", dir, cache] => check(Path::new(dir), Some(Path::new(cache)), None, None),
        ["check", dir, cache, report] => check(
            Path::new(dir),
            Some(Path::new(cache)),
            Some(Path::new(report)),
            None,
        ),
        ["check", dir, cache, report, links] => check(
            Path::new(dir),
            Some(Path::new(cache)),
            Some(Path::new(report)),
            Some(links),
        ),
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
