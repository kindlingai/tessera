//! Style rules for comments and prose, as part-of-speech patterns.

mod diff;
#[cfg(test)]
mod examples_test;
mod fetch;
mod pattern;
mod prose;
mod report;
mod rules;
mod tags;

use diff::Hunk;
use report::Found;
use std::path::Path;
use std::process::{Command, ExitCode};
use tessera_tasks::Tagger;

const USAGE: &str = "usage: tessera-style fetch <pipeline_dir>
       tessera-style diff <pipeline_dir> <base> <head> [json_file]
       tessera-style staged <pipeline_dir> [json_file]
       tessera-style message <pipeline_dir> <commit_msg_file> [json_file]
       tessera-style scan <pipeline_dir> <file>...

`diff` and `staged` check added lines. `message` checks a commit message.
Each command fails on any hit.";

/// This crate's rules and their patterns.
const RULES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/rules");
const TAGS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tags.txt");

struct Checker {
    rules: Vec<rules::Rule>,
    tagger: Tagger,
    patterns: Vec<tags::Rule>,
}

impl Checker {
    fn new(pipeline: &Path) -> Result<Self, String> {
        Ok(Self {
            rules: rules::load(Path::new(RULES))?,
            tagger: Tagger::open(pipeline, &tessera_frontend_settings::read())
                .map_err(|e| format!("{}: {e}", pipeline.display()))?,
            patterns: tags::load(Path::new(TAGS))?,
        })
    }

    fn hunk(&mut self, mut h: Hunk) -> Result<Vec<Found>, String> {
        // This tool's own rules quote the defects they describe.
        if h.path.starts_with("tools/style/") {
            return Ok(Vec::new());
        }
        prose::apply_ignore(&mut h.lines);
        if h.path.ends_with(".md") {
            prose::mask_markdown(&mut h.lines);
        }
        let hits = tags::check(&h, &mut self.tagger, &self.patterns).map_err(|e| e.to_string())?;
        let text_of = |n: usize| {
            h.numbered()
                .find(|(i, _)| *i == n)
                .map_or("", |(_, l)| &l[1..])
                .to_string()
        };
        let mut out: Vec<Found> = Vec::new();
        for hit in hits {
            if !out
                .iter()
                .any(|f| f.hit.line == hit.line && f.hit.rule == hit.rule)
            {
                out.push(Found {
                    path: h.path.clone(),
                    text: text_of(hit.line),
                    hit,
                });
            }
        }
        Ok(out)
    }
}

fn git(args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("git: {e}"))?;
    if !out.status.success() {
        return Err(format!("git {args:?} failed"));
    }
    String::from_utf8(out.stdout).map_err(|e| e.to_string())
}

fn print(found: &[Found]) {
    for f in found {
        println!("{}:{}: {}: {}", f.path, f.hit.line, f.hit.rule, f.hit.found);
    }
}

/// Hits on the hunks' added lines, also written to `json`.
fn check(dir: &str, hunks: Vec<Hunk>, json: Option<&&str>) -> Result<bool, String> {
    let mut checker = Checker::new(Path::new(dir))?;
    let mut found = Vec::new();
    for h in hunks {
        found.extend(checker.hunk(h)?);
    }
    print(&found);
    if let Some(path) = json {
        report::report(&found, &checker.rules).write(Path::new(path))?;
    }
    Ok(found.is_empty())
}

fn read(path: &str) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))
}

/// A commit message as one hunk, minus any diff.
fn message(path: &str) -> Result<Hunk, String> {
    let text = read(path)?;
    let kept: Vec<&str> = text.lines().take_while(|l| !l.contains(" >8 ")).collect();
    Ok(Hunk::file(path, &kept.join("\n")))
}

fn run(args: &[&str]) -> Result<bool, String> {
    match args {
        ["fetch", dir] => fetch::fetch(Path::new(dir)).map(|()| true),
        ["diff", dir, base, head, rest @ ..] if rest.len() <= 1 => {
            let diff = git(&["diff", "--no-color", &format!("{base}...{head}")])?;
            check(dir, diff::parse(&diff), rest.first())
        }
        ["staged", dir, rest @ ..] if rest.len() <= 1 => {
            let diff = git(&["diff", "--no-color", "--cached"])?;
            check(dir, diff::parse(&diff), rest.first())
        }
        ["message", dir, file, rest @ ..] if rest.len() <= 1 => {
            check(dir, vec![message(file)?], rest.first())
        }
        ["scan", dir, files @ ..] if !files.is_empty() => {
            let mut hunks = Vec::new();
            for path in files {
                hunks.push(Hunk::file(path, &read(path)?));
            }
            check(dir, hunks, None)
        }
        _ => Err(USAGE.into()),
    }
}

fn main() -> ExitCode {
    tessera_findings::main_of("tessera-style", run)
}
