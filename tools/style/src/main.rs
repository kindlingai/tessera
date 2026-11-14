//! Style rules for comments and prose: regexes and tag patterns.

mod detect;
mod diff;
#[cfg(test)]
mod examples_test;
mod fetch;
mod pattern;
mod report;
mod rules;
mod tags;
mod yaml;

use bundle::tag::Tagger;
use detect::{Detector, Level, apply_ignore, check, detectors, mask_markdown};
use diff::Hunk;
use report::Found;
use std::path::Path;
use std::process::{Command, ExitCode};

const USAGE: &str = "usage: tessera-style fetch <pipeline_dir>
       tessera-style diff <pipeline_dir> <base> <head> [report_file [link_base]]
       tessera-style scan <pipeline_dir> <file>...

A missing <pipeline_dir> skips the rules that read sentence structure.
`diff` checks added lines and always succeeds. `scan` fails on any failure.";

/// This crate's rules, and the patterns for parse rules.
const RULES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/rules");
const TAGS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tags.txt");

struct Checker {
    rules: Vec<rules::Rule>,
    detectors: Vec<Detector>,
    /// The tagger and its rules, when the pipeline is present.
    tags: Option<(Tagger, Vec<tags::Rule>)>,
}

impl Checker {
    fn new(pipeline: &Path) -> Result<Self, String> {
        let rules = rules::load(Path::new(RULES))?;
        let tags = match pipeline.join("tokenizer").exists() {
            true => Some((
                Tagger::open(pipeline).map_err(|e| e.to_string())?,
                tags::load(Path::new(TAGS))?,
            )),
            false => None,
        };
        // Tag patterns replace the detectors of rules that read structure.
        let detectors = detectors(rules.iter().filter(|r| !(r.parse && tags.is_some())))?;
        Ok(Self {
            rules,
            detectors,
            tags,
        })
    }

    fn hunk(&self, mut h: Hunk) -> Result<Vec<Found>, String> {
        // This tool's own rules quote the defects they describe.
        if h.path.starts_with("tools/style/") {
            return Ok(Vec::new());
        }
        apply_ignore(&mut h.lines);
        if h.path.ends_with(".md") {
            mask_markdown(&mut h.lines);
        }
        let mut hits = check(&h, &self.detectors);
        if let Some((tagger, rules)) = &self.tags {
            hits.extend(tags::check(&h, tagger, rules).map_err(|e| e.to_string())?);
        }
        let text_of = |n: usize| {
            h.numbered()
                .find(|(i, _)| *i == n)
                .map_or("", |(_, l)| &l[1..])
                .to_string()
        };
        let mut out: Vec<Found> = Vec::new();
        for hit in hits {
            let dup = out
                .iter()
                .any(|f| f.hit.line == hit.line && f.hit.rule == hit.rule);
            if !dup {
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
        let level = if f.hit.level == Level::Lead {
            "lead"
        } else {
            "fail"
        };
        println!(
            "{}:{}: {}: {level}: {}",
            f.path, f.hit.line, f.hit.rule, f.hit.found
        );
    }
}

fn run(args: &[&str]) -> Result<bool, String> {
    match args {
        ["fetch", dir] => fetch::fetch(Path::new(dir)).map(|()| true),
        ["diff", dir, base, head, rest @ ..] if rest.len() <= 2 => {
            let checker = Checker::new(Path::new(dir))?;
            let mut found = Vec::new();
            for h in diff::parse(&git(&["diff", "--no-color", &format!("{base}...{head}")])?) {
                found.extend(checker.hunk(h)?);
            }
            print(&found);
            if let Some(report) = rest.first() {
                let md = report::markdown(
                    &found,
                    &checker.rules,
                    rest.get(1).copied(),
                    checker.tags.is_some(),
                );
                std::fs::write(report, md.map_err(|e| e.to_string())?)
                    .map_err(|e| format!("{report}: {e}"))?;
            }
            Ok(true)
        }
        ["scan", dir, files @ ..] if !files.is_empty() => {
            let checker = Checker::new(Path::new(dir))?;
            let mut found = Vec::new();
            for path in files {
                let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
                found.extend(checker.hunk(Hunk::file(path, &text))?);
            }
            print(&found);
            Ok(!found.iter().any(|f| f.hit.level == Level::Fail))
        }
        _ => Err(USAGE.into()),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("tessera-style: {e}");
            ExitCode::from(2)
        }
    }
}
