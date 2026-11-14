//! Rules over part-of-speech tags from spaCy's tagger.

use crate::diff::Hunk;
use crate::pattern;
use crate::prose::{comment, is_prose_file};
use bundle::tag::Tagger;
use bundle::tokenize::Error;
use fancy_regex::Regex;
use std::path::Path;

/// A rule's hit: its line, rule id and matched words.
pub struct Hit {
    pub line: usize,
    pub rule: String,
    pub found: String,
}

pub struct Rule {
    id: String,
    re: Regex,
}

/// Rules from a file of `<rule>: <pattern>` lines.
pub fn load(path: &Path) -> Result<Vec<Rule>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for line in text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
    {
        let (id, pat) = line
            .split_once(": ")
            .ok_or_else(|| format!("no ': ' in {line:?}"))?;
        out.push(Rule {
            id: id.to_string(),
            re: pattern::compile(pat)?,
        });
    }
    Ok(out)
}

/// Added prose without markers, headings or tables. Code is `Foo`.
fn prose_lines(h: &Hunk) -> Vec<(usize, String)> {
    let marker = Regex::new(r"^\s*(//+!?|#+|/?\*+)\s?").expect("valid");
    let code = Regex::new(r"`[^`]*`").expect("valid");
    let markdown = is_prose_file(&h.path);
    let mut out = Vec::new();
    for (n, raw) in h.numbered() {
        let (mark, line) = raw.split_at(1);
        if mark != "+" || line.trim_start().starts_with('|') {
            continue;
        }
        let heading = markdown && line.trim_start().starts_with('#');
        let Some(text) = comment(&h.path, line).filter(|_| !heading) else {
            continue;
        };
        let text = marker.replace(&text, "");
        let text = code.replace_all(&text, "Foo").replace("\"…\"", "Foo");
        if !text.trim().is_empty() {
            out.push((n, text.trim().to_string()));
        }
    }
    out
}

/// Tag-rule hits on added prose, by comment block.
pub fn check(h: &Hunk, tagger: &Tagger, rules: &[Rule]) -> Result<Vec<Hit>, Error> {
    let lines = prose_lines(h);
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..=lines.len() {
        if i == lines.len() || lines[i].0 != lines[i - 1].0 + 1 {
            out.extend(block(&lines[start..i], tagger, rules)?);
            start = i;
        }
    }
    Ok(out)
}

/// Consecutive prose lines as one text, split into sentences.
fn block(lines: &[(usize, String)], tagger: &Tagger, rules: &[Rule]) -> Result<Vec<Hit>, Error> {
    let (mut text, mut starts) = (String::new(), Vec::new());
    for (n, line) in lines {
        starts.push((text.len(), *n));
        text.push_str(line);
        text.push(' ');
    }
    let line_of = |at: usize| {
        starts
            .iter()
            .rev()
            .find(|(s, _)| *s <= at)
            .map_or(0, |(_, n)| *n)
    };
    // Each sentence as `word/TAG ` tokens, with its first byte.
    let mut sentences: Vec<(usize, String)> = vec![(0, String::new())];
    for (w, tag) in tagger.tag(&text)? {
        if tag == "_SP" {
            continue;
        }
        let (start, s) = sentences.last_mut().expect("never empty");
        if s.is_empty() {
            *start = w.span.start;
        }
        s.push_str(&format!("{}/{tag} ", text[w.span].to_lowercase()));
        if tag == "." {
            sentences.push((0, String::new()));
        }
    }
    let mut out = Vec::new();
    for (start, s) in sentences.iter().filter(|(_, s)| !s.is_empty()) {
        for r in rules {
            if let Ok(Some(m)) = r.re.find(s) {
                let words = m
                    .as_str()
                    .split_whitespace()
                    .filter_map(|t| t.rsplit_once('/'));
                let found = words.map(|(w, _)| w).collect::<Vec<_>>().join(" ");
                out.push(Hit {
                    line: line_of(*start),
                    rule: r.id.clone(),
                    found,
                });
            }
        }
    }
    Ok(out)
}
