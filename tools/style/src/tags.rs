//! Rules over part-of-speech tags from spaCy's tagger.

use crate::detect::{Hit, Level, comment, is_prose_file};
use crate::diff::Hunk;
use crate::pattern;
use bundle::tag::Tagger;
use bundle::tokenize::Error;
use fancy_regex::Regex;
use std::collections::BTreeSet;
use std::path::Path;

/// A sentence's tokens: lower-case text and tag.
type Sentence = Vec<(String, String)>;

pub struct Rule {
    id: String,
    level: Level,
    re: Regex,
}

/// Rules from a file of `<level> <rule>: <pattern>` lines.
pub fn load(path: &Path) -> Result<Vec<Rule>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for line in text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
    {
        let (head, pat) = line
            .split_once(": ")
            .ok_or_else(|| format!("no ': ' in {line:?}"))?;
        let (level, id) = head
            .split_once(' ')
            .ok_or_else(|| format!("no level in {line:?}"))?;
        let level = match level {
            "fail" => Level::Fail,
            "lead" => Level::Lead,
            other => return Err(format!("level {other:?} in {line:?}")),
        };
        out.push(Rule {
            id: id.to_string(),
            level,
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

fn encode(s: &Sentence) -> String {
    s.iter().map(|(w, t)| format!("{w}/{t} ")).collect()
}

/// Head nouns of one number. Modifiers precede other nouns.
fn heads(s: &Sentence, plural: bool) -> Vec<usize> {
    let noun = |t: &str| t.starts_with("NN");
    let is_plural = |t: &str| t == "NNS" || t == "NNPS";
    let mut out = Vec::new();
    for (i, (_, t)) in s.iter().enumerate() {
        let modifier = s.get(i + 1).is_some_and(|n| noun(&n.1));
        if noun(t) && is_plural(t) == plural && !modifier {
            out.push(i);
        }
    }
    out
}

fn first_verb(s: &Sentence) -> Option<usize> {
    s.iter()
        .position(|(_, t)| matches!(t.as_str(), "VBD" | "VBP" | "VBZ" | "MD"))
}

/// A pronoun whose noun is ambiguous in the previous sentence.
fn distant_pronoun(prev: &Sentence, s: &Sentence) -> Option<(Level, String)> {
    for (i, (w, t)) in s.iter().enumerate() {
        let plural = match w.as_str() {
            "it" | "its" => false,
            "they" | "them" | "their" => true,
            _ => continue,
        };
        // "It is safe to ..." names nothing.
        let dummy = w == "it"
            && s.get(i + 1).is_some_and(|n| n.0 == "is")
            && s.get(i + 2).is_some_and(|n| n.1.starts_with("JJ"));
        if !t.starts_with("PRP") || dummy || heads(s, plural).iter().any(|&j| j < i) {
            continue;
        }
        let before = heads(prev, plural);
        let distinct: BTreeSet<&str> = before.iter().map(|&j| prev[j].0.as_str()).collect();
        let Some(&nearest) = before.last() else {
            continue;
        };
        if distinct.len() < 2 || first_verb(prev).is_some_and(|v| nearest < v) {
            continue;
        }
        // A subject pronoun reads as the previous subject.
        let level = if first_verb(s).is_some_and(|v| i < v) {
            Level::Lead
        } else {
            Level::Fail
        };
        return Some((level, w.clone()));
    }
    None
}

/// Tag-rule hits on added prose, by comment block.
pub fn check(h: &Hunk, tagger: &Tagger, rules: &[Rule]) -> Result<Vec<Hit>, Error> {
    let lines = prose_lines(h);
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..=lines.len() {
        let gap = i == lines.len() || lines[i].0 != lines[i - 1].0 + 1;
        if gap {
            out.extend(block(&lines[start..i], tagger, rules)?);
            start = i;
        }
    }
    Ok(out)
}

/// Consecutive prose lines, read as one text.
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
    let mut sentences: Vec<(usize, Sentence)> = vec![(0, Vec::new())];
    for (w, tag) in tagger.tag(&text)? {
        if tag == "_SP" {
            continue;
        }
        let (start, s) = sentences.last_mut().expect("never empty");
        if s.is_empty() {
            *start = w.span.start;
        }
        s.push((text[w.span].to_lowercase(), tag.to_string()));
        if tag == "." {
            sentences.push((0, Vec::new()));
        }
    }
    sentences.retain(|(_, s)| !s.is_empty());
    let mut out = Vec::new();
    for (k, (start, s)) in sentences.iter().enumerate() {
        let line = line_of(*start);
        if k > 0
            && let Some((level, w)) = distant_pronoun(&sentences[k - 1].1, s)
        {
            out.push(Hit {
                line,
                rule: "distant-pronoun".into(),
                level,
                found: w,
            });
        }
        let enc = encode(s);
        for r in rules {
            if let Ok(Some(m)) = r.re.find(&enc) {
                let found = m
                    .as_str()
                    .split_whitespace()
                    .filter_map(|t| t.rsplit_once('/'))
                    .map(|(w, _)| w);
                out.push(Hit {
                    line,
                    rule: r.id.clone(),
                    level: r.level,
                    found: found.collect::<Vec<_>>().join(" "),
                });
            }
        }
    }
    Ok(out)
}
