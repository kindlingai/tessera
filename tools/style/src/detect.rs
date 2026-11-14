//! Each rule's regexes and literals, run on comments and prose.

use crate::diff::Hunk;
use crate::rules::Rule;
use fancy_regex::Regex;

/// One rule's hit: line, rule id, detector and matched text.
pub struct Hit {
    pub line: usize,
    pub rule: String,
    pub level: Level,
    pub found: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Fail,
    /// Worth a look. A human decides.
    Lead,
}

/// Files whose comments start with `#`. Others use `//`.
const HASH_EXTS: [&str; 12] = [
    "py", "sh", "bash", "zsh", "rb", "pl", "toml", "yaml", "yml", "nix", "cfg", "ini",
];
/// Files where every line is prose.
const PROSE_EXTS: [&str; 4] = ["md", "txt", "rst", ""];

pub struct Detector {
    rule: String,
    /// Reads prose files only.
    prose_only: bool,
    stampable: bool,
    test_docs_exempt: bool,
    re: Regex,
}

/// `re:` marks a regex. Other entries are case-insensitive literals.
fn compile(entry: &str) -> Result<Regex, Box<fancy_regex::Error>> {
    if let Some(re) = entry.strip_prefix("re:") {
        return Ok(Regex::new(re)?);
    }
    let alpha = entry.chars().all(char::is_alphabetic);
    let pattern = match entry.strip_suffix('e') {
        // One word also takes -s, -ed and -ing.
        Some(stem) if alpha => format!(r"(?i)\b{stem}(?:e|es|ed|ing)\b"),
        _ if alpha => format!(r"(?i)\b{entry}(?:s|es|ed|ing)?\b"),
        _ => {
            let starts = entry.starts_with(char::is_alphanumeric);
            let ends = entry.ends_with(char::is_alphanumeric);
            let edge = |on: bool| if on { r"\b" } else { "" };
            format!(
                "(?i){}{}{}",
                edge(starts),
                fancy_regex::escape(entry),
                edge(ends)
            )
        }
    };
    Ok(Regex::new(&pattern)?)
}

pub fn detectors<'a>(rules: impl Iterator<Item = &'a Rule>) -> Result<Vec<Detector>, String> {
    let mut out = Vec::new();
    for rule in rules {
        let all = rule.detectors.iter().map(|d| (d, false));
        for (entry, prose_only) in all.chain(rule.prose_detectors.iter().map(|d| (d, true))) {
            let re = compile(entry).map_err(|e| format!("{}: {entry}: {e}", rule.id))?;
            out.push(Detector {
                rule: rule.id.clone(),
                prose_only,
                stampable: rule.stampable,
                test_docs_exempt: rule.ignore_test_docs,
                re,
            });
        }
    }
    Ok(out)
}

pub fn extension(path: &str) -> &str {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.rsplit_once('.').map_or("", |(_, e)| e)
}

pub fn is_prose_file(path: &str) -> bool {
    PROSE_EXTS.contains(&extension(path))
}

/// A line's comment and marker. Prose files keep whole lines.
pub fn comment(path: &str, line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if is_prose_file(path) || trimmed.starts_with('*') || trimmed.starts_with("/*") {
        return Some(line.to_string());
    }
    let marker = if HASH_EXTS.contains(&extension(path)) {
        "#"
    } else {
        "//"
    };
    // A marker after whitespace only: "http://" and "a#b" are code.
    let at = line
        .match_indices(marker)
        .find(|(i, _)| *i == 0 || line[..*i].ends_with(char::is_whitespace))?;
    Some(line[at.0..].trim_start().to_string())
}

/// Marks `!` each line under a `prose: ignore [N]` marker.
pub fn apply_ignore(lines: &mut [String]) {
    let re = Regex::new(r"(?:<!--|//+|#+)\s*prose:\s*ignore(?:\s+(\d+))?\s*(?:-->)?\s*$")
        .expect("valid");
    let mut left = 0;
    for line in lines.iter_mut() {
        if line.starts_with('-') {
            continue;
        }
        if left > 0 {
            line.replace_range(..1, "!");
            left -= 1;
            continue;
        }
        let Ok(Some(caps)) = re.captures(&line[1..]) else {
            continue;
        };
        let n: usize = caps
            .get(1)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(1);
        let own_line = line[1..caps.get(0).map_or(0, |m| m.start())]
            .trim()
            .is_empty();
        line.replace_range(..1, "!");
        left = if own_line { n } else { n - 1 };
    }
}

/// Blanks code blocks, and turns quoted and backticked spans into `"…"`.
pub fn mask_markdown(lines: &mut [String]) {
    let (mut fence, mut indented, mut prev_blank, mut close) = (false, false, true, None);
    for raw in lines.iter_mut() {
        let (mark, line) = raw.split_at(1);
        let (mark, line) = (mark.to_string(), line.to_string());
        if mark == "-" {
            continue;
        }
        let blank = line.trim().is_empty();
        if blank {
            close = None;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            fence = !fence;
            *raw = mark;
        } else if fence
            || ((line.starts_with("    ") || line.starts_with('\t')) && (prev_blank || indented))
        {
            indented = indented || !fence;
            *raw = mark;
        } else {
            indented = indented && blank;
            let mut kept = mark;
            for ch in line.chars() {
                match close {
                    Some(c) if ch == c => close = None,
                    Some(_) => {}
                    None => match ch {
                        '`' | '"' => {
                            close = Some(ch);
                            kept.push_str("\"…\"");
                        }
                        '“' => {
                            close = Some('”');
                            kept.push_str("\"…\"");
                        }
                        c => kept.push(c),
                    },
                }
            }
            *raw = kept;
        }
        prev_blank = blank;
    }
}

/// Whether the context after the additions starts a test.
pub fn documents_test(h: &Hunk) -> bool {
    let test = Regex::new(
        r"^\s*#\[(\w+::)*test\b|^\s*(pub\s+)?(async\s+)?fn\s+test_|^\s*(async\s+)?def\s+test",
    )
    .expect("valid");
    let after = h.lines.iter().skip_while(|l| !l.starts_with('+'));
    let mut after = after;
    let next = after.find(|l| l.starts_with(' ') && !l[1..].trim().is_empty());
    next.is_some_and(|l| test.is_match(&l[1..]).unwrap_or(false))
}

/// Detector hits on a hunk's added lines.
pub fn check(h: &Hunk, detectors: &[Detector]) -> Vec<Hit> {
    let stamp = Regex::new(r"style-ok:\s*(\S)?").expect("valid");
    let stamp_line = Regex::new(r"^\s*(//+|#+|\*)\s*style-ok:").expect("valid");
    let leading = Regex::new(r"^\s*(//+!?|#+|/?\*+)\s?").expect("valid");
    let (prose_file, tests) = (is_prose_file(&h.path), documents_test(h));
    let mut out = Vec::new();
    let mut prev = "";
    for (n, raw) in h.numbered() {
        let (mark, line) = raw.split_at(1);
        let ignored = mark == "!";
        let Some(text) = (mark != " ").then(|| comment(&h.path, line)).flatten() else {
            prev = line;
            continue;
        };
        let (text, body) = match prose_file {
            true => (format!("// {line}"), line.to_string()),
            false => (text.clone(), leading.replace(&text, "").into_owned()),
        };
        let own = stamp.captures(line).ok().flatten();
        let above = stamp_line
            .is_match(prev)
            .unwrap_or(false)
            .then(|| stamp.captures(prev).ok().flatten())
            .flatten();
        let stamped = own.or(above);
        let reason = stamped.as_ref().is_some_and(|c| c.get(1).is_some());
        if stamped.is_some() && !reason && !ignored {
            out.push(Hit {
                line: n,
                rule: "stamp".into(),
                level: Level::Fail,
                found: "style-ok without a reason".into(),
            });
        }
        for d in detectors {
            let skip = (d.prose_only && !prose_file)
                || (ignored && d.stampable)
                || (tests && d.test_docs_exempt);
            if skip || (d.stampable && reason) {
                continue;
            }
            let m =
                d.re.find(&text)
                    .ok()
                    .flatten()
                    .or_else(|| d.re.find(&body).ok().flatten());
            if let Some(m) = m {
                out.push(Hit {
                    line: n,
                    rule: d.rule.clone(),
                    level: Level::Fail,
                    found: m.as_str().to_string(),
                });
            }
        }
        prev = line;
    }
    out
}
