//! Near-copies, stale allow pairs and leaning files, as a report.

use crate::far::Lean;
use crate::pairs::Hit;
use crate::windows::Window;
use anyhow::Result;
use findings::{Finding, Level, Report, Rule, Snippet};
use std::path::Path;

const ALLOW: &str = "tools/simcheck/allow.toml";

/// A window's lines as written, indentation kept.
fn snippet(root: &Path, w: &Window, window: usize) -> Result<Snippet> {
    let text = std::fs::read_to_string(root.join(&w.path))?;
    let lines: Vec<&str> = text.lines().skip(w.line - 1).take(window).collect();
    Ok(Snippet {
        path: w.path.clone(),
        line: w.line,
        text: lines.join("\n"),
    })
}

fn rule(name: &str, fix: &str) -> Rule {
    Rule {
        name: name.into(),
        fix: fix.into(),
    }
}

/// Allow pairs that allow nothing.
type Stale<'a> = &'a [&'a (String, String)];

pub fn report(
    (hits, stale, leans): (&[Hit], Stale, &[Lean]),
    wins: &[Window],
    root: &Path,
    (window, min): (usize, f32),
) -> Result<Report> {
    let mut r = Report::new("simcheck", "Near-copies of code");
    let rules = [
        (
            "near-copy",
            rule(
                "Near-copy",
                &format!("Share the code, or add the pair to `{ALLOW}`."),
            ),
        ),
        (
            "stale-allow",
            rule(
                "Stale allow pair",
                &format!("Remove the pair from `{ALLOW}`."),
            ),
        ),
        (
            "new-file-lean",
            rule(
                "New file nearer another crate",
                "Think about moving the file.",
            ),
        ),
    ];
    r.rules = rules.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    for h in hits {
        let (a, b) = (&wins[h.a], &wins[h.b]);
        r.findings.push(Finding {
            rule: "near-copy".into(),
            level: Level::Fail,
            path: a.path.clone(),
            line: a.line,
            message: format!(
                "{:.3} similar to {}:{} ({} window pairs at {min} or more)",
                h.score, b.path, b.line, h.count
            ),
            snippets: vec![snippet(root, a, window)?, snippet(root, b, window)?],
            fix: Some(format!("[\"{}\", \"{}\"],", a.path, b.path)),
        });
    }
    let allow = std::fs::read_to_string(root.join(ALLOW)).unwrap_or_default();
    for (a, b) in stale {
        let at = allow
            .lines()
            .position(|l| l.contains(a.as_str()) && l.contains(b.as_str()));
        r.findings.push(Finding {
            rule: "stale-allow".into(),
            level: Level::Fail,
            path: ALLOW.into(),
            line: at.map_or(1, |i| i + 1),
            message: format!("{a} and {b} are not near-copies"),
            snippets: Vec::new(),
            fix: None,
        });
    }
    for l in leans {
        r.findings.push(Finding {
            rule: "new-file-lean".into(),
            level: Level::Note,
            path: l.path.clone(),
            line: 1,
            message: format!(
                "reads {:.3} like `{}`, {:.3} like its own `{}`",
                l.near, l.other, l.own, l.home
            ),
            snippets: Vec::new(),
            fix: None,
        });
    }
    Ok(r)
}
