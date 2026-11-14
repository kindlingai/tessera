//! One PR comment for every check's report.

use findings::{Finding, Level, Report};
use std::fmt::Write;

/// CI finds the comment to edit by this first line.
pub const MARKER: &str = "<!-- tessera-checks -->";
/// Keeps the comment under GitHub's 65,536-character limit.
const MAX_PER_RULE: usize = 15;

/// A code fence's language, from a file's extension.
fn language(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("rs") => "rust",
        Some("cu" | "cuh" | "hip") => "cuda",
        Some("metal" | "h" | "hpp" | "c" | "cc" | "cpp") => "cpp",
        Some("toml") => "toml",
        _ => "",
    }
}

fn place(path: &str, line: usize, links: Option<&str>) -> String {
    let at = format!("`{path}:{line}`");
    match links {
        Some(base) => format!("[{at}]({base}/{path}#L{line})"),
        None => at,
    }
}

fn status(r: &Report) -> String {
    let notes = r.findings.len() - r.fails();
    match (&r.skipped, r.fails(), notes) {
        (Some(why), _, _) => format!("skipped: {why}"),
        (None, 0, 0) => "none found".into(),
        (None, f, 0) => format!("{f} failed"),
        (None, 0, n) => format!("{n} notes"),
        (None, f, n) => format!("{f} failed, {n} notes"),
    }
}

fn finding(out: &mut String, f: &Finding, links: Option<&str>) -> std::fmt::Result {
    writeln!(out, "- {}: {}", place(&f.path, f.line, links), f.message)?;
    for s in &f.snippets {
        let at = place(&s.path, s.line, links);
        writeln!(
            out,
            "\n  {at}\n\n  ````{}\n{}\n  ````",
            language(&s.path),
            s.text
        )?;
    }
    if let Some(fix) = &f.fix {
        writeln!(out, "\n  ````{}\n  {fix}\n  ````", language(&f.path))?;
    }
    Ok(())
}

/// A report's findings by rule, each rule with its fix.
fn section(out: &mut String, r: &Report, links: Option<&str>) -> std::fmt::Result {
    writeln!(
        out,
        "\n<details open><summary>{}: {}</summary>",
        r.title,
        status(r)
    )?;
    let mut ids: Vec<&str> = r.findings.iter().map(|f| f.rule.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    for id in ids {
        let rows: Vec<&Finding> = r.findings.iter().filter(|f| f.rule == id).collect();
        let name = r.rules.get(id).map_or(id, |rule| rule.name.as_str());
        let note = if rows.iter().all(|f| f.level == Level::Note) {
            " (advice only)"
        } else {
            ""
        };
        writeln!(out, "\n#### {name}{note}\n")?;
        if let Some(rule) = r.rules.get(id) {
            writeln!(out, "{}\n", rule.fix)?;
        }
        for f in rows.iter().take(MAX_PER_RULE) {
            finding(out, f, links)?;
        }
        if rows.len() > MAX_PER_RULE {
            writeln!(out, "- {} more not shown.", rows.len() - MAX_PER_RULE)?;
        }
    }
    writeln!(out, "\n</details>")
}

pub fn render(reports: &[Report], links: Option<&str>) -> Result<String, std::fmt::Error> {
    let mut out = format!("{MARKER}\n### Checks\n\n");
    if reports.iter().all(|r| r.findings.is_empty()) {
        out.push_str("None found.\n\n");
    }
    writeln!(out, "| Check | Result |\n|---|---|")?;
    for r in reports {
        writeln!(out, "| {} | {} |", r.title, status(r))?;
    }
    for r in reports.iter().filter(|r| !r.findings.is_empty()) {
        section(&mut out, r, links)?;
    }
    let fails: usize = reports.iter().map(Report::fails).sum();
    if fails > 0 {
        writeln!(
            out,
            "\nFix each failure. A false style hit can take `prose: ignore` above it."
        )?;
    }
    Ok(out)
}
