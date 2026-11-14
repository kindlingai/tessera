//! The PR comment: each broken rule, its fix, and where.

use crate::rules::Rule;
use crate::tags::Hit;
use std::fmt::Write;

/// CI finds the comment to edit by this first line.
pub const MARKER: &str = "<!-- tessera-style -->";
/// Keeps the comment under GitHub's 65,536-character limit.
const MAX_PER_RULE: usize = 15;

/// A hit, its file and its line of text.
pub struct Found {
    pub path: String,
    pub hit: Hit,
    pub text: String,
}

fn code(text: &str) -> String {
    match text.contains('`') {
        true => format!("`` {text} ``"),
        false => format!("`{text}`"),
    }
}

fn place(f: &Found, links: Option<&str>) -> String {
    let at = format!("`{}:{}`", f.path, f.hit.line);
    match links {
        Some(base) => format!("[{at}]({base}/{}#L{})", f.path, f.hit.line),
        None => at,
    }
}

/// A rule's fix, then each hit.
fn section(
    out: &mut String,
    rule: &Rule,
    rows: &[&Found],
    links: Option<&str>,
) -> std::fmt::Result {
    writeln!(
        out,
        "\n<details open><summary>{} ({})</summary>\n",
        rule.name,
        rows.len()
    )?;
    let doc = format!("tools/style/rules/{}/rule.md", rule.id);
    let see = links.map_or(format!("`{doc}`"), |b| format!("[`{doc}`]({b}/{doc})"));
    let why = Some(rule.guidance())
        .filter(|g| !g.is_empty())
        .unwrap_or(rule.description.clone());
    writeln!(out, "{why} More in {see}.")?;
    writeln!(out)?;
    for f in rows.iter().take(MAX_PER_RULE) {
        let text = f.text.trim();
        writeln!(
            out,
            "- {}: {} (matched {})",
            place(f, links),
            code(text),
            code(&f.hit.found)
        )?;
    }
    if rows.len() > MAX_PER_RULE {
        writeln!(out, "- {} more not shown.", rows.len() - MAX_PER_RULE)?;
    }
    writeln!(out, "\n</details>")
}

pub fn markdown(
    found: &[Found],
    rules: &[Rule],
    links: Option<&str>,
) -> Result<String, std::fmt::Error> {
    let mut out = format!("{MARKER}\n### Comment style\n\n");
    if found.is_empty() {
        out.push_str("None found.\n");
        return Ok(out);
    }
    writeln!(
        out,
        "{} added lines break the rules in `tools/style/rules`.",
        found.len()
    )?;
    writeln!(out, "This check does not fail CI.")?;
    for rule in rules {
        let mine: Vec<&Found> = found.iter().filter(|f| f.hit.rule == rule.id).collect();
        if !mine.is_empty() {
            section(&mut out, rule, &mine, links)?;
        }
    }
    Ok(out)
}
