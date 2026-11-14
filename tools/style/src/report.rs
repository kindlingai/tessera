//! The PR comment: each broken rule, its fix, and where.

use crate::detect::{Hit, Level};
use crate::rules::Rule;
use std::fmt::Write;

/// CI finds the comment to edit by this first line.
pub const MARKER: &str = "<!-- tessera-style -->";
/// Keeps the comment under GitHub's 65,536-character limit.
const MAX_PER_RULE: usize = 15;

/// A file's hits, with each hit's line of text.
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

/// One section per rule, in rule order.
fn section(
    out: &mut String,
    rule: Option<&Rule>,
    id: &str,
    rows: &[&Found],
    links: Option<&str>,
) -> std::fmt::Result {
    let name = rule.map_or(id, |r| r.name.as_str());
    writeln!(
        out,
        "\n<details open><summary>{name} ({})</summary>\n",
        rows.len()
    )?;
    if let Some(r) = rule {
        let doc = format!("tools/style/rules/{id}/rule.md");
        let see = links.map_or(format!("`{doc}`"), |b| format!("[`{doc}`]({b}/{doc})"));
        let why = Some(r.guidance())
            .filter(|g| !g.is_empty())
            .unwrap_or(r.description.clone());
        writeln!(out, "{why} More in {see}.")?;
        if let Some((bad, good)) = r.example() {
            writeln!(
                out,
                "\n````text\n{bad}\n````\nbecomes\n````text\n{good}\n````"
            )?;
        }
    }
    writeln!(out)?;
    for f in rows.iter().take(MAX_PER_RULE) {
        let text = f.text.trim();
        let seen = match f.hit.found.trim() == text {
            true => String::new(),
            false => format!(" (matched {})", code(&f.hit.found)),
        };
        writeln!(out, "- {}: {}{seen}", place(f, links), code(text))?;
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
    tagged: bool,
) -> Result<String, std::fmt::Error> {
    let mut out = format!("{MARKER}\n### Comment style\n\n");
    if found.is_empty() {
        out.push_str("None found.\n");
        return Ok(out);
    }
    let fails = found.iter().filter(|f| f.hit.level == Level::Fail).count();
    match fails {
        0 => writeln!(
            out,
            "No added line breaks a rule in `tools/style/rules`. Leads follow."
        )?,
        n => writeln!(
            out,
            "{n} added lines break the rules in `tools/style/rules`."
        )?,
    }
    writeln!(out, "This check does not fail CI.")?;
    if !tagged {
        writeln!(
            out,
            "\nThe spaCy pipeline was missing, so rules that read sentence structure did not run."
        )?;
    }
    for level in [Level::Fail, Level::Lead] {
        let rows: Vec<&Found> = found.iter().filter(|f| f.hit.level == level).collect();
        if level == Level::Lead && !rows.is_empty() {
            writeln!(out, "\n#### Leads\n\nThese may be fine. A human decides.")?;
        }
        let mut ids: Vec<&str> = rows.iter().map(|f| f.hit.rule.as_str()).collect();
        ids.dedup();
        let order = |id: &&str| rules.iter().position(|r| r.id == *id).unwrap_or(usize::MAX);
        ids.sort_by_key(order);
        ids.dedup();
        for id in ids {
            let mine: Vec<&Found> = rows.iter().copied().filter(|f| f.hit.rule == id).collect();
            section(
                &mut out,
                rules.iter().find(|r| r.id == id),
                id,
                &mine,
                links,
            )?;
        }
    }
    Ok(out)
}
