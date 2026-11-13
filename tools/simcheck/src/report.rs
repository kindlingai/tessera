//! A Markdown report of near-copies, for a pull request comment.

use crate::far::Lean;
use crate::pairs::Hit;
use crate::windows::Window;
use anyhow::Result;
use std::fmt::Write;
use std::path::Path;

/// CI finds its comment to edit by this opening line.
pub const MARKER: &str = "<!-- tessera-simcheck -->";
/// Keeps the comment under GitHub's 65,536-character limit.
const MAX_HITS: usize = 20;

fn language(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("rs") => "rust",
        Some("cu" | "cuh" | "hip") => "cuda",
        Some("metal" | "h" | "hpp" | "c" | "cc" | "cpp") => "cpp",
        _ => "",
    }
}

/// A window's lines as written, indentation kept.
fn hunk(root: &Path, w: &Window, window: usize) -> Result<String> {
    let text = std::fs::read_to_string(root.join(&w.path))?;
    let lines: Vec<&str> = text.lines().skip(w.line - 1).take(window).collect();
    Ok(lines.join("\n"))
}

/// `path:a-b`, linked under `links` when given.
fn place(w: &Window, window: usize, links: Option<&str>) -> String {
    let (path, a, b) = (&w.path, w.line, w.line + window - 1);
    match links {
        Some(base) => format!("[`{path}:{a}-{b}`]({base}/{path}#L{a}-L{b})"),
        None => format!("`{path}:{a}-{b}`"),
    }
}

/// Each file pair's closest hunks, links and allow line.
fn near_copies(
    out: &mut String,
    hits: &[Hit],
    wins: &[Window],
    root: &Path,
    (window, min): (usize, f32),
    links: Option<&str>,
) -> Result<()> {
    writeln!(
        out,
        "{} file pairs hold {window}-line windows at or above {min} cosine similarity. \
         Share the code, or add the pair to `tools/simcheck/allow.toml`.\n",
        hits.len()
    )?;
    for h in hits.iter().take(MAX_HITS) {
        let (a, b) = (&wins[h.a], &wins[h.b]);
        writeln!(
            out,
            "<details open><summary>{:.3}: {} and {}</summary>\n",
            h.score, a.path, b.path
        )?;
        writeln!(out, "{} window pairs at or above {min}.\n", h.count)?;
        for w in [a, b] {
            let fence = format!("````{}", language(&w.path));
            writeln!(
                out,
                "{}\n\n{fence}\n{}\n````\n",
                place(w, window, links),
                hunk(root, w, window)?
            )?;
        }
        writeln!(
            out,
            "````toml\n[\"{}\", \"{}\"],\n````\n</details>\n",
            a.path, b.path
        )?;
    }
    if hits.len() > MAX_HITS {
        writeln!(out, "{} more pairs are not shown.\n", hits.len() - MAX_HITS)?;
    }
    Ok(())
}

/// New files that read more like another crate. Advice only.
fn leans(out: &mut String, leans: &[Lean]) -> Result<()> {
    if leans.is_empty() {
        return Ok(());
    }
    writeln!(out, "\n### New files nearer another crate\n")?;
    writeln!(
        out,
        "These new files read more like another crate than their own. \
         Think about moving them. This does not fail the check.\n"
    )?;
    writeln!(out, "| File | Own crate | Nearer crate |\n|---|---|---|")?;
    for l in leans {
        writeln!(
            out,
            "| `{}` | `{}` {:.3} | `{}` {:.3} |",
            l.path, l.home, l.own, l.other, l.near
        )?;
    }
    Ok(())
}

/// Allow pairs that allow nothing, as TOML lines to remove.
type Stale<'a> = &'a [&'a (String, String)];

/// Near-copies, stale allow pairs, then leaning new files.
pub fn markdown(
    (hits, stale, lean): (&[Hit], Stale, &[Lean]),
    wins: &[Window],
    root: &Path,
    limits: (usize, f32),
    links: Option<&str>,
) -> Result<String> {
    let mut out = format!("{MARKER}\n### Near-copies of code\n\n");
    if hits.is_empty() && stale.is_empty() {
        out.push_str("None found.\n");
        leans(&mut out, lean)?;
        return Ok(out);
    }
    if !hits.is_empty() {
        near_copies(&mut out, hits, wins, root, limits, links)?;
    }
    if !stale.is_empty() {
        out.push_str(
            "These `tools/simcheck/allow.toml` pairs allow nothing. Remove them.\n\n````toml\n",
        );
        for (a, b) in stale {
            writeln!(out, "[\"{a}\", \"{b}\"],")?;
        }
        out.push_str("````\n");
    }
    leans(&mut out, lean)?;
    Ok(out)
}
