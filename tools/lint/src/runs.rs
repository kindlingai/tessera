//! Runs of lines whose shapes repeat in too many places.

use std::collections::BTreeMap;

use crate::files::Source;
use crate::{LIMITS, Violation};
use shape::shape;

/// Shapes longer than this count as code.
const SUBSTANTIVE: usize = 3;

/// One file's non-blank lines, as shapes with their line numbers.
struct Shaped<'a> {
    src: &'a Source,
    ext: &'a str,
    line: Vec<usize>,
    shape: Vec<String>,
}

fn shaped(src: &Source) -> Shaped<'_> {
    let mut out = Shaped {
        src,
        ext: src.path.rsplit_once('.').map_or("", |(_, e)| e),
        line: Vec::new(),
        shape: Vec::new(),
    };
    let mut fields = false;
    for (i, l) in src.text.lines().enumerate() {
        let t = l.trim();
        if t.is_empty() {
            continue;
        }
        fields &= !t.starts_with('}');
        out.line.push(i + 1);
        out.shape.push(if fields { named(t) } else { shape(t) });
        fields |= t.ends_with('{') && (t.starts_with("struct ") || t.contains(" struct "));
    }
    out
}

/// A struct field keeps its name. Its type collapses.
fn named(line: &str) -> String {
    match line.split_once(':') {
        Some((name, ty)) if !ty.starts_with(':') && !line.starts_with(['/', '#']) => {
            format!("{name}:{}", shape(ty))
        }
        _ => shape(line),
    }
}

/// False for field lists, tables and runs of mostly braces.
fn worth(run: &[String]) -> bool {
    let code = run.iter().filter(|s| s.len() > SUBSTANTIVE).count();
    let repeats = run.windows(2).filter(|w| w[0] == w[1]).count();
    code * 2 >= run.len() && repeats * 2 < run.len()
}

/// Places by file and start. Overlaps in one file drop.
type Places = Vec<(usize, usize)>;

fn places<'a>(files: &'a [Shaped<'a>], len: usize) -> BTreeMap<(&'a str, &'a [String]), Places> {
    let mut found: BTreeMap<(&'a str, &'a [String]), Places> = BTreeMap::new();
    for (f, file) in files.iter().enumerate() {
        for s in 0..file.shape.len().saturating_sub(len - 1) {
            let run = &file.shape[s..s + len];
            if !worth(run) {
                continue;
            }
            let at = found.entry((file.ext, run)).or_default();
            if at.last().is_none_or(|&(lf, ls)| lf != f || s >= ls + len) {
                at.push((f, s));
            }
        }
    }
    found
}

/// Reported regions: file, start and length.
type Done = Vec<(usize, usize, usize)>;

/// Longest steps go first. Each region reports once.
pub fn check(sources: &[Source], out: &mut Vec<Violation>) {
    let files: Vec<Shaped> = sources.iter().map(shaped).collect();
    let mut steps = LIMITS.shape_steps.clone();
    steps.sort_by(|a, b| b.cmp(a));
    let mut done = Done::new();
    for (len, max) in steps {
        step(&files, len, max, &mut done, out);
    }
}

fn step(files: &[Shaped], len: usize, max: usize, done: &mut Done, out: &mut Vec<Violation>) {
    let mut hits: Vec<Places> = places(files, len)
        .into_values()
        .filter(|p| p.len() > max)
        .collect();
    hits.sort();
    for p in hits {
        // Shifted runs of one repeat: report the first only.
        let (f, s) = p[0];
        let near = |&(df, ds, dl): &(usize, usize, usize)| df == f && s < ds + dl && ds < s + len;
        if done.iter().any(near) {
            continue;
        }
        done.push((f, s, len));
        let at = |&(f, s): &(usize, usize)| format!("{}:{}", files[f].src.path, files[f].line[s]);
        let others: Vec<String> = p[1..].iter().map(at).collect();
        let msg = format!(
            "{len} lines share a shape with {} other places (max {max} in all): {}",
            others.len(),
            others.join(", ")
        );
        out.push(files[f].src.violation(files[f].line[s], "shape-runs", msg));
    }
}
