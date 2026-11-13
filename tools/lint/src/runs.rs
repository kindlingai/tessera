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
    for (i, l) in src.text.lines().enumerate() {
        if !l.trim().is_empty() {
            out.line.push(i + 1);
            out.shape.push(shape(l));
        }
    }
    out
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

/// Runs in over `max_shape_copies` places want a macro.
pub fn check(sources: &[Source], out: &mut Vec<Violation>) {
    let (len, max) = (LIMITS.shape_run, LIMITS.max_shape_copies);
    let files: Vec<Shaped> = sources.iter().map(shaped).collect();
    let mut hits: Vec<Places> = places(&files, len)
        .into_values()
        .filter(|p| p.len() > max)
        .collect();
    hits.sort();
    // Shifted runs of one repeat: report the first only.
    let mut last: Option<(usize, usize)> = None;
    for p in hits {
        let (f, s) = p[0];
        if last.is_some_and(|(lf, ls)| lf == f && s < ls + len) {
            continue;
        }
        last = Some((f, s));
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
