//! Cutting source files into windows that start at content anchors.

use shape::shape;

/// Lines `line..line + window` of `path`, trimmed and joined.
pub struct Window {
    pub path: String,
    pub line: usize,
    pub text: String,
}

/// Shapes longer than this count as code.
const SUBSTANTIVE: usize = 3;

/// FNV-1a of a line's shape. Short shapes never anchor.
fn line_hash(line: &str) -> u64 {
    let s = shape(line);
    if s.len() <= SUBSTANTIVE {
        return u64::MAX;
    }
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Each run of `w` lines anchors its smallest hash.
fn winnow(hashes: &[u64], w: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for i in 0..=hashes.len().saturating_sub(w) {
        let run = &hashes[i..(i + w).min(hashes.len())];
        // Reversing makes ties pick the rightmost line.
        let Some((j, &h)) = run.iter().enumerate().rev().min_by_key(|&(_, h)| h) else {
            continue;
        };
        if h != u64::MAX && out.last() != Some(&(i + j)) {
            out.push(i + j);
        }
    }
    out
}

/// Windows at anchors about `spacing` apart, plus both file ends.
pub fn cut(path: &str, text: &str, window: usize, spacing: usize) -> Vec<Window> {
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let Some(last) = lines.len().checked_sub(window) else {
        return Vec::new();
    };
    let hashes: Vec<u64> = lines.iter().map(|l| line_hash(l)).collect();
    // Runs of 2s - 1 lines give anchors about s apart.
    let mut starts: Vec<usize> = winnow(&hashes, 2 * spacing.max(1) - 1);
    starts.retain(|&s| s <= last);
    starts.extend([0, last]);
    starts.sort_unstable();
    starts.dedup();
    starts
        .into_iter()
        .filter_map(|s| {
            let chunk = &lines[s..s + window];
            // Mostly blank lines or braces: nothing worth comparing.
            let code = chunk.iter().filter(|l| l.len() > SUBSTANTIVE).count();
            (code * 2 >= window).then(|| Window {
                path: path.to_string(),
                line: s + 1,
                text: chunk.join("\n"),
            })
        })
        .collect()
}

/// Whether two windows share lines of one file.
pub fn overlap(a: &Window, b: &Window, window: usize) -> bool {
    a.path == b.path && a.line.abs_diff(b.line) < window
}
