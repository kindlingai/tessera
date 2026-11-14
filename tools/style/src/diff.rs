//! Unified diffs into hunks of marked lines.

/// Lines marked `+` added, `-` removed, ` ` context or `!` ignored.
pub struct Hunk {
    pub path: String,
    /// The new file's line number of the first non-removed line.
    pub first: usize,
    pub lines: Vec<String>,
}

impl Hunk {
    /// A whole file as one hunk of added lines.
    pub fn file(path: &str, text: &str) -> Hunk {
        Hunk {
            path: path.to_string(),
            first: 1,
            lines: text.lines().map(|l| format!("+{l}")).collect(),
        }
    }

    /// Each non-removed line with its number.
    pub fn numbered(&self) -> impl Iterator<Item = (usize, &str)> {
        let kept = self.lines.iter().filter(|l| !l.starts_with('-'));
        kept.enumerate().map(|(i, l)| (self.first + i, l.as_str()))
    }
}

/// The new start line in `@@ -a,b +c,d @@`.
fn new_start(header: &str) -> Option<usize> {
    let plus = header.split_whitespace().find(|w| w.starts_with('+'))?;
    plus[1..].split(',').next()?.parse().ok()
}

/// Hunks that add at least one line.
pub fn parse(text: &str) -> Vec<Hunk> {
    let mut out: Vec<Hunk> = Vec::new();
    let mut path: Option<String> = None;
    let lines: Vec<&str> = text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        // A "--- a/x" header would read as a removed line.
        if line.starts_with("--- ") && lines.get(i + 1).is_some_and(|n| n.starts_with("+++ ")) {
            continue;
        }
        if let Some(p) = line.strip_prefix("+++ ") {
            path = Some(p.strip_prefix("b/").unwrap_or(p).to_string());
        } else if line.starts_with("@@") {
            if let (Some(p), Some(first)) = (&path, new_start(line)) {
                out.push(Hunk {
                    path: p.clone(),
                    first,
                    lines: Vec::new(),
                });
            }
        } else if path.is_some()
            && matches!(line.chars().next(), Some('+' | '-' | ' '))
            && let Some(h) = out.last_mut()
        {
            h.lines.push(line.to_string());
        }
    }
    out.retain(|h| h.lines.iter().any(|l| l.starts_with('+')));
    out
}
