//! Reads the comment `tools/scripts/conflict-notice.sh` writes.

/// The first line of that comment.
pub(crate) const MARKER: &str = "<!-- conflict-notice -->";
/// The bot that writes the comment.
pub(crate) const AUTHOR: &str = "github-actions";
/// A list of pull requests follows each of these lines.
const HEADS: [(&str, Kind); 2] = [
    ("These open PRs conflict with this PR", Kind::Conflict),
    ("These open PRs change the same files", Kind::Shared),
];

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Kind {
    /// The two pull requests do not merge together.
    Conflict,
    /// Both change the file, and they merge.
    Shared,
}

pub struct Overlap {
    pub kind: Kind,
    pub other: u64,
    pub files: Vec<String>,
}

/// Reads list lines such as ``- #12: `a.rs` `b.rs` ``.
pub fn read(body: &str) -> Vec<Overlap> {
    let mut found = Vec::new();
    let mut kind = None;
    for line in body.lines() {
        for (head, listed) in HEADS {
            if line.starts_with(head) {
                kind = Some(listed);
            }
        }
        let item = line.strip_prefix("- #").and_then(|l| l.split_once(": "));
        let (Some(kind), Some((other, files))) = (kind, item) else {
            continue;
        };
        let Ok(other) = other.parse() else {
            continue;
        };
        let files = files.split('`').skip(1).step_by(2).map(String::from);
        found.push(Overlap {
            kind,
            other,
            files: files.collect(),
        });
    }
    found
}
