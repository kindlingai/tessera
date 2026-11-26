//! Reads the comment `tools/scripts/conflict-notice.sh` writes.

/// The first line of that comment.
pub(crate) const MARKER: &str = "<!-- conflict-notice -->";
/// The bot that writes the comment.
pub(crate) const AUTHOR: &str = "github-actions";

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
    /// Untrusted file names, not escaped.
    pub files: Vec<String>,
}

/// Reads lines such as ``- conflict #12: `a.rs` `b.rs` ``.
pub fn read(body: &str) -> Vec<Overlap> {
    let mut found = Vec::new();
    for line in body.lines() {
        let Some((head, files)) = line.split_once(": ") else {
            continue;
        };
        let (kind, other) = match head.split_once(" #") {
            Some(("- conflict", other)) => (Kind::Conflict, other),
            Some(("- shared", other)) => (Kind::Shared, other),
            _ => continue,
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
