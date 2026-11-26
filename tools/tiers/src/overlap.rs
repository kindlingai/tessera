//! Reads the comment the conflict-check workflow writes.

/// The first line of that comment.
pub(crate) const MARKER: &str = "<!-- pr-overlap -->";
/// The bot that writes the comment.
pub(crate) const AUTHOR: &str = "github-actions";

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Kind {
    /// The two pull requests do not merge together.
    Conflict,
    /// Both change the file.
    Shared,
}

pub struct Overlap {
    pub kind: Kind,
    pub other: u64,
    pub files: Vec<String>,
}

/// Reads lines such as ``- conflict #12: `a.rs`, `b.rs` ``.
pub fn read(body: &str) -> Vec<Overlap> {
    let mut found = Vec::new();
    if !body.starts_with(MARKER) {
        return found;
    }
    for line in body.lines() {
        let Some((head, files)) = line.split_once(": ") else {
            continue;
        };
        let kind = match head.split_once(" #") {
            Some(("- conflict", _)) => Kind::Conflict,
            Some(("- shared", _)) => Kind::Shared,
            _ => continue,
        };
        let Some(Ok(other)) = head.split('#').nth(1).map(str::parse) else {
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
