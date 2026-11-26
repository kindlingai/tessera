//! Conflict hunks in a file with diff3 markers.

/// Why a conflict stays, at a base branch line.
#[derive(Debug, PartialEq, Eq)]
pub struct Refusal {
    pub line: usize,
    pub why: &'static str,
}

/// The three versions of one conflicting region.
#[derive(Default)]
pub struct Hunk<'a> {
    /// First line of the region in the base branch's file.
    pub line: usize,
    pub ours: Vec<&'a str>,
    pub base: Vec<&'a str>,
    pub theirs: Vec<&'a str>,
}

pub enum Part<'a> {
    Line(&'a str),
    Hunk(Hunk<'a>),
}

enum Side {
    Ours,
    Base,
    Theirs,
}

/// Splits a file into its plain lines and its hunks.
pub fn parts(text: &str) -> Result<Vec<Part<'_>>, Refusal> {
    let mut out = Vec::new();
    let mut open: Option<(Hunk, Side)> = None;
    let mut line = 1;
    for raw in text.split_inclusive('\n') {
        let mark = raw.trim_end();
        let Some((hunk, side)) = &mut open else {
            if mark.starts_with("<<<<<<< ") {
                let hunk = Hunk {
                    line,
                    ..Hunk::default()
                };
                open = Some((hunk, Side::Ours));
            } else {
                out.push(Part::Line(raw));
                line += 1;
            }
            continue;
        };
        match side {
            Side::Ours if mark.starts_with("||||||| ") => *side = Side::Base,
            Side::Base if mark == "=======" => *side = Side::Theirs,
            Side::Theirs if mark.starts_with(">>>>>>> ") => {
                line += hunk.ours.len();
                out.push(Part::Hunk(std::mem::take(hunk)));
                open = None;
            }
            Side::Ours => hunk.ours.push(raw),
            Side::Base => hunk.base.push(raw),
            Side::Theirs => hunk.theirs.push(raw),
        }
    }
    match open {
        Some((hunk, _)) => Err(Refusal {
            line: hunk.line,
            why: "the conflict markers do not close",
        }),
        None => Ok(out),
    }
}
