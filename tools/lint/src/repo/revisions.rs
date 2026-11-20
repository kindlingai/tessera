//! CI cache key revisions match the fetchers' pinned revisions.

use std::path::Path;

use crate::Violation;
use crate::files::Source;

/// The workflow whose cache keys name model revisions.
const WORKFLOW: &str = ".github/workflows/ci.yml";
/// The constant a fetcher pins its revision in.
const PIN: &str = "const REVISION: &str = \"";

/// 40-character hex words in `line`.
fn hashes(line: &str) -> Vec<&str> {
    let words = line.split(|c: char| !c.is_ascii_hexdigit());
    words.filter(|w| w.len() == 40).collect()
}

pub fn check(root: &Path, sources: &[Source], out: &mut Vec<Violation>) {
    let mut pins = Vec::new();
    for src in sources {
        for (i, line) in src.text.lines().enumerate() {
            if let Some(at) = line.find(PIN) {
                let rev = line[at + PIN.len()..].split('"').next().unwrap_or_default();
                if hashes(rev) == [rev] {
                    pins.push((src, i + 1, rev.to_string()));
                }
            }
        }
    }
    let workflow = std::fs::read_to_string(root.join(WORKFLOW)).unwrap_or_default();
    let mut keyed = Vec::new();
    for (i, line) in workflow.lines().enumerate() {
        if !line.trim_start().starts_with("key:") {
            continue;
        }
        for rev in hashes(line) {
            keyed.push(rev.to_string());
            if !pins.iter().any(|(_, _, p)| p == rev) {
                out.push(Violation {
                    path: WORKFLOW.into(),
                    line: i + 1,
                    rule: "ci-revision",
                    message: format!(
                        "cache key revision {rev} differs from every `REVISION` constant"
                    ),
                });
            }
        }
    }
    for (src, line, rev) in &pins {
        if !keyed.contains(rev) {
            let message = format!("`REVISION` {rev} is in no `{WORKFLOW}` cache key");
            out.push(src.violation(*line, "ci-revision", message));
        }
    }
}
