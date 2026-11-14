//! Prose in a hunk: comments, Markdown lines, and ignore markers.

use fancy_regex::Regex;

/// Files whose comments start with `#`. Others use `//`.
const HASH_EXTS: [&str; 12] = [
    "py", "sh", "bash", "zsh", "rb", "pl", "toml", "yaml", "yml", "nix", "cfg", "ini",
];
/// Files where every line is prose.
const PROSE_EXTS: [&str; 4] = ["md", "txt", "rst", ""];

pub fn extension(path: &str) -> &str {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.rsplit_once('.').map_or("", |(_, e)| e)
}

pub fn is_prose_file(path: &str) -> bool {
    PROSE_EXTS.contains(&extension(path))
}

/// A line's comment and marker. Prose files keep whole lines.
pub fn comment(path: &str, line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if is_prose_file(path) || trimmed.starts_with('*') || trimmed.starts_with("/*") {
        return Some(line.to_string());
    }
    let marker = if HASH_EXTS.contains(&extension(path)) {
        "#"
    } else {
        "//"
    };
    // A marker after whitespace only: "http://" and "a#b" are code.
    let at = line
        .match_indices(marker)
        .find(|(i, _)| *i == 0 || line[..*i].ends_with(char::is_whitespace))?;
    Some(line[at.0..].trim_start().to_string())
}

/// Marks `!` each line under a `prose: ignore [N]` marker.
pub fn apply_ignore(lines: &mut [String]) {
    let re = Regex::new(r"(?:<!--|//+|#+)\s*prose:\s*ignore(?:\s+(\d+))?\s*(?:-->)?\s*$")
        .expect("valid");
    let mut left = 0;
    for line in lines.iter_mut() {
        if line.starts_with('-') {
            continue;
        }
        if left > 0 {
            line.replace_range(..1, "!");
            left -= 1;
            continue;
        }
        let Ok(Some(caps)) = re.captures(&line[1..]) else {
            continue;
        };
        let n: usize = caps
            .get(1)
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(1);
        let own_line = line[1..caps.get(0).map_or(0, |m| m.start())]
            .trim()
            .is_empty();
        line.replace_range(..1, "!");
        left = if own_line { n } else { n - 1 };
    }
}

/// Blanks code blocks, and turns quoted and backticked spans into `"…"`.
pub fn mask_markdown(lines: &mut [String]) {
    let (mut fence, mut indented, mut prev_blank, mut close) = (false, false, true, None);
    for raw in lines.iter_mut() {
        let (mark, line) = raw.split_at(1);
        let (mark, line) = (mark.to_string(), line.to_string());
        if mark == "-" {
            continue;
        }
        let blank = line.trim().is_empty();
        if blank {
            close = None;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            fence = !fence;
            *raw = mark;
        } else if fence
            || ((line.starts_with("    ") || line.starts_with('\t')) && (prev_blank || indented))
        {
            indented = indented || !fence;
            *raw = mark;
        } else {
            indented = indented && blank;
            let mut kept = mark;
            for ch in line.chars() {
                match close {
                    Some(c) if ch == c => close = None,
                    Some(_) => {}
                    None => match ch {
                        '`' | '"' => {
                            close = Some(ch);
                            kept.push_str("\"…\"");
                        }
                        '“' => {
                            close = Some('”');
                            kept.push_str("\"…\"");
                        }
                        c => kept.push(c),
                    },
                }
            }
            *raw = kept;
        }
        prev_blank = blank;
    }
}
