//! Kernel entry-point names in Metal and CUDA sources.

use crate::files::Source;
use crate::{MAX_NAME_SEGMENTS, Violation, comments, segments};

/// Words before a kernel entry point's name.
const MARKERS: &[&str] = &["kernel", "__global__"];

fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// The identifier right before the first `(` after `at`.
fn name_after(code: &[u8], at: usize) -> Option<(usize, String)> {
    let open = at + code[at..].iter().position(|&c| c == b'(')?;
    let end = (at..open).rev().find(|&i| is_ident(code[i]))? + 1;
    let start = (at..end).rev().take_while(|&i| is_ident(code[i])).last()?;
    let name = String::from_utf8_lossy(&code[start..end]).into_owned();
    Some((start, name))
}

pub fn check(src: &Source, out: &mut Vec<Violation>) {
    let code = comments::strip(&src.text, false);
    let b = code.as_bytes();
    for marker in MARKERS {
        let mut from = 0;
        while let Some(off) = code[from..].find(marker) {
            let at = from + off;
            from = at + marker.len();
            let whole =
                (at == 0 || !is_ident(b[at - 1])) && !is_ident(*b.get(from).unwrap_or(&b' '));
            if !whole {
                continue;
            }
            let Some((pos, name)) = name_after(b, from) else {
                continue;
            };
            if segments(&name) > MAX_NAME_SEGMENTS {
                let line = code[..pos].matches('\n').count() + 1;
                out.push(src.violation(
                    line,
                    "name-segments",
                    format!(
                        "kernel `{name}` has {} segments (max {MAX_NAME_SEGMENTS})",
                        segments(&name)
                    ),
                ));
            }
        }
    }
}
