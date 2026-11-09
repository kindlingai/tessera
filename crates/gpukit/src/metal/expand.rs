//! Runtime `#include "x.metal"` expansion for shader sources.
//!
//! The Metal runtime compiler takes one source string, so shared headers are
//! spliced in before compilation — string work measured in microseconds
//! against a millisecond Metal compile. Emits `#line` directives so compile
//! errors map back to the original files (the entry file is labeled
//! `kernel.metal`; its line numbers match the original source).
//!
//! Only quoted local includes (`#include "name.metal"`) are expanded; system
//! includes (`#include <metal_stdlib>`) pass through to the Metal compiler
//! untouched.

/// `#include "name.metal"` → `Some(name)`; anything else (system includes,
/// code) → `None`.
fn parse_local_include(line: &str) -> Option<&str> {
    let rest = line.trim().strip_prefix("#include \"")?;
    let name = &rest[..rest.find('"')?];
    name.ends_with(".metal").then_some(name)
}

/// Recursively expand quoted includes against the caller's header table.
///
/// Panics on an include name absent from `includes` — a missing header is a
/// build defect, not a runtime condition.
pub fn expand(source: &str, includes: &[(&str, &str)]) -> String {
    expand_labeled("kernel.metal", source, includes)
}

fn expand_labeled(label: &str, source: &str, includes: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(source.len() * 2);
    out.push_str(&format!("#line 1 \"{label}\"\n"));
    for (line_num, line) in (1_u32..).zip(source.split_inclusive('\n')) {
        if let Some(name) = parse_local_include(line) {
            let content = includes
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, s)| *s)
                .unwrap_or_else(|| {
                    panic!("expand: unknown include {name:?} (missing from the include table)")
                });
            let resume = line_num + 1;
            out.push_str(&expand_labeled(
                &format!("include/{name}"),
                content,
                includes,
            ));
            out.push_str(&format!("#line {resume} \"{label}\"\n"));
        } else {
            out.push_str(line);
        }
    }
    out
}
