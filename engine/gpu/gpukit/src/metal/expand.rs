//! Splices `#include "x.metal"` headers in; Metal compiles one string.

/// `#include "name.metal"` gives `Some(name)`; system includes pass.
fn parse_local_include(line: &str) -> Option<&str> {
    let rest = line.trim().strip_prefix("#include \"")?;
    let name = &rest[..rest.find('"')?];
    name.ends_with(".metal").then_some(name)
}

/// Expand quoted includes, with `#line` so errors name files.
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
                // A missing header is a build defect, so panic.
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
