use super::expand;

const INCLUDES: &[(&str, &str)] = &[
    ("outer.metal", "#include \"inner.metal\"\nfloat outer();\n"),
    ("inner.metal", "float inner();\n"),
];

/// Sources and text each expansion must contain.
const CASES: &[(&str, &[&str])] = &[
    (
        "line1\n#include \"outer.metal\"\nline3\n",
        &[
            "#line 1 \"kernel.metal\"",
            "#line 1 \"include/outer.metal\"",
            "#line 1 \"include/inner.metal\"",
            // The include is line 2, so line 3 resumes.
            "#line 3 \"kernel.metal\"",
            "float outer();",
            "float inner();",
        ],
    ),
    (
        "#include <metal_stdlib>\nkernel void k() {}\n",
        &["#include <metal_stdlib>", "kernel void k() {}"],
    ),
];

fn contains_all(&(source, wants): &(&str, &[&str])) {
    let s = expand(source, INCLUDES);
    for want in wants {
        assert!(s.contains(want), "{want} missing from:\n{s}");
    }
}

#[test]
fn includes_expand() {
    CASES.iter().for_each(contains_all);
}

#[test]
#[should_panic(expected = "unknown include")]
fn unknown_include_panics() {
    expand("#include \"missing.metal\"\n", INCLUDES);
}
