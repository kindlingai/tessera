use super::expand;

const INCLUDES: &[(&str, &str)] = &[
    ("outer.metal", "#include \"inner.metal\"\nfloat outer();\n"),
    ("inner.metal", "float inner();\n"),
];

#[test]
fn emits_line_directives() {
    let s = expand("line1\n#include \"outer.metal\"\nline3\n", INCLUDES);
    assert!(
        s.contains("#line 1 \"kernel.metal\""),
        "entry #line missing"
    );
    assert!(
        s.contains("#line 1 \"include/outer.metal\""),
        "include #line missing"
    );
    assert!(
        s.contains("#line 1 \"include/inner.metal\""),
        "nested #line missing"
    );
    // The include sits on line 2, so the entry file resumes at line 3.
    assert!(
        s.contains("#line 3 \"kernel.metal\""),
        "resume #line missing:\n{s}"
    );
    assert!(s.contains("float outer();"));
    assert!(s.contains("float inner();"));
}

#[test]
fn keeps_system_includes() {
    let s = expand("#include <metal_stdlib>\nkernel void k() {}\n", INCLUDES);
    assert!(s.contains("#include <metal_stdlib>"));
    assert!(s.contains("kernel void k() {}"));
}

#[test]
#[should_panic(expected = "unknown include")]
fn unknown_include_panics() {
    expand("#include \"missing.metal\"\n", INCLUDES);
}
