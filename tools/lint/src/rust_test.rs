use crate::files::Source;
use crate::rust::{check, is_test_file, test_lines};
use crate::{Violation, kernel, names};

fn source(path: &str, text: &str) -> Source {
    Source {
        path: path.into(),
        text: text.into(),
        lines: text.lines().count(),
    }
}

fn rules(path: &str, text: &str) -> Vec<&'static str> {
    let (mut out, mut found): (Vec<Violation>, _) = (Vec::new(), Vec::new());
    let src = source(path, text);
    if path.ends_with(".rs") {
        check(&src, &mut out, &mut found, &mut Vec::new());
    } else {
        kernel::check(&src, &mut found);
    }
    names::check(&[], &found, &mut out);
    out.iter().map(|v| v.rule).collect()
}

#[test]
fn long_names_fail() {
    assert_eq!(
        rules("a.rs", "fn one_two_three_four() {}\n"),
        ["name-segments"]
    );
    assert!(rules("a.rs", "fn one_two_three() {}\n").is_empty());
    let method = "struct S;\nimpl S { fn a_b_c_d(&self) {} }\n";
    assert_eq!(rules("a.rs", method), ["name-segments"]);
    let foreign = "impl Iterator for S { fn size_hint_of_it(&self) {} }\n";
    assert!(rules("a.rs", foreign).is_empty());
    let ours = "trait T { fn a_b_c_d(&self); }\n";
    assert_eq!(rules("a.rs", ours), ["name-segments"]);
}

#[test]
fn kernel_names_fail() {
    let metal = "kernel void add_rows_to_out(device float* x) {}\n";
    assert_eq!(rules("k.metal", metal), ["name-segments"]);
    let cuda = "extern \"C\" __global__ void add_rows(float* x) {}\n";
    assert!(rules("k.cu", cuda).is_empty());
}

#[test]
fn inline_tests_count() {
    let file = syn::parse_file("#[cfg(test)]\nmod t {\n    fn a() {}\n}\n").unwrap();
    assert_eq!(test_lines(&file.items), (4, Some(1)));
    let decl = syn::parse_file("#[cfg(test)]\nmod t_test;\n").unwrap();
    assert_eq!(test_lines(&decl.items), (0, None));
}

#[test]
fn inline_tests_fail() {
    let mut text = "fn a() {}\n".repeat(50);
    text.push_str("#[test]\nfn t() {}\n");
    assert_eq!(rules("a.rs", &text), ["inline-tests"]);
    assert!(rules("a_test.rs", &text).is_empty());
}

#[test]
fn whole_file_tests() {
    assert_eq!(
        rules("helpers.rs", "#![cfg(test)]\nfn a() {}\n"),
        ["inline-tests"]
    );
}

#[test]
fn test_file_names() {
    assert!(is_test_file("crates/x/src/ops_test.rs"));
    assert!(!is_test_file("crates/x/src/tests.rs"));
}
