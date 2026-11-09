use crate::comments::{Comment, group, scan, strip, words};

fn grouped(text: &str, rust: bool) -> Vec<Comment> {
    group(&scan(text, rust))
}

#[test]
fn words_skip_punctuation() {
    assert_eq!(words(" a b-c  -- * x"), 3);
}

#[test]
fn adjacent_lines_join() {
    let c = grouped("// one two\n// three\nfn f() {}\n// four\n", true);
    assert_eq!(
        c,
        vec![Comment { line: 1, words: 3 }, Comment { line: 4, words: 1 }]
    );
}

#[test]
fn trailing_stays_alone() {
    let c = grouped("let a = 1; // one\n// two three\n", true);
    assert_eq!(
        c,
        vec![Comment { line: 1, words: 1 }, Comment { line: 2, words: 2 }]
    );
}

#[test]
fn doc_kinds_split() {
    let c = grouped("//! inner\n/// outer doc\n", true);
    assert_eq!(
        c,
        vec![Comment { line: 1, words: 1 }, Comment { line: 2, words: 2 }]
    );
}

#[test]
fn strings_hide_slashes() {
    let src = "let a = \"// not\"; let b = r#\"/* no \"# ; let c = '/';\n";
    assert!(scan(src, true).is_empty());
}

#[test]
fn lifetimes_are_code() {
    let src = "fn f<'a>(x: &'a str) {} // yes\n";
    let c = grouped(src, true);
    assert_eq!(c, vec![Comment { line: 1, words: 1 }]);
}

#[test]
fn rust_blocks_nest() {
    let c = grouped("/* a /* b */ c */ x\n", true);
    assert_eq!(c, vec![Comment { line: 1, words: 3 }]);
}

#[test]
fn kernel_blocks_flat() {
    let c = grouped("/* a /* b */ int c; /* d */\n", false);
    assert_eq!(c.len(), 2);
}

#[test]
fn strip_keeps_lines() {
    let s = strip("a // x\n/* y\nz */ b\n", false);
    assert_eq!(s.lines().count(), 3);
    assert!(!s.contains('x') && !s.contains('y') && s.contains('b'));
}
