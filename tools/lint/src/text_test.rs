use crate::files::Source;
use crate::text::check;

fn lines(path: &str, text: &str) -> Vec<usize> {
    let src = Source {
        path: path.into(),
        text: text.into(),
        lines: text.lines().count(),
    };
    let mut out = Vec::new();
    check(&src, &mut out);
    out.iter().map(|v| v.line).collect()
}

#[test]
fn kernel_text_rule() {
    let body = "fn f() -> String {\n    format!(\"float {name}(u32 x) {{\\n\")\n}\n";
    assert_eq!(lines("a/b.rs", body), [2]);
    assert_eq!(
        lines("a/b.rs", "const P: &str = r\"f_const(u8) p\";\n"),
        [1]
    );
    assert!(lines("a/b.rs", "const P: &str = \"f_bits(x)\";\n").is_empty());
    let printer = "engine/plan/codegen/src/dialect.rs";
    assert!(lines(printer, "const P: &str = \"f_fn\";\n").is_empty());
    assert!(lines("a/b_test.rs", "const P: &str = \"f_fn\";\n").is_empty());
    assert!(lines("a/b.rs", "/// Prints `f_fn`.\nfn f() {}\n").is_empty());
}
