use tessera_device::{Backend, Options, Problem, flatten};

/// Block comments, slashes, CRLF, and missing final newlines flatten cleanly.
#[test]
fn flatten_comment_edges() {
    let opts = Options::plain(Backend::Metal);
    let none = |_: &str| None;
    let src = "/* a */ #include <tessera/1/core> /* b */\n\
               // __has_include(<x>) inside comment\n\
               float y = a / b /* c */ + c / d; // end\n\
               float tail = 1.0f";
    let flat = flatten("k.kernel", src, &none, &opts).unwrap();
    assert!(!flat.text.contains("#include"));
    assert!(flat.text.ends_with("float tail = 1.0f\n"));
    let crlf = "#include <tessera/1/core>\r\nint x = 1;\r\n";
    let flat_crlf = flatten("k.kernel", crlf, &none, &opts).unwrap();
    assert!(flat_crlf.text.ends_with("\nint x = 1;\n"));
    let bad = "#include <tessera/1/core>\n/* ok */ #if __has_include(<x>)\n#endif\n";
    let err = flatten("k.kernel", bad, &none, &opts).unwrap_err();
    assert_eq!(err.problem, Problem::Late);
}
