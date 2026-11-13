use crate::files::Source;
use crate::runs::check;

/// Eleven lines with `i` arguments each, named by `n`.
fn source(path: &str, n: usize) -> Source {
    let text: String = (0..11)
        .map(|i| format!("let v{n} = f{n}({});\n", format!("x{n}, ").repeat(i)))
        .collect();
    let lines = text.lines().count();
    Source {
        path: path.into(),
        text,
        lines,
    }
}

fn hits(paths: &[&str]) -> usize {
    let srcs: Vec<Source> = paths
        .iter()
        .enumerate()
        .map(|(n, p)| source(p, n))
        .collect();
    let mut out = Vec::new();
    check(&srcs, &mut out);
    out.len()
}

#[test]
fn copies_per_type() {
    assert_eq!(hits(&["a.rs", "b.rs", "c.rs"]), 0);
    assert_eq!(hits(&["a.rs", "b.rs", "c.rs", "d.rs"]), 1);
    assert_eq!(hits(&["a.rs", "b.rs", "c.rs", "d.cu"]), 0);
}
