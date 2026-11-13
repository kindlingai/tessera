use crate::files::Source;
use crate::runs::check;

fn source(path: &str, text: String) -> Source {
    let lines = text.lines().count();
    Source {
        path: path.into(),
        text,
        lines,
    }
}

/// Violations over files of `line(i, n)`, one per path.
fn hits(paths: &[&str], line: impl Fn(usize, usize) -> String) -> usize {
    let srcs: Vec<Source> = paths
        .iter()
        .enumerate()
        .map(|(n, p)| source(p, (0..11).map(|i| line(i, n) + "\n").collect()))
        .collect();
    let mut out = Vec::new();
    check(&srcs, &mut out);
    out.len()
}

/// A call with `i` arguments, named by `n`.
fn call(i: usize, n: usize) -> String {
    format!("let v{n} = f{n}({});", format!("x{n}, ").repeat(i))
}

#[test]
fn copies_per_type() {
    assert_eq!(hits(&["a.rs", "b.rs", "c.rs"], call), 0);
    assert_eq!(hits(&["a.rs", "b.rs", "c.rs", "d.rs"], call), 1);
    assert_eq!(hits(&["a.rs", "b.rs", "c.rs", "d.cu"], call), 0);
}

#[test]
fn fields_are_not_code() {
    let all = ["a.rs", "b.rs", "c.rs", "d.rs"];
    let field = |i: usize, n: usize| match i % 2 {
        0 => format!("pub f{n}: Vec<u{n}>,"),
        _ => format!("g{n}: u{n},"),
    };
    let arm = |i: usize, n: usize| match i % 2 {
        0 => format!("K{n}::A => f(x{n}),"),
        _ => format!("K{n}::B => {n},"),
    };
    assert_eq!(hits(&all, field), 0);
    assert_eq!(hits(&all, arm), 1);
}
