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

/// Violations over `len` lines of `line(i, n)` per path.
fn hits_of(len: usize, paths: &[&str], line: impl Fn(usize, usize) -> String) -> usize {
    let srcs: Vec<Source> = paths
        .iter()
        .enumerate()
        .map(|(n, p)| source(p, (0..len).map(|i| line(i, n) + "\n").collect()))
        .collect();
    let mut out = Vec::new();
    check(&srcs, &mut out);
    out.len()
}

fn hits(paths: &[&str], line: impl Fn(usize, usize) -> String) -> usize {
    hits_of(11, paths, line)
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
fn struct_fields_keep_names() {
    let all = ["a.rs", "b.rs", "c.rs", "d.rs"];
    let def = |own: bool| {
        move |i: usize, n: usize| match (i, i % 2) {
            (0, _) => format!("pub struct S{n} {{"),
            (_, 0) => format!("pub f{i}{}: Vec<u8>,", if own { n } else { 0 }),
            _ => format!("pub g{i}{}: u8,", if own { n } else { 0 }),
        }
    };
    let init = |i: usize, n: usize| match (i, i % 2) {
        (0, _) => format!("let s{n} = S {{"),
        (_, 0) => format!("a{n}: x{n},"),
        _ => format!("b{n}: f(x{n}),"),
    };
    assert_eq!(hits(&all, def(true)), 0);
    assert_eq!(hits(&all, def(false)), 1);
    assert_eq!(hits(&all, init), 1);
}

#[test]
fn copies_per_step() {
    let (two, three) = (["a.rs", "b.rs"], ["a.rs", "b.rs", "c.rs"]);
    assert_eq!(hits_of(19, &three, call), 0);
    assert_eq!(hits_of(20, &three, call), 1);
    assert_eq!(hits_of(39, &two, call), 0);
    assert_eq!(hits_of(40, &two, call), 1);
    assert_eq!(hits_of(40, &["a.rs", "b.rs", "c.rs", "d.rs"], call), 1);
}
