use super::layers::{Allow, check};
use super::workspace::{Dep, Kind, Pkg};

const MANIFEST: &str =
    "[dependencies]\nlow.workspace = true\n\n[dev-dependencies]\nhigh.workspace = true\n";

fn pkg(name: &str, dir: &str, deps: &[(&str, Kind)]) -> Pkg {
    let dep = |(n, kind): &(&str, Kind)| Dep {
        name: (*n).into(),
        kind: *kind,
    };
    Pkg {
        name: name.into(),
        dir: dir.into(),
        manifest: MANIFEST.into(),
        deps: deps.iter().map(dep).collect(),
    }
}

fn allow(from: &str, to: &str) -> Allow {
    Allow {
        from: from.into(),
        to: to.into(),
        reason: "test".into(),
    }
}

/// Flagged paths and lines, with `a/` below `b/` and `c/`.
fn run(pkgs: &[Pkg], allowed: &[Allow]) -> Vec<(String, usize)> {
    let layers = vec![vec!["a/".into()], vec!["b/".into(), "c/".into()]];
    let mut out = Vec::new();
    check(pkgs, &layers, allowed, &mut out);
    out.into_iter().map(|v| (v.path, v.line)).collect()
}

/// `low` below `high`, with `deps` on `low`.
fn pair(deps: &[(&str, Kind)]) -> Vec<Pkg> {
    vec![pkg("low", "a/low/", deps), pkg("high", "b/high/", &[])]
}

/// Packages that depend on their own layer or lower.
fn ladder() -> Vec<Pkg> {
    let mid = pkg("mid", "b/mid/", &[("low", Kind::Normal)]);
    let side = pkg("side", "c/side/", &[("mid", Kind::Dev)]);
    vec![pkg("low", "a/low/", &[]), mid, side]
}

fn at(path: &str, line: usize) -> Vec<(String, usize)> {
    vec![(path.to_string(), line)]
}

#[test]
fn up_fails() {
    let up = |kind| run(&pair(&[("high", kind)]), &[]);
    assert_eq!(up(Kind::Normal), at("a/low/Cargo.toml", 1));
    assert_eq!(up(Kind::Dev), at("a/low/Cargo.toml", 5));
}

#[test]
fn lower_passes() {
    assert!(run(&ladder(), &[]).is_empty());
}

#[test]
fn allow_list() {
    let up = pair(&[("high", Kind::Dev)]);
    assert!(run(&up, &[allow("low", "high")]).is_empty());
    assert_eq!(run(&pair(&[]), &[allow("low", "high")]), at("lint.toml", 1));
}

#[test]
fn unknown_directory_fails() {
    let odd = [pkg("odd", "z/odd/", &[])];
    assert_eq!(run(&odd, &[]), at("z/odd/Cargo.toml", 1));
}
