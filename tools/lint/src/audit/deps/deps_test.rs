use super::{Thin, check};
use crate::audit::workspace::{Dep, Kind, Pkg};
use crate::files::Source;

fn pkg(name: &str, deps: &[(&str, Kind)]) -> Pkg {
    let dep = |(n, kind): &(&str, Kind)| Dep {
        name: (*n).into(),
        kind: *kind,
    };
    Pkg {
        name: name.into(),
        dir: format!("e/{name}/"),
        manifest: String::new(),
        deps: deps.iter().map(dep).collect(),
    }
}

/// `user` uses `big`, `low` and, in tests, `kit`.
fn pkgs() -> Vec<Pkg> {
    let user = pkg(
        "user",
        &[
            ("big", Kind::Normal),
            ("low", Kind::Normal),
            ("kit", Kind::Dev),
        ],
    );
    let rest = ["big", "low", "kit", "other"].map(|n| pkg(n, &[]));
    [user].into_iter().chain(rest).collect()
}

/// Rules and messages. `other` names six and four things.
fn run(files: &[(&str, &str)]) -> Vec<(&'static str, String)> {
    let big = (
        "e/other/src/lib.rs",
        "use big::{A, B, C, D, E, F};\nuse kit::{a, b, c, d};\n",
    );
    let sources: Vec<Source> = files
        .iter()
        .chain([&big])
        .map(|(p, t)| Source {
            path: (*p).into(),
            text: (*t).into(),
            lines: t.lines().count(),
        })
        .collect();
    let thin = Thin {
        dirs: vec!["e/".into()],
        min_names: 2,
        min_dev_names: 1,
        min_surface: 4,
    };
    let mut out = Vec::new();
    check(&pkgs(), &sources, &["e/low/".into()], &thin, &mut out);
    out.into_iter().map(|v| (v.rule, v.message)).collect()
}

fn rules(found: &[(&'static str, String)]) -> Vec<&'static str> {
    found.iter().map(|(r, _)| *r).collect()
}

#[test]
fn thin_edges() {
    let one = run(&[("e/user/src/lib.rs", "fn f() { big::A::new(); low::x(); }\n")]);
    assert_eq!(rules(&one), ["thin-edge", "thin-edge"]);
    assert!(one[0].1.contains("for 1 of the 6 names") && one[1].1.contains("`kit`"));
    let lib = "use big::{A, m::B};\nuse big as _;\n";
    let test = ("e/user/tests/t.rs", "use kit::scratch;\n");
    assert!(run(&[("e/user/src/lib.rs", lib), test]).is_empty());
}

#[test]
fn testing_paths() {
    let lib = "use big::{A, B};\nfn f() { crate::testing::ci(); }\n\
               #[cfg(test)]\nmod t { use big::testing::x; }\n";
    let test = ("e/user/src/lib_test.rs", "use kit::testing::y;\n");
    let main = ("e/user/src/main.rs", "fn main() {}\n");
    let found = run(&[("e/user/src/lib.rs", lib), test, main]);
    assert_eq!(rules(&found), ["testing-path"]);
    let harness = run(&[("e/user/src/lib.rs", lib), test]);
    assert!(!rules(&harness).contains(&"testing-path"));
}

#[test]
fn reexports() {
    let lib = "pub use big::{A, B};\npub use low::C;\npub use own::D;\n";
    let user = (
        "e/kit/src/lib.rs",
        "use user::m::*;\nfn f() { user::A; user::C; }\n",
    );
    let found = run(&[
        ("e/user/src/lib.rs", lib),
        ("e/user/src/m.rs", "pub use big::E;\n"),
        user,
        ("e/user/tests/t.rs", "use kit::k;\n"),
    ]);
    assert_eq!(rules(&found), ["reexport"]);
    assert!(found[0].1.starts_with("`B`"));
}
