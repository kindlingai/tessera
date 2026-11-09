use crate::names::{Name, check, fault, pieces};

/// Names and their segments.
const PIECES: &[(&str, &[&str])] = &[
    ("a_b_c", &["a", "b", "c"]),
    ("__a__b_", &["a", "b"]),
    ("r#type", &["type"]),
];

/// A name, the crate's names, and whether it fails.
const FAULTS: &[(&str, &[&str], bool)] = &[
    ("a_b_c", &["a_b_c"], false),
    ("rms_norm_backward_cpu", &["rms_norm_backward_cpu"], true),
    (
        "rms_norm_backward_cpu",
        &["rms_norm_backward_cpu", "gelu_cpu"],
        false,
    ),
    (
        "dgemm_tile_load_f16",
        &["dgemm_tile_load_f16", "dgemm_run"],
        false,
    ),
    // Twins are not a family.
    ("a_b_c_d", &["a_b_c_d", "a_b_c_d"], true),
    ("a_b_c_d_cpu", &["a_b_c_d_cpu", "x_cpu"], true),
];

fn name(path: &str, name: &str) -> Name {
    Name {
        path: path.into(),
        line: 1,
        name: name.into(),
        what: "fn",
    }
}

#[test]
fn names_need_family() {
    for &(n, want) in PIECES {
        assert_eq!(pieces(n), want);
    }
    for &(n, names, fails) in FAULTS {
        assert_eq!(fault(n, names).is_some(), fails, "{n} among {names:?}");
    }
    // Families do not cross crates.
    let dirs = ["crates/a/".to_string(), "crates/b/".to_string()];
    let names = [
        name("crates/a/x.rs", "one_two_three_cpu"),
        name("crates/b/y.rs", "four_cpu"),
    ];
    let mut out = Vec::new();
    check(&dirs, &names, &mut out);
    assert_eq!(out.len(), 1);
}
