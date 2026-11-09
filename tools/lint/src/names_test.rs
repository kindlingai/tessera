use crate::names::{Name, check, fault, pieces};

#[test]
fn pieces_skip_underscores() {
    assert_eq!(pieces("a_b_c"), ["a", "b", "c"]);
    assert_eq!(pieces("__a__b_"), ["a", "b"]);
    assert_eq!(pieces("r#type"), ["type"]);
}

#[test]
fn three_always_pass() {
    assert!(fault("a_b_c", &["a_b_c"]).is_none());
}

#[test]
fn four_need_family() {
    let alone = ["rms_norm_backward_cpu"];
    assert!(fault("rms_norm_backward_cpu", &alone).is_some());
    let suffix = ["rms_norm_backward_cpu", "gelu_cpu"];
    assert!(fault("rms_norm_backward_cpu", &suffix).is_none());
    let prefix = ["dgemm_tile_load_f16", "dgemm_run"];
    assert!(fault("dgemm_tile_load_f16", &prefix).is_none());
}

#[test]
fn twins_are_not_family() {
    let twins = ["a_b_c_d", "a_b_c_d"];
    assert!(fault("a_b_c_d", &twins).is_some());
}

#[test]
fn five_always_fail() {
    assert!(fault("a_b_c_d_cpu", &["a_b_c_d_cpu", "x_cpu"]).is_some());
}

#[test]
fn family_per_crate() {
    let name = |path: &str, name: &str| Name {
        path: path.into(),
        line: 1,
        name: name.into(),
        what: "fn",
    };
    let dirs = ["crates/a/".to_string(), "crates/b/".to_string()];
    let names = [
        name("crates/a/x.rs", "one_two_three_cpu"),
        name("crates/b/y.rs", "four_cpu"),
    ];
    let mut out = Vec::new();
    check(&dirs, &names, &mut out);
    assert_eq!(out.len(), 1);
}
