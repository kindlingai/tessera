use crate::lists::resolve;

fn conflict(ours: &str, base: &str, theirs: &str) -> String {
    format!("<<<<<<< main\n{ours}||||||| base\n{base}=======\n{theirs}>>>>>>> head\n")
}

/// The refusal's line, when its reason holds `word`.
fn refusal(path: &str, text: &str, word: &str) -> Option<usize> {
    let refusal = resolve(path, text).err()?;
    refusal.why.contains(word).then_some(refusal.line)
}

#[test]
fn use_lines_join() {
    let hunk = conflict(
        "use a::A;\nuse b::B;\nuse c::{\n    C,\n    D,\n};\n",
        "use b::B;\n",
        "use b::B;\nuse a::A;\npub(crate) use e::E;\n",
    );
    let text = format!("//! Doc.\n{hunk}\nfn f() {{}}\n");
    let joined = "use a::A;\nuse b::B;\nuse c::{\n    C,\n    D,\n};\npub(crate) use e::E;\n";
    assert_eq!(
        resolve("src/lib.rs", &text).unwrap(),
        format!("//! Doc.\n{joined}\nfn f() {{}}\n")
    );
}

#[test]
fn group_items_join() {
    let hunk = conflict("    fs,\n", "", "    io::{self, Read},\n");
    let text = format!("use std::{{\n    env,\n{hunk}}};\n");
    let joined = "use std::{\n    env,\n    fs,\n    io::{self, Read},\n};\n";
    assert_eq!(resolve("a.rs", &text).unwrap(), joined);
    let text = format!("enum E {{\n    A,\n{hunk}}}\n");
    assert_eq!(refusal("a.rs", &text, "`use` group"), Some(3));
}

#[test]
fn mod_lines_join() {
    let hunk = conflict(
        "#[cfg(test)]\nmod a_test;\n",
        "",
        "#[cfg(test)]\nmod b_test;\npub mod c;\n",
    );
    let joined = "#[cfg(test)]\nmod a_test;\n#[cfg(test)]\nmod b_test;\npub mod c;\n";
    assert_eq!(resolve("src/lib.rs", &hunk).unwrap(), joined);
}

#[test]
fn dependencies_join() {
    let hunk = conflict(
        "serde = \"1\"\nureq = \"2\"\n",
        "serde = \"1\"\n",
        "ring = { version = \"0.17\", features = [\"std\"] }\nserde = \"1\"\n",
    );
    let head = "[package]\nname = \"x\"\n\n[dependencies]\n";
    let joined =
        "ring = { version = \"0.17\", features = [\"std\"] }\nserde = \"1\"\nureq = \"2\"\n";
    assert_eq!(
        resolve("x/Cargo.toml", &format!("{head}{hunk}")).unwrap(),
        format!("{head}{joined}")
    );
    let text = format!("[features]\n{hunk}");
    assert_eq!(refusal("x/Cargo.toml", &text, "dependency table"), Some(2));
}

#[test]
fn mixed_hunk_refused() {
    let hunk = conflict("use a::A;\n", "", "use b::B;\n\nfn f() {}\n");
    assert_eq!(
        refusal("a.rs", &format!("//! Doc.\n\n{hunk}"), "not a `use`"),
        Some(3)
    );
    let hunk = conflict("mod a;\n", "", "use b::B;\n");
    assert_eq!(refusal("a.rs", &hunk, "not a `use`"), Some(1));
}

#[test]
fn edited_line_refused() {
    let hunk = conflict("use a::{C};\n", "use a::{B};\n", "use a::{B};\nuse d::D;\n");
    assert_eq!(refusal("a.rs", &hunk, "changes or removes"), Some(1));
    let hunk = conflict(
        "ring = \"0.17\"\n",
        "ring = \"0.16\"\n",
        "ring = \"0.16\"\nureq = \"2\"\n",
    );
    let text = format!("[dependencies]\n{hunk}");
    assert_eq!(refusal("Cargo.toml", &text, "changes or removes"), Some(2));
}

#[test]
fn edited_uses_join() {
    let hunk = conflict(
        "use a::B;\nuse a::c::D;\n",
        "use a::B;\n",
        "#[cfg(test)]\nuse t::T;\nuse a::{self, B, E as F};\npub use g::*;\n",
    );
    let joined =
        "use a::{B, E as F, self};\nuse a::c::D;\npub use g::*;\n#[cfg(test)]\nuse t::T;\n";
    assert_eq!(resolve("a.rs", &hunk).unwrap(), joined);
    let hunk = conflict(
        "    A, B, C,\n    D,\n",
        "    A, B,\n    D,\n",
        "    A, B, D,\n    x::{Y, Z},\n",
    );
    let text = format!("use m::{{\n{hunk}}};\n");
    assert_eq!(
        resolve("a.rs", &text).unwrap(),
        "use m::{\n    A, B, C, D, x::{Y, Z},\n};\n"
    );
}
