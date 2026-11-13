use crate::readme::{Crate, MARKER, folders, splice, table};

fn krate(dir: &str, name: &str) -> Crate {
    Crate {
        dir: dir.into(),
        name: name.into(),
        description: format!("{name} | text"),
    }
}

#[test]
fn tables_per_folder() {
    let crates = [krate("a/x/one/", "one"), krate("a/two/", "two")];
    let f = folders(&crates);
    let keys: Vec<&str> = f.keys().map(String::as_str).collect();
    assert_eq!(keys, ["", "a/", "a/x/"]);
    let t = table("a/", &f["a/"]);
    let rows = "[two](two) | two \\| text\n[one](x/one) | one \\| text\n";
    assert_eq!(t, format!("Crate | Description\n--- | ---\n{rows}"));
}

#[test]
fn splice_keeps_other_text() {
    let text = format!("# T\n{MARKER}\nold | row\n--- | ---\n\nAfter.\n");
    let new = splice(&text, "new | row\n").unwrap();
    assert_eq!(new, format!("# T\n{MARKER}\nnew | row\n\nAfter.\n"));
    assert!(splice("# T\n", "x").is_none());
}
