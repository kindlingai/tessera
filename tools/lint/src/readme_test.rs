use crate::readme::{Crate, MARKER, folders, splice, table};

fn krate(dir: &str, name: &str) -> Crate {
    Crate {
        dir: dir.into(),
        name: name.into(),
        description: format!("{name} <text>"),
    }
}

#[test]
fn tables_per_folder() {
    let crates = [
        krate("a/x/one/", "one"),
        krate("a/two/", "two"),
        krate("a/x-y/", "dash"),
    ];
    let f = folders(&crates);
    let keys: Vec<&str> = f.keys().map(String::as_str).collect();
    assert_eq!(keys, ["", "a/", "a/x/"]);
    let rows = [
        "<table>",
        "<tr><th>Folder</th><th>Crate</th><th>Description</th></tr>",
        "<tr><td>&emsp;<a href=\"a/two\">two/</a></td><td>two</td><td>two &lt;text&gt;</td></tr>",
        "<tr><td colspan=\"3\">&emsp;<a href=\"a/x\"><b>x/</b></a></td></tr>",
        "<tr><td>&emsp;&emsp;<a href=\"a/x/one\">one/</a></td><td>one</td><td>one &lt;text&gt;</td></tr>",
        "<tr><td>&emsp;<a href=\"a/x-y\">x-y/</a></td><td>dash</td><td>dash &lt;text&gt;</td></tr>",
        "</table>",
    ];
    let root = table("", &f[""]);
    let head = rows[..2].join("\n");
    assert!(root.starts_with(&format!(
        "{head}\n<tr><td colspan=\"3\"><a href=\"a\"><b>a/</b></a></td></tr>\n"
    )));
    assert!(root.ends_with(&(rows[2..].join("\n") + "\n")));
}

#[test]
fn splice_keeps_other_text() {
    let markdown = format!("# T\n{MARKER}\nold | row\n--- | ---\n\nAfter.\n");
    let html = format!("# T\n{MARKER}\n<table>\n<tr><td>x</td></tr>\n</table>\n\nAfter.\n");
    for text in [markdown, html] {
        let new = splice(&text, "<table>\n</table>\n").unwrap();
        assert_eq!(new, format!("# T\n{MARKER}\n<table>\n</table>\n\nAfter.\n"));
    }
    assert!(splice("# T\n", "x").is_none());
}
