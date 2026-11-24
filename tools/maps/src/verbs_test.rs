use std::path::Path;

use crate::Atlas;
use crate::map::read;

const GOOD: &str = r#"[task.add-thing]
steps = [{ do = "register", with = "tessera_lower::rule!", like = "engine/rules/kernels/src/cast.rs" }]
"#;

fn faults(text: &str) -> Vec<String> {
    let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    let mut atlas = Atlas::open(root).unwrap();
    let path = "maps/add-thing.toml";
    atlas.maps = read(path, text, &mut atlas.faults);
    atlas.texts.insert(path.into(), text.into());
    atlas.check().into_iter().map(|f| f.message).collect()
}

#[test]
fn refuses_false_maps() {
    assert_eq!(faults(GOOD), Vec::<String>::new());
    let cases = [
        ("rule!", "rule_x!", "names no item"),
        ("kernels/src/cast.rs", "kernels/src/lib.rs", "calls neither"),
        ("[task", "# why\n[task", "`#` comment"),
        (" }]", ", note = \"x\" }]", "unknown field"),
        ("add-thing", "add-one-more-thing", "over the limit"),
        (
            "\"register\"",
            "\"define\", via = \"tessera_lower::rule!\"",
            "does not take `via`",
        ),
    ];
    for (from, to, want) in cases {
        let got = faults(&GOOD.replace(from, to));
        assert!(got.iter().any(|m| m.contains(want)), "{to}: {got:?}");
    }
}
