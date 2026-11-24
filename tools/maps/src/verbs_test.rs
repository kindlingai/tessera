use std::path::Path;

use crate::Atlas;
use crate::map::read;

const GOOD: &str = r#"[task.add-thing]
steps = [{ do = "register", with = "tessera_lower::rule!", like = "engine/rules/kernels/src/cast.rs" }]
"#;

const PATH: &str = "maps/add-thing.toml";

fn faults(atlas: &mut Atlas, text: &str) -> Vec<String> {
    atlas.faults.clear();
    atlas.maps = read(PATH, text, &mut atlas.faults);
    atlas.texts.insert(PATH.into(), text.into());
    atlas.check().into_iter().map(|f| f.message).collect()
}

#[test]
fn refuses_false_maps() {
    let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."));
    let atlas = &mut Atlas::open(root).unwrap();
    assert_eq!(faults(atlas, GOOD), Vec::<String>::new());
    let each =
        "}]\neach = [{ of = \"tessera_bundle::arch::ARCHES\", has = \"tessera_parity::Spec\" }]";
    let cases = [
        ("rule!", "rule_x!", "names no item"),
        ("kernels/src/cast.rs", "kernels/src/lib.rs", "calls neither"),
        ("[task", "# why\n[task", "`#` comment"),
        (" }]", ", note = \"x\" }]", "unknown field"),
        ("}]\n", "}]\n[task.add-more]\nsteps = []\n", "one task"),
        (
            "\"register\"",
            "\"define\", via = \"tessera_lower::rule!\"",
            "does not take `via`",
        ),
        ("}]", each, "no `tessera_parity::Spec` for flux2"),
    ];
    for (from, to, want) in cases {
        let got = faults(atlas, &GOOD.replacen(from, to, 1));
        assert!(got.iter().any(|m| m.contains(want)), "{to}: {got:?}");
    }
}
