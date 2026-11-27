use std::collections::BTreeMap;

use crate::rewrite::{Moves, exports, rewrite};

fn moved(src: &str, module: &str, dir: &str) -> String {
    let names = BTreeMap::from([
        ("tessera_graph".to_string(), "crate::graph".to_string()),
        ("tessera_trace".to_string(), "crate::trace".to_string()),
    ]);
    let include = |old: &str| Some(format!("up/{old}"));
    let moves = Moves {
        module,
        names: &names,
        dir,
        include: &include,
    };
    rewrite(src, &moves).unwrap()
}

#[test]
fn paths_move_inside() {
    let src = "use tessera_graph::{Graph, shard};\nfn f(x: ::tessera_graph::G) { crate::a(); quote!(#t ::tessera_trace::W); include_str!(\"k.cu\"); }";
    let want = "use crate::graph::{Graph, shard};\nfn f(x: crate::graph::G) { crate::lower::a(); quote!(#t crate::trace::W); include_str!(\"up/k.cu\"); }";
    assert_eq!(moved(src, "lower", ""), want);
}

#[test]
fn aliases_drop() {
    let src = "extern crate self as tessera_trace;\nuse crate::{self as tessera_trace, *};\nfn g() { a::tessera_graph::x(); $crate::m!(); }";
    let want = "\nuse crate::trace::{ *};\nfn g() { a::tessera_graph::x(); $crate::trace::m!(); }";
    assert_eq!(moved(src, "trace", ""), want);
}

#[test]
fn manifest_dirs_and_exports() {
    let src = "static T: Dir = include_dir!(\"$CARGO_MANIFEST_DIR/platforms\");\n#[macro_export]\nmacro_rules! rule { () => {} }";
    assert!(
        moved(src, "platform", "src/gpu/platform")
            .contains("$CARGO_MANIFEST_DIR/src/gpu/platform/platforms")
    );
    assert_eq!(exports(src).unwrap(), ["rule"]);
}
