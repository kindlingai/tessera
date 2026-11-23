use tessera_guest::site::{Ask, Kind, Offer};

fn offer(_: &Ask) -> Option<Vec<Offer>> {
    None
}

tessera_guest::exports! {
    rules: { "asked" => Kind::Model },
    kernels: { "exports_ask_test.rs": ["asked"] },
    files: ["../src/site.rs"],
    ask: offer,
}

#[test]
fn exports_ask_compiles() {
    let _ = (
        alloc as extern "C" fn(i32) -> i32,
        manifest as extern "C" fn() -> i64,
        propose as extern "C" fn(i32, i32) -> i64,
    );
}
