use tessera_guest::site::{Ask, Kind, Offer};

fn offer(_: &Ask) -> Option<Vec<Offer>> {
    None
}

tessera_guest::exports!(
    ask: "asked",
    Kind::Model,
    "exports_ask_test.rs",
    offer,
    ["../src/site.rs"]
);

#[test]
fn exports_ask_compiles() {
    let _ = (
        alloc as extern "C" fn(i32) -> i32,
        manifest as extern "C" fn() -> i64,
        propose as extern "C" fn(i32, i32) -> i64,
    );
}
