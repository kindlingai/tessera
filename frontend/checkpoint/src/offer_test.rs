use crate::Stored;
use codegen::form::named;

#[test]
fn names_round_trip() {
    let form = Stored::Form(named("u32").unwrap());
    for stored in [Stored::F32, Stored::Bf16, Stored::F16, Stored::Other] {
        assert_eq!(Stored::parse(stored.name()).unwrap(), stored);
    }
    assert_eq!(Stored::parse(form.name()).unwrap().name(), "u32");
}

#[test]
fn names_keep_their_spelling() {
    let names = ["f32", "bf16", "f16", "other"];
    let stored = [Stored::F32, Stored::Bf16, Stored::F16, Stored::Other];
    for (name, stored) in names.iter().zip(stored) {
        assert_eq!(stored.name(), *name);
    }
}

#[test]
fn unknown_name_fails() {
    let e = Stored::parse("q9_9").unwrap_err();
    assert!(e.to_string().contains("q9_9"), "{e}");
}
