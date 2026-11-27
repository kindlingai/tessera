use tessera_linear::Turned;
use tessera_lower::form::named;
use tessera_lower::{Kernel, Source};

fn entry(k: &Turned) -> String {
    match k.source() {
        Source::Made(body) => body.entry,
        Source::Fixed { entry, .. } => entry.into(),
    }
}

#[test]
fn distinct_row_entries() {
    let form = named("trellis_k4").unwrap();
    let one = Turned {
        m: 1,
        n: 128,
        k: 512,
        rows_cap: 4,
        form,
    };
    let three = Turned { m: 3, ..one };
    assert_ne!(entry(&one), entry(&three));
}
