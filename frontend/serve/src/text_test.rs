use crate::text::Spool;

/// Ids 1 and 2 are two halves of one character.
fn decode(ids: &[u32]) -> String {
    let mut bytes = Vec::new();
    for id in ids {
        match id {
            1 => bytes.push(0xC3),
            2 => bytes.push(0xA9),
            _ => bytes.extend_from_slice(b"ab"),
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

#[test]
fn whole_characters() {
    let mut spool = Spool::default();
    let pieces = [7, 1, 2, 7].map(|id| spool.push(id, decode));
    assert_eq!(pieces, ["ab", "", "\u{e9}", "ab"]);
}
