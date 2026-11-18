use crate::{decode, encode};

#[test]
fn tables_round_trip() {
    let tables = vec![
        ("const.a".to_string(), vec![1.0, -2.5]),
        ("b".into(), vec![]),
    ];
    let bytes = encode(&tables);
    assert_eq!(decode(&bytes), Some(tables));
    assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
}
