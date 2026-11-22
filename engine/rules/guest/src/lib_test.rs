use crate::site::{Ask, Kind, Site};
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

#[test]
fn ask_parses() {
    let text = r#"{"rule": "r", "site": {"region": {"kind": "moe", "outputs": [8], "kept": [], "sums": [],
        "inputs": [{"shape": [2, 4], "dims": ["Rows", null], "form": "f32"}]}}}"#;
    let ask: Ask = serde_json::from_str(text).unwrap();
    let Site::Region { kind, inputs, .. } = &ask.site else {
        panic!("a region site");
    };
    assert!(*kind == Kind::Moe && inputs[0].dims[0].as_deref() == Some("Rows"));
    assert!(ask.config.is_none());
    let plain = serde_json::to_string(&ask).unwrap();
    assert!(!plain.contains("\"config\""));
    let with = r#"{"rule": "r", "site": {"matmul": {"m": 1, "n": 2, "k": 4, "form": "f32"}}, "config": {"hidden": 64}}"#;
    let parsed: Ask = serde_json::from_str(with).unwrap();
    assert_eq!(parsed.config.unwrap()["hidden"], 64);
}
