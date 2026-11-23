use crate::site::{Ask, Kind, Lane, Offer, Shape, Site};
use crate::{decode, encode, respond};

mod multi {
    use crate::site::{Kind, Offer, Site};
    fn offer(_: &Site) -> Option<Vec<Offer>> {
        None
    }
    crate::exports! {
        rules: { "multi" => Kind::Model },
        kernels: { "lib.rs": ["multi", "other"], "kit.rs": ["third"] },
        files: ["site.rs"],
        propose: offer,
    }
}

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
    assert!(ask.config.is_none() && !serde_json::to_string(&ask).unwrap().contains("\"config\""));
    let with = br#"{"rule": "r", "site": {"matmul": {"m": 1, "n": 2, "k": 4, "form": "f32"}}, "config": {"hidden": 64}}"#;
    let out = respond(with, |a| {
        let len = a.config.as_ref()?["hidden"].as_u64()? as usize;
        Some(vec![Offer {
            kernel: "k",
            reads: vec![],
            len,
            counts: vec![],
            shape: Shape::Line(1),
            work: 1.0,
            guess: 1.0,
            writes: None,
            resident: None,
        }])
    });
    let reply: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(reply[0]["len"], 64);
    let bad: serde_json::Value = serde_json::from_slice(&respond(b"{", |_| None)).unwrap();
    assert!(bad["unread"].is_string());
    for (lane, json) in [(Lane::Sum, "\"sum\""), (Lane::SumBf16, "\"sum_bf16\"")] {
        assert_eq!(serde_json::to_string(&lane).unwrap(), json);
        assert_eq!(serde_json::from_str::<Lane>(json).unwrap(), lane);
    }
}
