use crate::verify_test::{mlp, push};
use crate::*;

fn with_scale(scale: f32) -> Graph {
    let mut g = mlp();
    let last = NodeId(g.nodes.len() as u32 - 1);
    let y = push(&mut g, Op::Row(Row(vec![Stage::Scale(scale)])), &[last]);
    g.entries[0].outputs[0].node = y;
    g
}

/// Asserts `g` survives JSON with its text unchanged.
fn survives(g: &Graph) -> Graph {
    let back = Graph::from_json(&g.to_json()).unwrap();
    assert_eq!(&back, g);
    assert_eq!(back.to_json(), g.to_json());
    verify(&back).unwrap();
    back
}

#[test]
fn round_trip() {
    let scale = 1.0 / 3f32.sqrt();
    let back = survives(&with_scale(scale));
    let op = &back.nodes.last().unwrap().op;
    assert_eq!(op, &Op::Row(Row(vec![Stage::Scale(scale)])));
}

#[test]
fn version_checked() {
    let text = mlp()
        .to_json()
        .replacen(&format!("\"version\": {VERSION}"), "\"version\": 99", 1);
    let err = Graph::from_json(&text).unwrap_err();
    assert!(matches!(err, FormatError::Version { found: 99 }), "{err}");
}

#[test]
fn hash_tracks_content() {
    assert_eq!(with_scale(0.5).hash(), with_scale(0.5).hash());
    assert_ne!(with_scale(0.5).hash(), with_scale(0.25).hash());
}
