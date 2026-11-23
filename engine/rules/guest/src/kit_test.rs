use crate::site::{Count, Input, Place, Weights};

fn weight(path: &str, share: Option<[usize; 4]>) -> Input {
    Input {
        shape: vec![8, 4],
        dims: vec![Some("Rows".into()), None],
        form: "f32".into(),
        weight: Some(path.into()),
        share,
        parts: None,
    }
}

#[test]
fn weights_by_field() {
    let inputs = [
        weight("moe.3.gate", None),
        weight("moe.3.shared.gate", Some([1, 4, 4, 16])),
    ];
    let w = Weights::new(&inputs);
    let [gate, shared] = w.all(["gate", "shared.gate"]).unwrap();
    assert!(matches!(gate.read(), Place::Read(0)));
    assert!(matches!(shared.part(2), Place::Part { part: [1, 2] }));
    assert!(w.get("moe.3.gate").is_none() && w.typed("gate", "bf16").is_none());
    assert_eq!((shared.whole(1), shared.whole(0)), (16, 8));
    assert_eq!(shared.split(1, 2), Some(Some([2, 2])));
    assert_eq!((shared.split(0, 1), gate.split(0, 1)), (None, Some(None)));
    assert!(matches!(gate.count(), Count::Dim { mul: 1, .. }));
}
