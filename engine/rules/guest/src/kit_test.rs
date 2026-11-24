use crate::site::{Count, Input, Place, Weights, agreed, trellis};

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
    assert!(w.get("moe.3.gate").is_none() && w.all(["gate", "up"]).is_none());
    assert_eq!((shared.whole(1), shared.whole(0)), (16, 8));
    assert_eq!(shared.split(1, 2), Some(Some([2, 2])));
    assert_eq!((shared.split(0, 1), gate.split(0, 1)), (None, Some(None)));
    assert!(matches!(gate.count(), Count::Dim { mul: 1, .. }));
}

#[test]
fn trellis_and_agreed() {
    let t = trellis("trellis_k3_pmcgx_rot").unwrap();
    assert_eq!([t.k, t.book, t.rotated as u32], [3, 2, 1]);
    assert!(trellis("trellis_k5").is_none() && trellis("trellis_k3_rot_mul1").is_none());
    assert_eq!(agreed(&[None, Some(2), Some(2)], 9), Some(2));
    assert_eq!(agreed(&[None, None], 9), Some(9));
    assert_eq!(agreed(&[Some(1), Some(2)], 9), None);
}
