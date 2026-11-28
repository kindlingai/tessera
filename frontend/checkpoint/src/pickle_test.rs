use crate::pickle::{DType, Value, parse};

#[test]
fn pickle_memo_containers() {
    // Sparse memo, APPENDS, SETITEMS, and small/rank-5 tuples.
    let mut p = vec![0x80, 4, b'}', b'('];
    p.extend(b"\x8c\x01k\x94]");
    p.push(b'(');
    p.extend(b"\x8c\x02hi");
    p.push(b'r');
    p.extend(100_000u32.to_le_bytes());
    p.extend(b"\x8c\x02ok\x94");
    p.push(b'j');
    p.extend(100_000u32.to_le_bytes());
    p.extend(b"h\x02K\x07\x85K\x01K\x02\x86K\x03K\x04K\x05\x87(K\x01K\x02K\x03K\x04K\x05teu.");
    let Value::Dict(pairs) = parse(&p).unwrap() else {
        panic!("expected dict");
    };
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].0, Value::Str("k".into()));
    let Value::List(items) = &pairs[0].1 else {
        panic!("expected list");
    };
    assert_eq!(items.len(), 8);
    assert_eq!(items[2], Value::Str("hi".into()));
    assert_eq!(items[3], Value::Str("ok".into()));
    assert_eq!(items[4], Value::Tuple(vec![Value::Int(7)]));
    assert_eq!(items[5], Value::Tuple(vec![Value::Int(1), Value::Int(2)]));
    assert_eq!(items[7], Value::Tuple((1..=5).map(Value::Int).collect()));
    // Protocol-4 tensor with fused tQ/tR and Parameter unwrap.
    let mut t = b"\x80\x04\x8c\x0ctorch._utils\x8c\x12_rebuild_parameter\x93".to_vec();
    t.extend(b"(\x8c\x0ctorch._utils\x8c\x12_rebuild_tensor_v2\x93");
    t.extend(
        b"((\x8c\x07storage\x8c\x05torch\x8c\x0fBFloat16Storage\x93\x8c\x010\x8c\x03cpuK\x06tQ\x94",
    );
    t.extend(b"K\x00K\x02K\x03\x86K\x03K\x01\x86\x89}\x94tR\x94\x89}tR.");
    let Value::Tensor(view) = parse(&t).unwrap() else {
        panic!("expected tensor");
    };
    assert_eq!(view.storage.dtype, DType::Bf16);
    assert_eq!(view.shape, [2, 3]);
    assert_eq!(view.stride, [3, 1]);
}
