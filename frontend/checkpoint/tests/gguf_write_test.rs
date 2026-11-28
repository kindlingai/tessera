use tessera_checkpoint::{Gguf, GgufTensor, GgufType, Meta, write_gguf};

#[test]
fn gguf_write_edges() {
    let meta = vec![
        ("general.alignment".into(), Meta::U32(64)),
        ("empty_arr".into(), Meta::Array(vec![])),
        (
            "nested".into(),
            Meta::Array(vec![
                Meta::Array(vec![Meta::U16(7), Meta::U16(9)]),
                Meta::Array(vec![Meta::U16(11)]),
            ]),
        ),
        ("flag".into(), Meta::Bool(false)),
    ];
    let tensors = vec![
        GgufTensor {
            name: "a".into(),
            ty: GgufType::F32,
            dims: vec![3],
            bytes: [1.0f32, 2.0, 3.0]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect(),
        },
        GgufTensor {
            name: "b".into(),
            ty: GgufType::F32,
            dims: vec![1],
            bytes: 4.5f32.to_le_bytes().to_vec(),
        },
    ];
    let bytes = write_gguf(&meta, &tensors);
    let path = tessera_testkit::stage("gguf-write-edges.gguf", &bytes);
    let g = Gguf::open(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(g.meta("empty_arr"), Some(&Meta::Array(vec![])));
    assert_eq!(g.meta("nested"), Some(&meta[2].1));
    assert_eq!(g.meta("flag"), Some(&Meta::Bool(false)));
    assert_eq!(g.floats("a").unwrap(), [1.0, 2.0, 3.0]);
    assert_eq!(g.floats("b").unwrap(), [4.5]);
}
