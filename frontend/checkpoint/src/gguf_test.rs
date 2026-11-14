use crate::{Error, Gguf, GgufTensor, GgufType, Meta, Offer, Stored, write_gguf};

const IDS: [u32; 11] = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12];

fn scalars() -> [Meta; 11] {
    [
        Meta::U8(255),
        Meta::I8(-2),
        Meta::U16(9),
        Meta::I16(-10),
        Meta::U32(11),
        Meta::I32(-12),
        Meta::F32(1.5),
        Meta::Bool(true),
        Meta::U64(13),
        Meta::I64(-14),
        Meta::F64(2.5),
    ]
}

fn sample(alignment: u64) -> Vec<u8> {
    let mut meta: Vec<(String, Meta)> = IDS
        .iter()
        .zip(scalars())
        .map(|(ty, v)| (format!("t{ty}"), v))
        .collect();
    let hello = Meta::Str("hello".into());
    meta.extend([
        ("t8".into(), hello.clone()),
        ("strings".into(), Meta::Array(vec![hello])),
        (
            "integers".into(),
            Meta::Array(vec![Meta::I32(1), Meta::I32(2)]),
        ),
        ("positive".into(), Meta::I64(3)),
        ("general.alignment".into(), Meta::U64(alignment)),
    ]);
    let floats = [1.25f32, -2.5].iter().flat_map(|f| f.to_le_bytes());
    let tensor = |name: &str, ty, count, bytes: Vec<u8>| GgufTensor {
        name: name.into(),
        ty,
        dims: vec![count],
        bytes,
    };
    let tensors = [
        tensor("f", GgufType::F32, 2, floats.collect()),
        tensor("h", GgufType::F16, 2, vec![0, 0x3c, 0, 0xc0]),
        tensor("q8", GgufType::Q8_0, 32, vec![23; 34]),
        tensor("q4", GgufType::Q4_0, 32, vec![45; 18]),
    ];
    write_gguf(&meta, &tensors)
}

fn open(bytes: &[u8]) -> Result<Gguf, Error> {
    let path = testkit::stage("checkpoint.gguf", bytes);
    let result = Gguf::open(&path);
    std::fs::remove_file(path).unwrap();
    result
}

fn metadata(g: &Gguf) {
    for (ty, expected) in IDS.into_iter().zip(scalars()) {
        assert_eq!(g.meta(&format!("t{ty}")), Some(&expected));
    }
    assert_eq!(g.str("t8"), Some("hello"));
    assert_eq!(g.strs("strings").unwrap(), ["hello"]);
    assert_eq!(g.u32s("integers").unwrap(), [1, 2]);
    assert_eq!(g.u64("positive"), Some(3));
    assert_eq!(g.u64("t5"), None);
    assert_eq!(g.u64("t4"), Some(11));
    assert_eq!(g.u64("t10"), Some(13));
    assert_eq!(g.f32("t12"), Some(2.5));
    assert_eq!(g.f32("t6"), Some(1.5));
}

fn rejected(bytes: &[u8]) {
    assert!(matches!(open(bytes), Err(Error::Format(_))));
}

fn table_at(bytes: &[u8]) -> usize {
    bytes
        .windows(9)
        .position(|w| w == [1, 0, 0, 0, 0, 0, 0, 0, b'f'])
        .unwrap()
}

#[test]
fn reads_gguf() {
    for alignment in [1, 32, 64] {
        let bytes = sample(alignment);
        let g = open(&bytes).unwrap();
        metadata(&g);
        assert_eq!(g.floats("f").unwrap(), [1.25, -2.5]);
        assert_eq!(g.floats("h").unwrap(), [1., -2.]);
        assert_eq!(g.bytes("q8").unwrap(), (GgufType::Q8_0, 32, vec![23; 34]));
        assert_eq!(g.bytes("q4").unwrap(), (GgufType::Q4_0, 32, vec![45; 18]));
        assert_eq!(g.dims("q8").unwrap(), [32]);
        assert!(matches!(g.floats("q4"), Err(Error::DType(_))));
        assert!(matches!(g.elements("absent"), Err(Error::Missing(_))));
        assert!(g.has("q8") && !g.has("absent"));
        let slices = g.slices();
        let names: Vec<&str> = slices.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["f", "h", "q4", "q8"]);
        let q8 = &slices[3];
        assert_eq!(
            (q8.elements(), q8.stored),
            (32, Stored::Form(&codegen::block::Q8_0))
        );
        assert_eq!(Offer::packed(&g, "q8").unwrap(), vec![23; 34]);
        assert!(matches!(Offer::packed(&g, "f"), Err(Error::DType(_))));
        let mut v2 = bytes.clone();
        v2[4] = 2;
        assert!(open(&v2).is_ok());
    }
}

#[test]
fn rejects_damage() {
    let good = sample(64);
    for end in 0..good.len() {
        rejected(&good[..end]);
    }
    for (at, value) in [(0, 0), (4, 4), (8, 255), (24, 255)] {
        let mut bad = good.clone();
        bad[at] = value;
        rejected(&bad);
    }
    for alignment in [0, 3] {
        rejected(&sample(alignment));
    }
    let at = table_at(&good);
    let mut bad = good.clone();
    bad[at + 9..at + 13].copy_from_slice(&u32::MAX.to_le_bytes());
    rejected(&bad);
    let mut bad = good.clone();
    bad[at + 13..at + 21].copy_from_slice(&u64::MAX.to_le_bytes());
    rejected(&bad);
    let mut bad = good.clone();
    bad[at + 25..at + 33].copy_from_slice(&u64::MAX.to_le_bytes());
    rejected(&bad);
    let mut bad = good.clone();
    bad[at + 21..at + 25].copy_from_slice(&99u32.to_le_bytes());
    assert!(matches!(open(&bad), Err(Error::DType(s)) if s.contains("f") && s.contains("99")));
    let mut bad = good.clone();
    bad[at + 8] = 255;
    rejected(&bad);
}
