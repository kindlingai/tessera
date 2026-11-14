use crate::{Error, Gguf, GgufType, Meta, Offer, Stored};
use std::path::PathBuf;

fn string(out: &mut Vec<u8>, text: &str) {
    out.extend((text.len() as u64).to_le_bytes());
    out.extend(text.as_bytes());
}

fn field(out: &mut Vec<u8>, key: &str, ty: u32, bytes: &[u8]) {
    string(out, key);
    out.extend(ty.to_le_bytes());
    out.extend(bytes);
}

fn sample(alignment: u64) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(4u64.to_le_bytes());
    out.extend(16u64.to_le_bytes());
    let values: &[(u32, &[u8])] = &[
        (0, &[255]),
        (1, &[254]),
        (2, &9u16.to_le_bytes()),
        (3, &(-10i16).to_le_bytes()),
        (4, &11u32.to_le_bytes()),
        (5, &(-12i32).to_le_bytes()),
        (6, &1.5f32.to_le_bytes()),
        (7, &[1]),
        (10, &13u64.to_le_bytes()),
        (11, &(-14i64).to_le_bytes()),
        (12, &2.5f64.to_le_bytes()),
    ];
    for (ty, bytes) in values {
        field(&mut out, &format!("t{ty}"), *ty, bytes);
    }
    let mut text = Vec::new();
    string(&mut text, "hello");
    field(&mut out, "t8", 8, &text);
    let mut array = 8u32.to_le_bytes().to_vec();
    array.extend(1u64.to_le_bytes());
    array.extend(&text);
    field(&mut out, "strings", 9, &array);
    let mut ints = 5u32.to_le_bytes().to_vec();
    ints.extend(2u64.to_le_bytes());
    ints.extend(1i32.to_le_bytes());
    ints.extend(2i32.to_le_bytes());
    field(&mut out, "integers", 9, &ints);
    field(&mut out, "positive", 11, &3i64.to_le_bytes());
    field(&mut out, "general.alignment", 10, &alignment.to_le_bytes());
    for (name, count, ty, offset) in [
        ("f", 2u64, 0u32, 0u64),
        ("h", 2, 1, 64),
        ("q8", 32, 8, 128),
        ("q4", 32, 2, 192),
    ] {
        string(&mut out, name);
        out.extend(1u32.to_le_bytes());
        out.extend(count.to_le_bytes());
        out.extend(ty.to_le_bytes());
        out.extend(offset.to_le_bytes());
    }
    out.resize(out.len().next_multiple_of(alignment.max(1) as usize), 0);
    let mut data = vec![0; 210];
    data[..4].copy_from_slice(&1.25f32.to_le_bytes());
    data[4..8].copy_from_slice(&(-2.5f32).to_le_bytes());
    data[64..68].copy_from_slice(&[0, 0x3c, 0, 0xc0]);
    data[128..162].fill(23);
    data[192..].fill(45);
    out.extend(data);
    out
}

fn write(bytes: &[u8]) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/scratch"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("checkpoint-gguf-{}-{id}", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path
}

fn open(bytes: &[u8]) -> Result<Gguf, Error> {
    let path = write(bytes);
    let result = Gguf::open(&path);
    std::fs::remove_file(path).unwrap();
    result
}

fn metadata(g: &Gguf) {
    let values = [
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
    ];
    for (ty, expected) in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12].into_iter().zip(values) {
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
