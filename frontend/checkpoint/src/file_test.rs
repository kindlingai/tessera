use crate::{Error, File, Offer, Stored};
use std::path::PathBuf;
use tessera_engine::Reads;

/// Writes a header and tensor bytes to a temporary file.
fn write(name: &str, header: &str, data: &[u8]) -> PathBuf {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    tessera_testkit::stage(&format!("{name}.safetensors"), &bytes)
}

const HEADER: &str = r#"{"__metadata__":{"format":"pt"},
"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
"b":{"dtype":"BF16","shape":[1,2],"data_offsets":[8,12]},
"c":{"dtype":"I64","shape":[1],"data_offsets":[12,20]}}"#;

#[test]
fn widens_to_f32() {
    let mut data = Vec::new();
    data.extend(1.5f32.to_le_bytes());
    data.extend((-2.0f32).to_le_bytes());
    data.extend([0x80, 0x3f, 0x00, 0xbf]);
    data.extend(7i64.to_le_bytes());
    let path = write("widen", HEADER, &data);
    let f = File::open(&path).unwrap();
    assert_eq!(f.floats("a").unwrap(), [1.5, -2.0]);
    assert_eq!(f.floats("b").unwrap(), [1.0, -0.5]);
    assert_eq!(f.shape("b"), Some(&[1, 2][..]));
    assert!(matches!(f.floats("c"), Err(Error::DType(_))));
    assert!(matches!(f.floats("d"), Err(Error::Missing(_))));
    let stored: Vec<Stored> = f.slices().iter().map(|s| s.stored).collect();
    assert_eq!(stored, [Stored::F32, Stored::BF16, Stored::Other]);
    assert!(matches!(Offer::packed(&f, "a"), Err(Error::DType(_))));
    assert_eq!(Offer::packed(&f, "b").unwrap(), [0x80, 0x3f, 0x00, 0xbf]);
    assert_eq!(
        Offer::packed_rows(&f, "b", (0, 1)).unwrap(),
        [0x80, 0x3f, 0x00, 0xbf]
    );
    let opened = crate::open(&path, None).unwrap();
    assert_eq!(opened.packed("b").unwrap(), [0x80, 0x3f, 0x00, 0xbf]);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn refuses_short_files() {
    let path = write("short", HEADER, &[0; 8]);
    assert!(matches!(File::open(&path), Err(Error::Format(_))));
    std::fs::remove_file(&path).unwrap();
    std::fs::write(&path, 99u64.to_le_bytes()).unwrap();
    assert!(matches!(File::open(&path), Err(Error::Format(_))));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn widens_f16() {
    let header = r#"{"h":{"dtype":"F16","shape":[3],"data_offsets":[0,6]}}"#;
    let data: Vec<u8> = [0x3c00u16, 0x2e66, 0xc280]
        .iter()
        .flat_map(|h| h.to_le_bytes())
        .collect();
    let path = write("f16", header, &data);
    let f = File::open(&path).unwrap();
    // 0.1 is not exact in f16.
    assert_eq!(f.floats("h").unwrap(), [1.0, 0.099_975_586, -3.25]);
    assert_eq!(f.slices()[0].stored, Stored::F16);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn reads_filed_spans() {
    // Two reader threads, starting off a block boundary.
    let bytes = 10 << 20;
    let header = format!(
        r#"{{"w":{{"dtype":"BF16","shape":[{}],"data_offsets":[2,{}]}}}}"#,
        bytes / 2,
        2 + bytes
    );
    let data: Vec<u8> = (0..2 + bytes).map(|i| (i % 251) as u8).collect();
    let path = write("filed", &header, &data);
    assert!(crate::disk::is_local(&std::fs::File::open(&path).unwrap()));
    let span = File::open(&path).unwrap().filed("w").unwrap();
    for reads in [Reads::Cached, Reads::Direct] {
        let mut got = vec![0u8; span.bytes()];
        span.read(&mut got, reads).unwrap();
        assert!(got == data[2..], "{reads:?} reads differ");
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn widens_large_tensors() {
    use tessera_engine::{Bf16, F16};

    let n = 200_003usize;
    let bf_bits: Vec<u16> = (0..n)
        .map(|i| ((i * 257) as u16) & 0x7f7f | ((i as u16 & 1) << 15))
        .collect();
    let f16_bits: Vec<u16> = (0..n)
        .map(|i| ((i * 193) as u16) & 0x7bff | ((i as u16 & 1) << 15))
        .collect();
    let f32_vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.125 - 10_000.0).collect();

    let mut data = Vec::with_capacity(n * 8);
    for &b in &bf_bits {
        data.extend(b.to_le_bytes());
    }
    let f16_at = data.len();
    for &h in &f16_bits {
        data.extend(h.to_le_bytes());
    }
    let f32_at = data.len();
    for &x in &f32_vals {
        data.extend(x.to_le_bytes());
    }
    let end = data.len();
    let header = format!(
        r#"{{"bf":{{"dtype":"BF16","shape":[{n}],"data_offsets":[0,{f16_at}]}},"h":{{"dtype":"F16","shape":[{n}],"data_offsets":[{f16_at},{f32_at}]}},"f":{{"dtype":"F32","shape":[{n}],"data_offsets":[{f32_at},{end}]}}}}"#
    );
    let path = write("large_floats", &header, &data);
    let f = File::open(&path).unwrap();

    let got_bf = f.floats("bf").unwrap();
    for (g, &b) in got_bf.iter().zip(&bf_bits) {
        assert_eq!(g.to_bits(), f32::from(Bf16(b)).to_bits());
    }
    let got_f16 = f.floats("h").unwrap();
    for (g, &h) in got_f16.iter().zip(&f16_bits) {
        assert_eq!(g.to_bits(), f32::from(F16(h)).to_bits());
    }
    let got_f32 = f.floats("f").unwrap();
    assert_eq!(got_f32, f32_vals);
    std::fs::remove_file(path).unwrap();
}
