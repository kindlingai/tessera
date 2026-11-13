use crate::{Error, File};
use std::path::PathBuf;

/// Writes a header and tensor bytes to a temporary file.
fn write(name: &str, header: &str, data: &[u8]) -> PathBuf {
    let id = std::process::id();
    let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/scratch"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("tessera-{name}-{id}.safetensors"));
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    std::fs::write(&path, bytes).unwrap();
    path
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
