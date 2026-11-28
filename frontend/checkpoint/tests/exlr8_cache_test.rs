//! Tests `exlr8::Release` slice list caching and invalidation.

use std::path::Path;
use tessera_checkpoint::Offer;
use tessera_checkpoint::exlr8::Release;

fn write_f32(path: &Path, name: &str, values: &[f32]) {
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let header = format!(
        r#"{{"{name}":{{"dtype":"F32","shape":[{}],"data_offsets":[0,{}]}}}}"#,
        values.len(),
        data.len()
    );
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.extend(data);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn slices_cache_invalidates() {
    let dir = tessera_testkit::scratch(&format!("{}-exlr8-cache", std::process::id()));
    write_f32(
        &dir.join("dense/exlr8-vocab.safetensors"),
        "norm.weight",
        &[1.0, 2.0],
    );
    write_f32(
        &dir.join("dense/exlr8-dense-000.safetensors"),
        "a.weight",
        &[1.0, 2.0],
    );
    let release = Release::open(&dir);
    let first = release.slices();
    write_f32(
        &dir.join("dense/exlr8-dense-001.safetensors"),
        "b.weight",
        &[3.0, 4.0],
    );
    let second = release.slices();
    assert_eq!(first.len(), second.len());
    assert!(second.iter().all(|s| s.name != "model.layers.1.b.weight"));
    let release = release.width(2);
    let refreshed = release.slices();
    assert!(
        refreshed
            .iter()
            .any(|s| s.name == "model.layers.1.b.weight")
    );
    std::fs::remove_dir_all(dir).unwrap();
}
