use crate::cache::{Cache, key};

#[test]
fn cache_round_trips() {
    let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/target/scratch"));
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(format!("simcheck-{}.bin", std::process::id()));
    let mut c = Cache::default();
    c.insert(key("a"), vec![1.0, 2.0]);
    c.insert(key("b"), vec![3.0, 4.0]);
    c.save(&path, &[key("a")]).unwrap();
    let back = Cache::load(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(back.get(key("a")), Some(&vec![1.0, 2.0]));
    assert!(back.get(key("b")).is_none() && key("a") != key("b"));
    assert!(Cache::load(&path).unwrap().get(key("a")).is_none());
}
