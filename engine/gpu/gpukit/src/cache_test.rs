use crate::cache::DiskCache;

#[test]
fn caches_compiled_images() {
    let dir = tessera_testkit::scratch(&format!("gpukit-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let plain = DiskCache::new(Some(dir.clone()), Vec::new());
    let base = ["--gpu-architecture=compute_121", "--std=c++17"];
    let mut builds = 0usize;
    let first = plain
        .load_or::<()>("cuda", "13.1", "compute_121", "ptx", &base, "ka", |opts| {
            builds += 1;
            assert_eq!(opts, base);
            Ok(b"PTX_A".to_vec())
        })
        .unwrap();
    let second = plain
        .load_or::<()>("cuda", "13.1", "compute_121", "ptx", &base, "ka", |_| {
            builds += 1;
            Ok(b"PTX_STALE".to_vec())
        })
        .unwrap();
    assert_eq!(
        (first, second, builds),
        (b"PTX_A".to_vec(), b"PTX_A".to_vec(), 1)
    );
    assert_ne!(
        plain.key("cuda", "13.1", "compute_121", &base, "ka"),
        plain.key("cuda", "13.2", "compute_121", &base, "ka")
    );
    let tuned = DiskCache::new(Some(dir.clone()), vec!["--maxrregcount=128".into()]);
    assert_ne!(
        plain.key("cuda", "13.1", "compute_121", &base, "ka"),
        tuned.key("cuda", "13.1", "compute_121", &base, "ka")
    );
    let fast = tuned
        .load_or::<()>("cuda", "13.1", "compute_121", "ptx", &base, "ka", |opts| {
            builds += 1;
            assert_eq!(opts.last(), Some(&"--maxrregcount=128"));
            Ok(b"PTX_TUNED".to_vec())
        })
        .unwrap();
    assert_eq!((fast, builds), (b"PTX_TUNED".to_vec(), 2));
    let entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(entries.len() == 2 && entries.iter().all(|n| n.ends_with(".ptx")));
    let _ = std::fs::remove_dir_all(&dir);
}
