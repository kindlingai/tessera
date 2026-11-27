use crate::cache::{CacheKey, DiskCache};

#[test]
fn caches_compiled_images() {
    let dir = tessera_testkit::scratch(&format!("gpukit-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let plain = DiskCache::new(Some(dir.clone()), Vec::new());
    let base = ["--gpu-architecture=compute_121", "--std=c++17"];
    let make = |compiler, unit| CacheKey {
        backend: "cuda",
        compiler,
        arch: "compute_121",
        unit,
        base: &base,
        source: "ka",
    };
    let key = make("13.1", "a.cu");
    let mut builds = 0usize;
    let build = |c: &DiskCache, k: CacheKey<'_>, out: &[u8], n: &mut usize| {
        c.load_or::<()>(k, "ptx", |_| {
            *n += 1;
            Ok(out.to_vec())
        })
        .unwrap()
    };
    assert_eq!(build(&plain, key, b"PTX_A", &mut builds), b"PTX_A");
    assert_eq!(build(&plain, key, b"PTX_STALE", &mut builds), b"PTX_A");
    assert_eq!(builds, 1);
    assert_ne!(plain.key(key), plain.key(make("13.2", "a.cu")));
    let other = make("13.1", "b.cu");
    assert_ne!(plain.key(key), plain.key(other));
    assert_eq!(build(&plain, other, b"PTX_B", &mut builds), b"PTX_B");
    assert_eq!(builds, 2);
    let tuned = DiskCache::new(Some(dir.clone()), vec!["--maxrregcount=128".into()]);
    assert_ne!(plain.key(key), tuned.key(key));
    assert_eq!(build(&tuned, key, b"PTX_TUNED", &mut builds), b"PTX_TUNED");
    assert_eq!(builds, 3);
    let _ = std::fs::remove_dir_all(&dir);
}
