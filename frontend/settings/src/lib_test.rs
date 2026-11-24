#[test]
fn reads_variables() {
    // SAFETY: this crate has one test, so one thread reads.
    unsafe {
        std::env::set_var("TESSERA_LANES", "8");
        std::env::set_var("TESSERA_EMULATE", "cuda");
        std::env::set_var("TESSERA_BACKEND", "host");
        std::env::set_var("TESSERA_CHECK", "/shared/check");
        std::env::set_var("TESSERA_MEMORY", "24G");
        std::env::set_var("TESSERA_READS", "direct");
        std::env::set_var("TESSERA_KERNEL_CACHE", "/tmp/kcache");
        std::env::set_var("TESSERA_NVRTC", "--maxrregcount=128 -use_fast_math");
        std::env::set_var("TESSERA_DUMP_SITES", "/tmp/sites.jsonl");
    }
    let s = crate::read();
    assert_eq!(s.host_lanes, 8);
    assert_eq!(s.emulate, Some(tessera_engine::Backend::Cuda));
    assert_eq!(s.gpu(), None);
    assert_eq!(s.memory, tessera_engine::Memory::Bytes(24 << 30));
    assert_eq!(s.reads, tessera_engine::Reads::Direct);
    assert_eq!(s.check, Some(std::path::PathBuf::from("/shared/check")));
    assert_eq!(
        s.kernel_cache,
        Some(std::path::PathBuf::from("/tmp/kcache"))
    );
    assert_eq!(s.nvrtc, vec!["--maxrregcount=128", "-use_fast_math"]);
    assert_eq!(
        s.dump_sites,
        Some(std::path::PathBuf::from("/tmp/sites.jsonl"))
    );
}

#[test]
fn selects_proxy() {
    use crate::select;
    let (none, some) = (None, Some("http://proxy:3128".to_string()));
    let no = |names: &str| Some(names.to_string());
    let host = crate::HUB;
    assert_eq!(select(&some, &none, &None, host), Some("http://proxy:3128"));
    assert_eq!(select(&none, &some, &None, host), Some("http://proxy:3128"));
    assert_eq!(select(&none, &none, &None, host), None);
    assert_eq!(select(&some, &none, &no(".co, .org"), host), None);
    assert_eq!(
        select(&some, &none, &no(".org"), host),
        Some("http://proxy:3128")
    );
}
