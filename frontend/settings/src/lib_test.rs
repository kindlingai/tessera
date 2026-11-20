#[test]
fn reads_variables() {
    // SAFETY: this crate has one test, so one thread reads.
    unsafe {
        std::env::set_var("TESSERA_LANES", "8");
        std::env::set_var("TESSERA_EMULATE", "cuda");
        std::env::set_var("TESSERA_BACKEND", "host");
        std::env::set_var("TESSERA_CHECK", "/shared/check");
        std::env::set_var("TESSERA_MEMORY", "24G");
    }
    let s = crate::read();
    assert_eq!(s.host_lanes, 8);
    assert_eq!(s.emulate, Some(sensai_engine::Backend::Cuda));
    assert_eq!(s.gpu(), None);
    assert_eq!(s.memory, sensai_engine::Memory::Bytes(24 << 30));
    assert_eq!(s.check, Some(std::path::PathBuf::from("/shared/check")));
}

#[test]
fn selects_proxy() {
    use crate::select;
    let (none, some) = (None, Some("http://proxy:3128".to_string()));
    let no = |names: &str| Some(names.to_string());
    let host = "huggingface.co";
    assert_eq!(select(&some, &none, &None, host), Some("http://proxy:3128"));
    assert_eq!(select(&none, &some, &None, host), Some("http://proxy:3128"));
    assert_eq!(select(&none, &none, &None, host), None);
    assert_eq!(select(&some, &none, &no(".co, .org"), host), None);
    assert_eq!(
        select(&some, &none, &no(".org"), host),
        Some("http://proxy:3128")
    );
}
