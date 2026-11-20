use super::Embedder;
use sensai_bundle::Overrides;
use sensai_engine::{Pick, Settings};
use sensai_testkit::cosine_f32;

/// Each eg2 case through the exlr8 release, against eg2cpu.
#[test]
fn eg2_cases() {
    let dir = sensai_testkit::reference("eg2");
    let Some(dir) = dir.filter(|d| d.join("exlr8/tokenizer.json").is_file()) else {
        return eprintln!("no eg2 reference with its exlr8 release: skipped");
    };
    let mut settings = Settings::default();
    // The host takes minutes on the long case.
    settings.backend = sensai_engine::gpu().map_or(settings.backend, Pick::Gpu);
    let mut embedder =
        Embedder::open(&dir.join("exlr8"), &settings, &Overrides::default()).unwrap();
    for entry in std::fs::read_dir(dir.join("cases")).unwrap() {
        let case = entry.unwrap().path();
        let ids = std::fs::read_to_string(case.join("tokens.txt")).unwrap();
        let ids: Vec<u32> = ids.split_whitespace().map(|t| t.parse().unwrap()).collect();
        if let Ok(text) = std::fs::read_to_string(case.join("text.txt")) {
            assert_eq!(embedder.tokens(&text).unwrap(), ids, "{case:?}");
        }
        let bytes = std::fs::read(case.join("embedding.f32")).unwrap();
        let want = bytes.as_chunks().0.iter().map(|b| f32::from_le_bytes(*b));
        let got = embedder.embed_ids(&[ids]).unwrap();
        let cos = cosine_f32(&got[0], &want.collect::<Vec<_>>());
        eprintln!("{case:?} on {}: cos {cos:.5}", embedder.device());
        // The golden test's floor for the exlr8 release.
        assert!(cos > 0.98, "{case:?}: cos {cos}");
    }
}
