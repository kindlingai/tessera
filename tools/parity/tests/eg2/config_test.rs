use super::release;
use sensai_bundle::Overrides;
use sensai_models::eg2::{Config, program};
use sensai_models::keys::{Keys, Offered};
use sensai_trace::Linear;

#[test]
fn read_matches_release() {
    let Some(dir) = sensai_testkit::reference("eg2") else {
        return;
    };
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let offers = [Offered::parse("config.json", &text).unwrap()];
    let mut keys = Keys::new(&offers);
    let graph = program::<Linear>(&Config::read(&mut keys, 1, 1024));
    let unread = keys.finish().unwrap();
    assert_eq!(graph.to_json(), release().to_json());
    println!("{}", unread.join("\n"));
}

#[test]
fn reports_each_failure() {
    let json = r#"{"text_config": {"vocab_size": "big", "rope_theta": 1.0,
        "rope_parameters": {"full_attention": {"rope_theta": 2.0}}}}"#;
    let offers = [Offered::parse("config.json", json).unwrap()];
    let mut keys = Keys::new(&offers);
    Config::read(&mut keys, 1, 16);
    let failed = keys.finish().unwrap_err();
    assert!(failed.missing.contains(&"text_config.hidden_size".into()));
    assert_eq!(failed.misfit[0].0, "text_config.vocab_size");
    let full = "text_config.rope_parameters.full_attention.rope_theta";
    assert_eq!(failed.ambiguous[0].0, full);
}

#[test]
fn recipe_beats_file() {
    let path = sensai_testkit::stage("config.json", br#"{"window": 512, "eps": 1e-6}"#);
    let table = serde_json::json!({"config.json": {"window": 64}});
    let config = Overrides::new(table.as_object().unwrap().clone());
    let read = |k: &mut Keys| [k.need::<f32>(&["window"]), k.need(&["eps"])];
    let got = config.read("config.json", Some(&path), read).unwrap();
    assert_eq!(got, [64.0, 1e-6]);
}
