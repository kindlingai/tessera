use tessera_bundle::Overrides;
use tessera_models::decoder::{Config, Swiglu, stepper};
use tessera_trace::Norm;

#[test]
fn read_matches_release() {
    let Some(dir) = tessera_testkit::reference("qwen3-taps") else {
        return;
    };
    let path = dir.join("model/config.json");
    let none = Overrides::default();
    let config = none.read("config.json", Some(&path), |k| Config::read(k, 1, 1024));
    let want = stepper::<Norm, Swiglu>(&Config::qwen3(1, 1024));
    assert_eq!(
        stepper::<Norm, Swiglu>(&config.unwrap()).to_json(),
        want.to_json()
    );
}
