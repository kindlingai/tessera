use super::release;
use tessera_models::gemma4::{Config, program};
use tessera_models::keys::{Keys, Offered};
use tessera_trace::Linear;

#[test]
fn read_matches_release() {
    let Some(dir) = tessera_testkit::reference("gemma4") else {
        return;
    };
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let offers = [Offered::parse("config.json", &text).unwrap()];
    let mut keys = Keys::new(&offers);
    let graph = program::<Linear>(&Config::read(&mut keys, 1, 1024));
    keys.finish().unwrap();
    assert_eq!(graph.to_json(), release().to_json());
}
