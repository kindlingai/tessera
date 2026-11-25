use crate::flow::Flow;
use tessera_bundle::Overrides;
use tessera_checkpoint::File;
use tessera_engine::{Pick, Settings};
use tessera_testkit::cosine_f32;

#[test]
fn matches_torch() {
    let Some(dir) = tessera_testkit::reference("flux2-klein") else {
        return eprintln!("no flux2-klein reference here; skipped");
    };
    let r = File::open(&dir.join("ref.safetensors")).unwrap();
    let s = Settings {
        backend: Pick::Cpu,
        ..Settings::default()
    };
    let q = &Default::default();
    let mut flow = Flow::open(
        &dir.join("model"),
        &s,
        [1, 64, 256],
        &Overrides::default(),
        q,
    )
    .unwrap();
    let [latents, text, want] = ["latents", "text", "velocity"].map(|n| r.floats(n).unwrap());
    let got = flow.velocity(&latents, &text, 16, 0.7).unwrap();
    let cos = cosine_f32(&got, &want);
    assert!(cos > 0.9999, "cos {cos}");
}
