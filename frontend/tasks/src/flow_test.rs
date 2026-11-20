use crate::flow::Flow;
use sensai_bundle::Overrides;
use sensai_checkpoint::File;
use sensai_engine::{Pick, Settings};
use sensai_testkit::cosine_f32;

#[test]
fn matches_torch() {
    let Some(dir) = sensai_testkit::reference("flux2-klein") else {
        return eprintln!("no flux2-klein reference here; skipped");
    };
    let r = File::open(&dir.join("ref.safetensors")).unwrap();
    let s = Settings {
        backend: Pick::Cpu,
        ..Settings::default()
    };
    let mut flow = Flow::open(&dir.join("model"), &s, [1, 64, 256], &Overrides::default()).unwrap();
    let [latents, text, want] = ["latents", "text", "velocity"].map(|n| r.floats(n).unwrap());
    let got = flow.velocity(&latents, &text, 16, 0.7).unwrap();
    let cos = cosine_f32(&got, &want);
    assert!(cos > 0.9999, "cos {cos}");
}
