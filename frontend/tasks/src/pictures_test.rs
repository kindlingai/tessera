use crate::pictures::Pictures;
use sensai_bundle::Overrides;
use sensai_checkpoint::File;
use sensai_engine::{Pick, Settings};

#[test]
fn matches_torch() {
    let Some(dir) = sensai_testkit::reference("flux2-klein") else {
        return eprintln!("no flux2-klein reference here; skipped");
    };
    let r = File::open(&dir.join("decode.safetensors")).unwrap();
    let s = Settings {
        backend: Pick::Cpu,
        ..Settings::default()
    };
    let mut p = Pictures::open(&dir.join("model"), &s, [8, 1], &Overrides::default()).unwrap();
    let got = p.decode(&r.floats("tokens").unwrap()).unwrap().remove(0);
    // Torch's planes `[3, H, W]` against our RGBA rows.
    let (want, px) = (r.floats("image").unwrap(), got.width * got.height);
    let byte = |v: f32| ((v / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as i32;
    let off = (0..px * 3)
        .map(|i| (got.rgba[i / 3 * 4 + i % 3] as i32 - byte(want[i % 3 * px + i / 3])).abs());
    assert!(
        off.max().unwrap() <= 1,
        "pixels differ by more than one level"
    );
}
