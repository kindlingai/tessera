use crate::features::Features;
use tessera_checkpoint::File;
use tessera_engine::{Pick, Settings};

#[test]
fn matches_torch() {
    let Some(dir) = tessera_testkit::reference("qwen3-taps") else {
        return eprintln!("no qwen3-taps reference here; skipped");
    };
    let r = File::open(&dir.join("ref.safetensors")).unwrap();
    let s = Settings {
        backend: Pick::Cpu,
        ..Settings::default()
    };
    // The layers FLUX.2 klein reads.
    let m = dir.join("model");
    let (c, p) = (Default::default(), Default::default());
    let mut f = Features::open([&m; 2], &s, &[9, 18, 27], [2, 64], &c, &p).unwrap();
    let got = f.encode(&["a yellow rubber duck", "a red potion on a table"]);
    let (got, want) = (got.unwrap().concat(), r.floats("embeds").unwrap());
    // Each row by its own largest value: outliers reach thousands.
    for (g, w) in got
        .chunks(want.len() / 128)
        .zip(want.chunks(want.len() / 128))
    {
        let scale = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = (g.iter().zip(w)).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-4 * scale, "off {worst} of {scale}");
    }
}
