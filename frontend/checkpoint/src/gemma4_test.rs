use crate::exlr8::Release;
use crate::{File, Offer, Stored};
use tessera_engine::Bf16;

/// v0.2's row trellis rows against the same rows of v0.1.
#[test]
fn row_trellis_table() {
    let Some(dir) = tessera_testkit::reference("gemma4-ple") else {
        return;
    };
    let name = "model.language_model.embed_tokens_per_layer.weight";
    let release = Release::open(&dir);
    let slice = release.slices().into_iter().find(|s| s.name == name);
    let slice = slice.expect("the table is on offer");
    assert_eq!(slice.stored, Stored::BF16);
    let wide = |b: &[u8]| f32::from(Bf16(u16::from_le_bytes([b[0], b[1]])));
    let got: Vec<f32> = Offer::packed(&release, name)
        .unwrap()
        .chunks(2)
        .map(wide)
        .collect();
    let want = File::open(&dir.join("want.safetensors")).unwrap();
    let want: Vec<f32> = want.bytes("want").unwrap().chunks(2).map(wide).collect();
    assert_eq!(got.len(), want.len());
    // K4 codes keep each row near its BF16 source.
    for (g, w) in got.chunks(slice.shape[1]).zip(want.chunks(slice.shape[1])) {
        let error: f32 = g.iter().zip(w).map(|(a, b)| (a - b) * (a - b)).sum();
        let size: f32 = w.iter().map(|b| b * b).sum();
        assert!(
            error < 0.01 * size,
            "relative error {}",
            (error / size).sqrt()
        );
    }
}
