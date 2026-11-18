//! The sprite denoiser and CLIP's text tower against torch.

use crate::sprite::{denoiser, text_tower};
use sensai_checkpoint::File;
use sensai_engine::{Graph, Model, Pick, Recipe, Settings, Tensor, Weight};
use sensai_testkit::cosine_f32;
use std::path::Path;

const TEXT: usize = 24;
const WIDTH: usize = 512;
const PIXELS: usize = 256;
const MASK: u32 = 512;

fn ints(file: &File, name: &str) -> Vec<u32> {
    let bytes = file.bytes(name).unwrap();
    let words = bytes.as_chunks::<4>().0.iter();
    words.map(|b| i32::from_le_bytes(*b) as u32).collect()
}

/// Native CPU kernels: lowered, and quicker than the host emulator.
fn run(g: Graph, w: &[Weight], inputs: &[Tensor]) -> Vec<f32> {
    let settings = Settings {
        backend: Pick::Cpu,
        ..Settings::default()
    };
    let mut m = Model::open(g, &Recipe::default(), &settings).unwrap();
    m.load(w).unwrap();
    m.run(inputs).unwrap()[0].floats().unwrap().to_vec()
}

fn close(name: &str, got: &[f32], want: &[f32]) {
    let worst = (got.iter().zip(want)).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let cos = cosine_f32(got, want);
    eprintln!("{name}: cos {cos:.6}, worst {worst:.2e}");
    assert!(cos > 0.9999, "{name}: cos {cos}");
}

/// CLIP's states for the reference prompts.
fn states(clip: &Path, r: &File) -> Vec<f32> {
    let c = text_tower(clip, 2, TEXT as u64).unwrap();
    let ids = ints(r, "clip.ids");
    for (row, text) in ids
        .chunks(TEXT)
        .zip(["cat wearing sunglasses", "slice of pizza"])
    {
        let own = c.tokenizer.encode_special(text).unwrap();
        assert_eq!(own, row[..own.len()], "{text}");
    }
    let positions = (0..2).flat_map(|_| 0..TEXT as u32).collect();
    let inputs = [
        Tensor::u32(vec![2, TEXT], ids),
        Tensor::u32(vec![2, TEXT], positions),
    ];
    let got = run(c.graph, &c.weights, &inputs);
    close("clip states", &got, &r.floats("clip.states").unwrap());
    got
}

#[test]
fn matches_torch() {
    let dir = sensai_testkit::reference("sprite");
    let Some(dir) = dir.filter(|d| d.join("model_config.json").is_file()) else {
        return eprintln!("no sprite reference here; skipped");
    };
    let r = File::open(&dir.join("ref.safetensors")).unwrap();
    let states = states(&dir.join("clip"), &r);
    let d = denoiser(&dir, 4, None).unwrap();
    let canvas = ints(&r, "den.canvas");
    let masked = |c: &[u32]| c.iter().filter(|&&t| t == MASK).count() as u32;
    let counts: Vec<u32> = canvas.chunks(PIXELS).map(masked).collect();
    let guide = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];
    let inputs = [
        Tensor::u32(vec![4, PIXELS], canvas.repeat(2)),
        Tensor::u32(vec![4], counts.repeat(2)),
        Tensor::f32(vec![4, TEXT, WIDTH], states.repeat(2)),
        Tensor::f32(vec![4, WIDTH], r.floats("clip.pooled").unwrap().repeat(2)),
        Tensor::f32(vec![4, 2], guide.concat()),
        Tensor::f32(vec![4, PIXELS, 512], r.floats("den.sc").unwrap().repeat(2)),
    ];
    let logits = run(d.graph, &d.weights, &inputs);
    close("denoiser logits", &logits, &r.floats("den.logits").unwrap());
}
