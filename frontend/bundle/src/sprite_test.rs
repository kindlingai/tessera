//! The sprite denoiser and CLIP's text tower against torch.

use crate::sprite::{denoiser, denoiser_offer, text_tower};
use sensai_checkpoint::{File, Offer, Torch};
use sensai_engine::sprite::Config;
use sensai_engine::{Graph, Model, Pick, Recipe, Settings, Tensor, Weight};
use sensai_testkit::cosine_f32;
use std::path::{Path, PathBuf};

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
    let Some(dir) = sensai_testkit::reference("sprite") else {
        return eprintln!("no sprite reference here; skipped");
    };
    let r = File::open(&dir.join("ref.safetensors")).unwrap();
    let states = states(&dir.join("clip"), &r);
    let offer = File::open(&dir.join("sprite.safetensors")).unwrap();
    let d = denoiser(&offer, &Config::small(4)).unwrap();
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

/// `want`'s tensors in `got`, within `tol` of their largest value.
fn same(got: &dyn Offer, want: &File, tol: f32) {
    let shapes = |o: &dyn Offer| {
        let mut s: Vec<_> = o.slices().into_iter().map(|s| (s.name, s.shape)).collect();
        s.sort();
        s
    };
    let offered = shapes(got);
    for want in shapes(want) {
        assert!(offered.contains(&want), "{want:?}");
    }
    for slice in want.slices() {
        let (g, w) = (
            got.floats(&slice.name).unwrap(),
            want.floats(&slice.name).unwrap(),
        );
        assert_eq!(g.len(), w.len(), "{}", slice.name);
        let scale = w.iter().fold(1.0f32, |m, v| m.max(v.abs()));
        let worst = (g.iter().zip(&w)).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst <= tol * scale, "{}: worst {worst:e}", slice.name);
    }
}

/// The cached snapshot `refs/main` names for `repo`.
fn snapshot(repo: &str) -> Option<PathBuf> {
    let hub = sensai_frontend_settings::read().hub?;
    let dir = hub.join(format!("models--{}", repo.replace('/', "--")));
    let main = std::fs::read_to_string(dir.join("refs/main")).ok()?;
    Some(dir.join("snapshots").join(main.trim()))
}

#[test]
fn upstream_matches_converted() {
    let (Some(dir), Some(snap), Some(clip)) = (
        sensai_testkit::reference("sprite"),
        snapshot("thejagstudio/tessera-small"),
        snapshot("openai/clip-vit-base-patch32"),
    ) else {
        return eprintln!("no sprite checkpoints here; skipped");
    };
    let pt = denoiser_offer(&snap.join("tessera-small-fp16.pt"), &Config::small(1)).unwrap();
    let converted = File::open(&dir.join("sprite.safetensors")).unwrap();
    same(&*pt, &converted, 1e-5);
    let torch = Torch::open(&clip.join("pytorch_model.bin")).unwrap();
    same(
        &torch,
        &File::open(&dir.join("clip/model.safetensors")).unwrap(),
        0.0,
    );
}
