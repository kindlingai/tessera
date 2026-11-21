//! Qwen3.5 pieces against tiny random Hugging Face modules.

mod attention_test;
mod deltanet_test;
mod model_test;

use std::path::{Path, PathBuf};
use tessera_checkpoint::File;
use tessera_interp::{Tensor, run};
use tessera_lower::{Device, lower};
use tessera_models::Stack;
use tessera_models::qwen35::{Config, Mixing, Recurrent, program};
use tessera_parity::{Spec, loaded, suite};
use tessera_run::testing::runs;
use tessera_testkit::max_abs_diff;
use tessera_trace::Composite::{Layer, Mlp, Model};
use tessera_trace::DimName::{Context, Seqs};
use tessera_trace::{Graph, PathSeg};

/// The fixture's sequences and their length.
const SEQS: usize = 2;
const LEN: usize = 12;

/// The tiny model's sizes.
fn tiny() -> Config {
    Config {
        stack: Stack::sized([64, 32, 4, 4, 16, 48], SEQS as u64, 16),
        kv_heads: 2,
        rotary: 8,
        rope_theta: 100.0,
        eps: 1e-6,
        linear: Recurrent {
            key_heads: 2,
            value_heads: 4,
            key_dim: 8,
            value_dim: 6,
            window: 4,
        },
        layers: vec![Mixing::Linear, Mixing::Linear, Mixing::Linear, Mixing::Full],
        tied: false,
    }
}

/// Clef's sizes.
fn clef(max_seqs: u64, max_context: u64) -> Config {
    let full = |i| match i % 4 {
        3 => Mixing::Full,
        _ => Mixing::Linear,
    };
    Config {
        stack: Stack::sized([248_320, 5120, 64, 24, 256, 17_408], max_seqs, max_context),
        kv_heads: 4,
        rotary: 64,
        rope_theta: 1e7,
        eps: 1e-6,
        linear: Recurrent {
            key_heads: 16,
            value_heads: 48,
            key_dim: 128,
            value_dim: 128,
            window: 4,
        },
        layers: (0..64).map(full).collect(),
        tied: false,
    }
}

fn release() -> Graph {
    program(&clef(1, 1024))
}

const SPEC: Spec = Spec {
    name: "qwen35",
    release,
    parameters: 26_895_998_464,
    regions: &[(Layer, 64), (Mlp, 64), (Model, 1)],
    small: Some(|| program(&tessera_models::small::qwen35())),
    stepper: None,
    files: &[(include_str!("manifests/hf.json"), release)],
    golden: None,
};

suite!(SPEC);

/// The fixture for `part`, if this host has it.
fn part(name: &str) -> Option<PathBuf> {
    let dir = tessera_testkit::reference("qwen35-tiny")?.join(name);
    dir.is_dir().then_some(dir)
}

/// `<name>.f32` in `dir`: little-endian floats.
fn read(dir: &Path, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(format!("{name}.f32"))).unwrap();
    let wide = |b: &[u8; 4]| f32::from_le_bytes(*b);
    bytes.as_chunks::<4>().0.iter().map(wide).collect()
}

/// `<name>.f32` in `dir`, as a tensor of `shape`.
fn floats(dir: &Path, name: &str, shape: Vec<usize>) -> Tensor {
    Tensor::f32(shape, read(dir, name))
}

/// Each sequence's positions, from zero.
fn positions() -> Tensor {
    let at = (0..SEQS * LEN).map(|i| (i % LEN) as u32).collect();
    Tensor::u32(vec![SEQS, LEN], at)
}

/// `g`'s weights, each named by its dotted path.
fn weights(g: &Graph, dir: &Path) -> Vec<Tensor> {
    let file = File::open(&dir.join("weights.safetensors")).unwrap();
    let mut out = Vec::new();
    for w in &g.weights {
        let axis = |e: &tessera_trace::DimExpr| e.value(&g.dims, &|_| None).unwrap() as usize;
        let shape = w.ty.shape.0.iter().map(axis).collect();
        let name = PathSeg::dotted(&w.path);
        out.push(Tensor::f32(shape, file.floats(&name).unwrap()));
    }
    out
}

/// `g` on the interpreter and each backend, against `want`.
fn check(g: &Graph, dir: &Path, inputs: &[Tensor], want: &str) {
    let weights = weights(g, dir);
    let want = &read(dir, want)[..];
    let got = run(g, &weights, 0, inputs).unwrap().remove(0);
    let off = max_abs_diff(got.floats().unwrap(), want);
    assert!(off < 1e-5, "interpreter: off by {off:e}");
    let sizes = [(Seqs, SEQS as u64), (Context, LEN as u64)];
    for settings in runs() {
        let plan = lower(g, 0, &sizes, &Device::guess(&settings)).unwrap();
        let mut runner = loaded(plan, g, &weights, &settings);
        let views: Vec<_> = inputs.iter().map(|t| t.view()).collect();
        let got = runner.run(&views).unwrap().remove(0);
        let off = max_abs_diff(got.floats().unwrap(), want);
        assert!(off < 1e-4, "{}: off by {off:e}", runner.device());
    }
}
