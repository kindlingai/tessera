//! Qwen3.5 pieces against tiny random Hugging Face modules.

mod attention_test;

use std::path::{Path, PathBuf};
use tessera_checkpoint::File;
use tessera_interp::{Tensor, run};
use tessera_lower::{Device, lower};
use tessera_models::Stack;
use tessera_models::qwen35::Config;
use tessera_parity::{loaded, pulled};
use tessera_run::testing::runs;
use tessera_testkit::max_abs_diff;
use tessera_trace::DimName::{Context, Seqs};
use tessera_trace::Graph;

/// The fixture's sequences and their length.
const SEQS: usize = 2;
const LEN: usize = 7;

/// The tiny model's sizes.
fn tiny() -> Config {
    Config {
        stack: Stack::sized([64, 32, 4, 4, 16, 48], SEQS as u64, 8),
        kv_heads: 2,
        rotary: 4,
        rope_theta: 1e7,
        eps: 1e-6,
    }
}

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

/// `g` on the interpreter and each backend, against `want`.
fn check(g: &Graph, dir: &Path, inputs: &[Tensor], want: &str) {
    let file = File::open(&dir.join("weights.safetensors")).unwrap();
    let weights = pulled("qwen35-tiny", g, &file);
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
