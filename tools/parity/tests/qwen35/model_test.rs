//! The whole model, and its config.json from Hugging Face.

use super::{LEN, SEQS, part, positions, tiny, tokens};
use tessera_bundle::Overrides;
use tessera_bundle::qwen35::config;
use tessera_engine::{Model, Recipe, Tensor, Weight};
use tessera_models::qwen35::{Config, program};
use tessera_run::testing::runs;
use tessera_testkit::max_abs_diff;
use tessera_trace::DimName::Context;

/// `config.json` in the fixture's `name` folder, read.
fn read(name: &str) -> Option<Config> {
    let sizes = [SEQS as u64, 16];
    Some(config(&part(name)?, sizes, &Overrides::default()).unwrap())
}

#[test]
fn configs_read() {
    let (Some(small), Some(clef)) = (read("model"), read("clef")) else {
        return eprintln!("qwen35-tiny: no reference here; skipped");
    };
    assert_eq!(program(&small).to_json(), program(&tiny()).to_json());
    // Clef nests its text model's keys under text_config.
    assert_eq!(
        program(&clef).to_json(),
        program(&super::clef(SEQS as u64, 16)).to_json()
    );
}

/// Sequence `s`'s first `len` rows of `t`, `[seqs, rows, ..]`.
fn head(t: &Tensor, s: usize, len: usize) -> Tensor {
    let [seqs, rows] = [t.shape[0], t.shape[1]];
    let width = t.bytes().len() / (seqs * rows);
    let bytes = &t.bytes()[s * rows * width..][..len * width];
    let mut shape = t.shape.clone();
    (shape[0], shape[1]) = (1, len);
    Tensor::from_bytes(shape, t.dtype(), bytes)
}

#[test]
fn logits_match_hf() {
    let Some(dir) = part("model") else {
        return eprintln!("qwen35-tiny: no reference here; skipped");
    };
    let g = program(&tiny());
    let weights: Vec<Weight> = super::weights(&g, &dir)
        .into_iter()
        .map(Weight::Floats)
        .collect();
    let want = Tensor::f32(vec![SEQS, LEN, 64], super::read(&dir, "logits"));
    let (ids, at) = (tokens(&dir), positions());
    for settings in runs() {
        let model = Model::open(g.clone(), &Recipe::default(), &settings).unwrap();
        let mut model = model.bucketed(&[Context]);
        model.load(&weights).unwrap();
        // Both lengths share one 16-row plan.
        for (s, len) in [(0, LEN), (1, 7)] {
            let got = model
                .run(&[head(&ids, s, len), head(&at, s, len)])
                .unwrap()
                .remove(0);
            assert_eq!(got.shape, [1, len, 64]);
            let want = head(&want, s, len);
            let off = max_abs_diff(got.floats().unwrap(), want.floats().unwrap());
            assert!(
                off < 1e-4,
                "{len} rows on {}: off by {off:e}",
                model.device()
            );
        }
    }
}
