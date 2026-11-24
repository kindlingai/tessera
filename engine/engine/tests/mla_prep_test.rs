//! End-to-end MLA parity and launch-count checks for fused Q/KV prep.

use tessera_engine::{Engine, Recipe, Row, Weight};
use tessera_graph::DimName::{Context, Rows, Seqs};
use tessera_interp::{Store, Tensor, run_kept};
use tessera_lower::form::{BF16_FORM, F32_FORM, FP8_FORM};
use tessera_lower::kept::{self, Layout};
use tessera_lower::{Device, Pins, Plan, lower};
use tessera_models::mla;
use tessera_run::testing::runs;
use tessera_testkit::{Noise, weights};

fn count(plan: &Plan, name: &str) -> usize {
    plan.steps
        .iter()
        .filter(|s| s.kernel.name() == name)
        .count()
}

fn step_ref(g: &tessera_graph::Graph, w: &[Tensor], store: &mut Store, rows: &[Row]) -> Vec<u32> {
    let n = rows.len();
    let tokens = Tensor::u32(vec![n], rows.iter().map(|r| r.token).collect());
    let pos = Tensor::u32(vec![n], rows.iter().map(|r| r.position).collect());
    let slots = Tensor::u32(vec![n], rows.iter().map(|r| r.slot).collect());
    let out = run_kept(g, w, store, 0, &[tokens, pos, slots]).unwrap();
    out[0]
        .indices()
        .unwrap()
        .iter()
        .map(|&i| i as u32)
        .collect()
}

/// Checks fused vs unfused launch counts on MLA.
fn check_launch_drop(cfg: &mla::Config, rows: u64, device: &Device) -> (usize, usize) {
    let g = mla::stepper::<mla::RotatedMoe>(cfg);
    let sizes = [(Seqs, cfg.stack.max_seqs), (Rows, rows), (Context, 64)];
    let fused = lower(&g, 0, &sizes, device).unwrap();
    let bare = device
        .clone()
        .without("mla_kv_scatter")
        .without("mla_kv_scatter_plain")
        .without("join_rope");
    let unfused = lower(&g, 0, &sizes, &bare).unwrap();
    let layers = cfg.stack.layers as usize;
    assert_eq!(count(&fused, "mla_kv_scatter"), layers);
    assert_eq!(count(&fused, "join_rope"), layers);
    assert_eq!(count(&fused, "rope_adjacent"), 0);
    assert_eq!(count(&fused, "join"), 0);
    // Saves 5 KV launches plus 2 or 3 Q launches.
    let min_drop = layers * if rows == 1 { 7 } else { 8 };
    assert!(
        unfused.steps.len() >= fused.steps.len() + min_drop,
        "rows={rows}: fused={} unfused={} (want >= {min_drop} drop)",
        fused.steps.len(),
        unfused.steps.len(),
    );
    (fused.steps.len(), unfused.steps.len())
}

#[test]
fn mla_prep_launch_reduction() {
    let mut small = mla::Config::glm53_small();
    small.stack.max_context = 64;
    let fwd = mla::program::<mla::RotatedMoe>(&small);
    for settings in runs() {
        let device = Device::guess(&settings);
        for rows in [1u64, 8] {
            check_launch_drop(&small, rows, &device);
        }
        let plan = lower(&fwd, 0, &[(Seqs, 1), (Context, 8)], &device).unwrap();
        assert_eq!(count(&plan, "join_rope"), small.stack.layers as usize);
    }
    let full = mla::Config::glm53(4, 512);
    let device = Device::guess(&tessera_run::testing::host());
    for rows in [1u64, 8] {
        check_launch_drop(&full, rows, &device);
    }
}

#[test]
fn mla_prep_parity() {
    let mut cfg = mla::Config::glm53_small();
    cfg.stack.max_context = 64;
    let g = mla::stepper::<mla::RotatedMoe>(&cfg);
    let w = weights(&g, &mut Noise(19));
    let stored: Vec<Weight> = w.iter().cloned().map(Weight::Floats).collect();
    let row = |slot, token, position| Row {
        slot,
        token,
        position,
    };
    let prefill: Vec<Row> = (0..8u32)
        .map(|i| row(i / 4, (i * 5 + 3) % 64, i % 4))
        .collect();
    let decode = [row(0, 7, 4), row(1, 11, 4)];
    let layouts: [&'static dyn Layout; 2] = [kept::DENSE, &kept::PAGED16];
    for settings in runs() {
        for &layout in &layouts {
            for form in [F32_FORM, BF16_FORM, FP8_FORM] {
                let recipe = Recipe {
                    kept: layout,
                    pins: Pins::default().kept(form),
                    ..Recipe::default()
                };
                let mut engine = Engine::open(g.clone(), 64, &recipe, &settings).unwrap();
                engine.load(&stored).unwrap();
                let mut store = Store::default();
                for batch in [&prefill[..], &decode[..]] {
                    let got = engine.step(batch).unwrap();
                    let want = step_ref(&g, &w, &mut store, batch);
                    if form.name() == "f32" {
                        assert_eq!(got, want, "{layout:?} {:?}", settings.backend);
                    } else {
                        assert_eq!(got.len(), want.len());
                    }
                }
            }
        }
    }
}
