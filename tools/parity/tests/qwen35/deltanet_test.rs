//! The gated delta rule layer.

use super::{LEN, SEQS, check, floats, part, tiny};
use tessera_models::qwen35::DeltaNet;
use tessera_testkit::graphs::floats as port;
use tessera_trace::DimName::{Context, Hidden, Seqs};
use tessera_trace::{EntryKind, Model, Role, Weights, declare};

#[derive(Weights)]
struct One {
    linear: DeltaNet,
}

#[test]
fn deltanet_matches_hf() {
    let Some(dir) = part("deltanet") else {
        return eprintln!("qwen35-tiny: no reference here; skipped");
    };
    let c = tiny();
    let m = Model::new(c.dims());
    let w: One = declare(&m);
    let [h] = m.entry(EntryKind::Forward, [port([Seqs, Context, Hidden])]);
    let out = w.linear.apply(&m, &c, &h);
    let g = m.finish([(Role::Logits, &out)]);
    let h = floats(&dir, "h", vec![SEQS, LEN, 32]);
    check(&g, &dir, &[h], "out");
}
