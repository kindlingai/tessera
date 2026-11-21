//! The gated full-attention layer.

use super::{LEN, SEQS, check, floats, part, positions, tiny};
use tessera_models::qwen35::Full;
use tessera_testkit::graphs::{floats as port, ids};
use tessera_trace::DimName::{Context, Hidden, Seqs};
use tessera_trace::{EntryKind, Model, Role, Weights, declare};

#[derive(Weights)]
struct One {
    attn: Full,
}

#[test]
fn attention_matches_hf() {
    let Some(dir) = part("attention") else {
        return eprintln!("qwen35-tiny: no reference here; skipped");
    };
    let c = tiny();
    let m = Model::new(c.dims());
    let w: One = declare(&m);
    let ports = [port([Seqs, Context, Hidden]), ids([Seqs, Context])];
    let [h, at] = m.entry(EntryKind::Forward, ports);
    let out = w.attn.apply(&m, &c, &h, &at);
    let g = m.finish([(Role::Logits, &out)]);
    let h = floats(&dir, "h", vec![SEQS, LEN, 32]);
    check(&g, &dir, &[h, positions()], "out");
}
