//! A small decoder's whole step as one persistent kernel.

use tessera_guest::site::{Found, Kind, Lane, Offer, Resident, Site, Weights};

const NAME: &str = "resident_decoder";
const THREADS: u32 = 256;
/// The fields of one layer, in the kernel's item order.
const LAYER: [&str; 11] = [
    "ln1", "q", "q_norm", "k", "k_norm", "v", "o", "ln2", "mlp.gate", "mlp.up", "mlp.down",
];

/// One resident offer for the whole step, else `None`.
fn offer(site: &Site) -> Option<Vec<Offer>> {
    let step = site.step()?;
    let (kept, sums) = (step.kept, step.sums);
    let w = Weights::new(step.inputs);
    let layers = w.layers("layers.{l}.q");
    let mut found = w.group("", &["embed", "norm"])?;
    for l in 0..layers {
        found.extend(w.group(&format!("layers.{l}."), &LAYER)?);
    }
    let at = |l: usize, f: usize| found[2 + l * LAYER.len() + f];
    let [vocab, hidden] = found[0].shape[..] else {
        return None;
    };
    let width = at(0, 2).shape[0];
    let (q, k) = (at(0, 1), at(0, 3));
    let (heads, kv, ffn) = (q.shape[0], k.shape[0], at(0, 8).shape[0]);
    let (first_head, kv_first) = (q.split(0, width)?, k.split(0, width)?);
    let (all_heads, all_kv) = (q.whole(0) / width, k.whole(0) / width);
    let split = [at(0, 6).share.is_some(), at(0, 10).share.is_some()];
    let rows = step.rows();
    let per = split.iter().filter(|s| **s).count();
    let summed = sums.len() == layers * per && sums.iter().all(|s| *s == rows * hidden);
    let paired = kept.len() == 2 * layers && kept.iter().all(|k| k.width == kv);
    let [code, page] = kept[0].code()?;
    let f32s = found.iter().all(|f| f.form == "f32");
    if !(f32s && summed && paired && kv > 0 && all_heads % all_kv.max(1) == 0) {
        return None;
    }
    let sizes = [
        vocab,
        hidden,
        layers,
        heads / width,
        kv / width,
        width,
        ffn,
        first_head.map_or(0, |p| p[0]),
        all_heads / all_kv,
        kv_first.map_or(0, |p| p[0]),
        split[0] as usize,
        split[1] as usize,
        code as usize,
        page as usize,
    ];
    let ops = ["rms_norm", "rope_halves", "attend"];
    let reads = found.iter().map(Found::read);
    let floats = rows * (3 * hidden + 2 * heads + 2 * kv + ffn);
    let work = (rows * layers * hidden * (heads + 2 * kv + 3 * ffn)) as f64;
    let resident = Resident::grid(THREADS, floats * 4, Lane::Sum);
    let whole = step.resident(NAME, reads, &sizes, &ops, work, resident)?;
    Some(vec![whole])
}

tessera_guest::exports! {
    rules: { NAME => Kind::Model },
    kernels: { "whole.kernel": [NAME] },
    propose: offer,
}
