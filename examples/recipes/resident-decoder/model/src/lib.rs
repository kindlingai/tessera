//! A small decoder's whole step as one persistent kernel.

use tessera_guest::site::{self, Count, Input, Kind, Lane, Offer};
use tessera_guest::site::{Place, Resident, Shape, Site};

const NAME: &str = "resident_decoder";
const THREADS: u32 = 256;
/// The fields of one layer, in the kernel's item order.
const LAYER: [&str; 11] = [
    "ln1", "q", "q_norm", "k", "k_norm", "v", "o", "ln2", "mlp.gate", "mlp.up", "mlp.down",
];

/// The place of the weight at `path`.
fn weight(inputs: &[Input], path: &str) -> Option<usize> {
    let named = |i: &Input| i.weight.as_deref() == Some(path) && i.form == "f32";
    inputs.iter().position(named)
}

/// A dimension read at launch, else its size now.
fn count(input: &Input) -> Count {
    match input.dims.first() {
        Some(Some(dim)) => Count::Dim {
            dim: dim.clone(),
            mul: 1,
        },
        _ => Count::Fixed(input.shape[0] as u32),
    }
}

/// The first unit of a split, and the whole count.
fn part(input: &Input, unit: usize) -> [usize; 2] {
    match input.share {
        Some([_, first, _, whole]) => [first / unit, whole / unit],
        None => [0, input.shape[0] / unit],
    }
}

/// One resident offer for the whole step, else `None`.
fn offer(site: &Site) -> Option<Vec<Offer>> {
    let step = site.step()?;
    let (inputs, kept, sums, scalars, len) =
        (step.inputs, step.kept, step.sums, step.scalars, step.len);
    let [tokens, positions, slots, picks] = step.ids;
    let first = &kept[0];
    let layers = (0..).take_while(|l| weight(inputs, &format!("layers.{l}.q")).is_some());
    let layers = layers.count();
    let mut reads = vec![tokens, positions, slots, picks];
    reads.push(weight(inputs, "embed")?);
    reads.push(weight(inputs, "norm")?);
    for l in 0..layers {
        for field in LAYER {
            reads.push(weight(inputs, &format!("layers.{l}.{field}"))?);
        }
    }
    let at = |l: usize, f: usize| &inputs[reads[6 + l * LAYER.len() + f]];
    let [vocab, hidden] = inputs[reads[4]].shape[..] else {
        return None;
    };
    let width = at(0, 2).shape[0];
    let (heads, kv, ffn) = (at(0, 1).shape[0], at(0, 3).shape[0], at(0, 8).shape[0]);
    let ([first_head, all_heads], [kv_first, all_kv]) =
        (part(at(0, 1), width), part(at(0, 3), width));
    let split = [at(0, 6).share.is_some(), at(0, 10).share.is_some()];
    let rows = inputs[tokens].shape[0];
    let per = split.iter().filter(|s| **s).count();
    let summed = sums.len() == layers * per && sums.iter().all(|s| *s == rows * hidden);
    let paired = kept.len() == 2 * layers && kept.iter().all(|k| k.width == kv);
    let [code, page] = first.code()?;
    if !(summed && paired && kv > 0 && all_heads % all_kv.max(1) == 0) {
        return None;
    }
    let fixed = |n: usize| Count::Fixed(n as u32);
    let mut counts = vec![count(&inputs[tokens]), count(&inputs[picks])];
    let sizes = [vocab, hidden, layers, heads / width, kv / width, width, ffn];
    counts.extend(sizes.map(fixed));
    let shape = [
        first_head,
        all_heads / all_kv,
        kv_first,
        split[0] as usize,
        split[1] as usize,
    ];
    counts.extend(shape.map(fixed));
    counts.extend([code, page].map(Count::Fixed));
    let eps = site::scalar(scalars, "rms_norm")?;
    let theta = site::scalar(scalars, "rope_halves")?;
    let root = site::scalar(scalars, "attend")?;
    counts.extend([eps, theta, root].map(|f| Count::Fixed(f.to_bits())));
    let mut places: Vec<Place> = reads.into_iter().map(Place::Read).collect();
    places.extend((0..kept.len()).map(|kept| Place::Kept { kept }));
    let floats = rows * (3 * hidden + 2 * heads + 2 * kv + ffn);
    Some(vec![Offer {
        kernel: NAME,
        reads: places,
        len,
        counts,
        shape: Shape::Line(1),
        work: (rows * layers * hidden * (heads + 2 * kv + 3 * ffn)) as f64,
        guess: 1e-12,
        writes: None,
        resident: Some(Resident {
            threads: THREADS,
            scratch: floats * 4,
            lanes: vec![Lane::Sum],
        }),
    }])
}

tessera_guest::exports!(NAME, Kind::Model, "whole.cu", offer);
