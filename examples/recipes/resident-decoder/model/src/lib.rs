//! A small decoder's whole step as one persistent kernel.

use tessera_guest::site::{self, Ask, Count, Input, Kept, Kind, Lane, Manifest, Offer};
use tessera_guest::site::{Place, Resident, Shape, Site};
use tessera_guest::{input, output};

const NAME: &str = "resident_decoder";
const ENTRY: &str = "resident_decoder";
const THREADS: u32 = 256;
/// The fields of one layer, in the kernel's item order.
const LAYER: [&str; 11] = [
    "ln1", "q", "q_norm", "k", "k_norm", "v", "o", "ln2", "mlp.gate", "mlp.up", "mlp.down",
];
/// The tests' small model: `tessera_models::small::qwen3`.
const EPS: f32 = 1e-6;
const THETA: f32 = 1e4;

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

/// The kept layout's code, and its rows per page.
fn layout(kept: &Kept) -> Option<[u32; 2]> {
    match (kept.layout.as_str(), kept.page) {
        ("dense", None) => Some([0, 0]),
        ("interleaved", None) => Some([1, 0]),
        (_, Some(page)) if page.is_power_of_two() => Some([2, page as u32]),
        _ => None,
    }
}

/// One resident offer for the whole step, else `None`.
fn offer(site: &Site) -> Option<Vec<Offer>> {
    let Site::Region {
        kind: Kind::Model,
        inputs,
        outputs,
        kept,
        sums,
    } = site
    else {
        return None;
    };
    let (&[len], Some(first)) = (&outputs[..], kept.first()) else {
        return None;
    };
    let [Some(slots), Some(positions)] = first.by else {
        return None;
    };
    let ids = |i: &usize| inputs[*i].form == "u32" && ![slots, positions].contains(i);
    let mut ids = (0..inputs.len()).filter(ids);
    let (tokens, picks) = (ids.next()?, ids.next()?);
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
    let [code, page] = layout(first)?;
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
    let root = 1.0 / (width as f32).sqrt();
    counts.extend([EPS, THETA, root].map(|f| Count::Fixed(f.to_bits())));
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

#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: i32) -> i32 {
    tessera_guest::alloc(len)
}

#[unsafe(no_mangle)]
pub extern "C" fn manifest() -> i64 {
    let manifest = Manifest {
        abi: tessera_guest::ABI,
        kernels: vec![site::imports(NAME, ENTRY, "whole.cu")],
        rules: vec![site::Rule {
            name: NAME,
            site: Kind::Model,
        }],
        graph: false,
        files: vec![site::File {
            path: "whole.cu",
            text: include_str!("whole.cu"),
        }],
    };
    output(serde_json::to_vec(&manifest).unwrap_or_default())
}

/// The offer for a site that fits, else `null`.
#[unsafe(no_mangle)]
pub extern "C" fn propose(at: i32, len: i32) -> i64 {
    // SAFETY: the host passes back what `alloc` returned.
    let bytes = unsafe { input(at, len) };
    let ask = serde_json::from_slice::<Ask>(bytes).ok();
    let offers = ask.and_then(|ask| offer(&ask.site));
    output(serde_json::to_vec(&offers).unwrap_or_else(|_| b"null".to_vec()))
}
