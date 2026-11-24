//! GLM-5.3's whole step, embedding to logits, as one persistent kernel.

use tessera_guest::site::{
    Ask, Found, Input, Kind, Lane, Offer, Place, Resident, Weights, agreed, layer, same, trellis,
};

const NAME: &str = "glm53_whole";
const THREADS: u32 = 256;

/// Weights outside the layers, in the kernel's item order.
const GLOBAL: [&str; 3] = ["embed", "norm", "head"];
/// Each layer's attention fields, then its MLP norm.
const ATTENTION: [&str; 9] = [
    "attn.{l}.norm",
    "attn.{l}.q_a",
    "attn.{l}.q_norm",
    "attn.{l}.q_b",
    "attn.{l}.kv_a",
    "attn.{l}.kv_norm",
    "attn.{l}.kv_b",
    "attn.{l}.o",
    "mlp_norm.{l}",
];
/// An indexing layer's indexer, after its attention fields.
const INDEXER: [&str; 5] = ["wq", "wk", "k_norm", "k_bias", "weights"];
const DENSE: [&str; 3] = ["gate", "up", "down"];
const ROUTED: [&str; 16] = [
    "router",
    "router_bias",
    "signs_in",
    "signs_out",
    "gate",
    "gate_su",
    "gate_sv",
    "up",
    "up_su",
    "up_sv",
    "down",
    "down_su",
    "down_sv",
    "shared.gate",
    "shared.up",
    "shared.down",
];
/// First field number of each group, in `whole.cu` order.
const AT_INDEXER: usize = GLOBAL.len() + ATTENTION.len();
const AT_DENSE: usize = AT_INDEXER + INDEXER.len();
const AT_ROUTED: usize = AT_DENSE + DENSE.len();
/// Expert code sets: plain trellis, not rotated.
const CODES: [usize; 3] = [AT_ROUTED + 4, AT_ROUTED + 7, AT_ROUTED + 10];
/// Signs `whole.cu` reads as plain f32.
const SIGNS: [usize; 2] = [AT_ROUTED + 2, AT_ROUTED + 3];
/// Form class for trellis weights.
const TRELLIS: u32 = 4;

/// Whether `whole.cu` decodes `input`'s form, and its class.
fn class(input: &Input) -> Option<u32> {
    let plain = match input.form.as_str() {
        "f32" => Some(0),
        "bf16" => Some(1),
        "int4_g32" => Some(2),
        "mxfp8" => Some(3),
        _ => None,
    };
    match plain {
        Some(c) => input.parts.is_none().then_some(c),
        None => trellis(&input.form).map(|_| TRELLIS),
    }
}

/// The reads collected in kernel item order.
struct Walk<'a> {
    weights: Weights<'a>,
    reads: Vec<Place>,
}

impl<'a> Walk<'a> {
    /// Reads `found` as field `field`, if `whole.cu` can.
    fn take(&mut self, found: Found<'a>, field: usize) -> Option<&'a Input> {
        let c = class(&found)?;
        let codes = CODES.contains(&field);
        // Rotated iff trellis outside the expert codes.
        let turned = c == TRELLIS && !codes;
        let fits = turned == found.parts.is_some() && (!SIGNS.contains(&field) || c == 0);
        if !fits {
            return None;
        }
        match turned {
            true => self.reads.extend(found.parts()),
            false => self.reads.push(found.read()),
        }
        Some(found.input)
    }

    /// Reads each of `fields` under `prefix`, from field `first`.
    fn group(&mut self, prefix: &str, fields: &[&str], first: usize) -> Option<Vec<&'a Input>> {
        let found = self.weights.group(prefix, fields)?.into_iter().enumerate();
        found.map(|(i, f)| self.take(f, first + i)).collect()
    }
}

/// Model sizes from the first layer's weights.
struct Sizes {
    hidden: usize,
    q_latent: usize,
    latent: usize,
    rope: usize,
    nope: usize,
    head_dim: usize,
    heads: usize,
}

fn sizes(a: &[&Input]) -> Option<Sizes> {
    let (q_latent, latent) = (a[2].shape[0], a[5].shape[0]);
    let rope = a[4].shape[0].checked_sub(latent)?;
    let [q, kv, o] = [a[3].whole(0), a[6].whole(0), a[7].whole(1)];
    let heads = (q + o).checked_sub(kv)? / rope.max(1);
    let fits = heads > 0 && q % heads == 0 && o % heads == 0 && q / heads > rope;
    fits.then(|| Sizes {
        hidden: a[0].shape[0],
        q_latent,
        latent,
        rope,
        nope: q / heads - rope,
        head_dim: o / heads,
        heads,
    })
}

/// This rank's heads: one part for `q_b`, `kv_b`, `o`.
fn heads(a: &[&Input], s: &Sizes) -> Option<Option<[usize; 2]>> {
    let q = a[3].split(0, s.nope + s.rope)?;
    let kv = a[6].split(0, s.nope + s.head_dim)?;
    same([q, kv, a[7].split(1, s.head_dim)?])
}

/// An MLP's part of its inner width, from all three.
fn inner(m: &[&Input]) -> Option<Option<[usize; 2]>> {
    same([m[0].split(0, 1)?, m[1].split(0, 1)?, m[2].split(1, 1)?])
}

/// The indexing layers: where each indexer's `wq` is read.
fn indexing(w: &Weights, layers: usize) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for i in 0.. {
        let Some(wq) = w.get(&format!("indexers.{i}.wq")) else {
            break;
        };
        let q_a = |l: &usize| w.get(&format!("attn.{l}.q_a"));
        let before = |l: &usize| q_a(l).is_some_and(|q| q.at < wq.at);
        out.push((0..layers).rfind(before)?);
    }
    Some(out)
}

/// The `every` that gives `layers` from `dense` dense layers.
fn every(indexing: &[usize], dense: usize, layers: usize) -> Option<usize> {
    let picks =
        |e: usize| (0..layers).filter(move |&l| l < dense || (l + 1 - dense).is_multiple_of(e));
    (1..=layers).find(|&e| picks(e).eq(indexing.iter().copied()))
}

/// One resident offer for the whole step, else `None`.
fn offer(ask: &Ask) -> Option<Vec<Offer>> {
    let step = ask.site.step()?;
    let (kept, sums) = (step.kept, step.sums);
    let rows = step.rows();
    let cfg = |key| ask.config.as_ref()?.get(key);
    if cfg("max_rows")
        .and_then(|v| v.as_u64())
        .is_some_and(|max| rows as u64 > max)
    {
        return None;
    }
    let lane = match cfg("sum_bf16").and_then(|v| v.as_bool()) {
        Some(true) => Lane::SumBf16,
        _ => Lane::Sum,
    };
    let first = kept.first()?;
    let mut w = Walk {
        weights: Weights::new(step.inputs),
        reads: Vec::new(),
    };
    let global = w.group("", &GLOBAL, 0)?;
    let layers = w.weights.layers("attn.{l}.q_a");
    let dense = w.weights.layers("dense.{l}.gate");
    let index = indexing(&w.weights, layers)?;
    let every = every(&index, dense, layers)?;
    let (mut attention, mut mlps, mut routed, mut indexers) = (vec![], vec![], vec![], vec![]);
    for l in 0..layers {
        let fields = ATTENTION.map(|f| layer(f, l));
        attention.push(w.group("", &fields.each_ref().map(|f| f.as_str()), GLOBAL.len())?);
        if let Some(i) = index.iter().position(|&x| x == l) {
            indexers.push(w.group(&format!("indexers.{i}."), &INDEXER, AT_INDEXER)?);
        }
        match l.checked_sub(dense) {
            None => mlps.push(w.group(&format!("dense.{l}."), &DENSE, AT_DENSE)?),
            Some(m) => routed.push(w.group(&format!("moe.{m}."), &ROUTED, AT_ROUTED)?),
        }
    }
    let s = sizes(attention.first()?)?;
    let [vocab, hidden] = global[0].shape[..] else {
        return None;
    };
    global[2].split(0, 1)?;
    let [head_out, head_in] = global[2].shape[..] else {
        return None;
    };
    if head_in != hidden || global[2].whole(0) != vocab || head_out == 0 {
        return None;
    }
    let ix = indexers.first()?;
    let (index_heads, index_dim) = (ix[4].shape[0], ix[1].shape[0]);
    let r = routed.first()?;
    let (experts, expert_ffn) = (r[0].shape[0], r[6].shape[1]);
    // Split layers share one part size.
    let attn: Vec<_> = attention
        .iter()
        .map(|a| heads(a, &s))
        .collect::<Option<_>>()?;
    let dense_parts: Vec<_> = mlps.iter().map(|m| inner(m)).collect::<Option<_>>()?;
    let mut moe_parts = Vec::with_capacity(routed.len());
    for m in &routed {
        let held = same([4, 7, 10].map(|i| m[i].split(0, 1)))??;
        let shared = inner(&m[13..])?;
        if held.is_some() != shared.is_some() {
            return None;
        }
        moe_parts.push(held.zip(shared));
    }
    let count = |parts: &[Option<[usize; 2]>], all: usize| Some(agreed(parts, [0, all])?[1]);
    let attended = count(&attn, s.heads)?;
    let dense_all = mlps.first()?[2].whole(1);
    let dense_ffn = count(&dense_parts, dense_all)?;
    let ffn_all = r[15].whole(1);
    let shared: Vec<_> = moe_parts.iter().map(|p| p.map(|(_, s)| s)).collect();
    let ffn = count(&shared, ffn_all)?;
    let held: Vec<_> = moe_parts.iter().map(|p| p.map(|(h, _)| h)).collect();
    let [held_first, held] = agreed(&held, [0, experts])?;
    let mlp_split =
        (dense_parts.iter().map(Option::is_some)).chain(moe_parts.iter().map(Option::is_some));
    let bits = |split: &mut dyn Iterator<Item = bool>| {
        let mut words = [0usize; 3];
        for (l, on) in split.enumerate() {
            words[l / 32] |= usize::from(on) << (l % 32);
        }
        words
    };
    let attn_bits = bits(&mut attn.iter().map(Option::is_some));
    let mlp_bits = bits(&mut mlp_split.clone());
    // Only heads, inner widths, experts and head split.
    let unsplit = |group: &[Vec<&Input>], split: &[usize]| {
        let kept = |(i, _): &(usize, &&Input)| !split.contains(i);
        let rest = group.iter().flat_map(|g| g.iter().enumerate().filter(kept));
        rest.into_iter().all(|(_, i)| i.share.is_none())
    };
    let whole_rest = unsplit(std::slice::from_ref(&global), &[2])
        && unsplit(&indexers, &[])
        && unsplit(&attention, &[3, 6, 7])
        && unsplit(&routed, &[4, 7, 10, 13, 14, 15]);
    let summed = attn.iter().flatten().count() + mlp_split.filter(|on| *on).count();
    // Trimmed last layers sum picked rows. The grid sums all.
    let sums_fit = sums.len() == summed && sums.iter().all(|n| n % hidden == 0);
    let kept_fit = kept.len() == layers + index.len();
    if !(whole_rest && sums_fit && kept_fit && layers <= 96) {
        return None;
    }
    if hidden != s.hidden || hidden % 128 != 0 {
        return None;
    }
    let tops: Vec<f32> = (step.scalars.iter())
        .filter(|s| s.op == "top_k")
        .map(|s| s.value)
        .collect();
    let keep = *tops.first()? as usize;
    let top_k = *tops.get(1).or(tops.first())? as usize;
    let context = first.context;
    let width = [
        hidden,
        s.q_latent,
        s.heads * (s.latent + s.rope).max(s.nope + s.rope).max(s.head_dim),
        index_heads * index_dim,
        context,
        dense_all,
        ffn_all,
        top_k * hidden.max(expert_ffn),
        experts,
    ];
    let width = width.into_iter().max()?;
    let shape = [
        head_out,
        hidden,
        layers,
        dense,
        every,
        attended,
        s.heads,
        s.nope,
        s.rope,
        s.head_dim,
        s.q_latent,
        s.latent,
        index_heads,
        index_dim,
        keep,
        context,
        dense_ffn,
        dense_all,
        ffn,
        ffn_all,
        expert_ffn,
        experts,
        held_first,
        held,
        top_k,
    ];
    let sizes = [&shape[..], &attn_bits, &mlp_bits, &[width]].concat();
    let ops = ["rms_norm", "rope_adjacent", "scale", "attend"];
    let floats =
        rows * (3 * hidden + s.q_latent + s.latent + s.rope + 6 * width + keep + 2 * top_k);
    let work = (rows * layers * hidden * (s.q_latent + s.heads * s.head_dim)) as f64;
    let resident = Resident::grid(THREADS, floats * 4, lane);
    let whole = step.resident(NAME, w.reads, &sizes, &ops, work, resident)?;
    Some(vec![whole])
}

tessera_guest::exports! {
    rules: { NAME => Kind::Model },
    kernels: { "whole.cu": [NAME] },
    files: ["math.cuh", "trellis.cuh", "gemv.cuh", "attn.cuh", "moe.cuh"],
    ask: offer,
}
