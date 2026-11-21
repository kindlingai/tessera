//! GLM-5.3's whole step, embedding to logits, as one persistent kernel.

use std::collections::HashMap;
use tessera_guest::site::{Count, Input, Kind, Lane, Offer};
use tessera_guest::site::{Place, Resident, Scalar, Shape, Site};

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
const FIELDS: usize = AT_ROUTED + ROUTED.len();
const AT_INDEXER: usize = GLOBAL.len() + ATTENTION.len();
const AT_DENSE: usize = AT_INDEXER + INDEXER.len();
const AT_ROUTED: usize = AT_DENSE + DENSE.len();
/// Expert code sets: plain trellis, not rotated.
const CODES: [usize; 3] = [AT_ROUTED + 4, AT_ROUTED + 7, AT_ROUTED + 10];
/// Signs `whole.cu` reads as plain f32.
const SIGNS: [usize; 2] = [AT_ROUTED + 2, AT_ROUTED + 3];
/// Form classes: f32, bf16, int4 g32, mxfp8, then trellis.
const TRELLIS: u32 = 4;

/// A form's class, and a trellis form's width and codebook.
fn class(input: &Input) -> Option<(u32, Option<[usize; 2]>)> {
    let form = input.form.as_str();
    let plain = match form {
        "f32" => Some(0),
        "bf16" => Some(1),
        "int4_g32" => Some(2),
        "mxfp8" => Some(3),
        _ => None,
    };
    if let Some(c) = plain {
        return input.parts.is_none().then_some((c, None));
    }
    let rest = form.strip_suffix("_rot").unwrap_or(form);
    let rest = rest.strip_prefix("trellis_k")?;
    let (k, book) = match rest.strip_suffix("_mul1") {
        Some(k) => (k, 1),
        None => (rest, 0),
    };
    let k: usize = k.parse().ok().filter(|k| (2..=4).contains(k))?;
    Some((TRELLIS, Some([k, book])))
}

/// The class `whole.cu` reads from `bytes` for `n` elements.
fn sniff(bytes: usize, n: usize) -> u32 {
    let blocks = n.is_multiple_of(32);
    match bytes {
        b if b == 4 * n => 0,
        b if b == 2 * n => 1,
        b if blocks && b == n / 32 * 18 => 2,
        b if blocks && b == n / 32 * 33 => 3,
        _ => TRELLIS,
    }
}

/// Bytes of `n` elements of a class, trellis width `k`.
fn bytes(class: u32, n: usize, k: usize) -> usize {
    match class {
        0 => 4 * n,
        1 => 2 * n,
        2 => n / 32 * 18,
        3 => n / 32 * 33,
        _ => 16 + n * k / 8,
    }
}

/// The reads, and each field's trellis codebook.
struct Walk<'a> {
    inputs: &'a [Input],
    /// Each weight's input place, by path.
    paths: HashMap<&'a str, usize>,
    reads: Vec<Place>,
    books: [Option<usize>; FIELDS],
}

impl<'a> Walk<'a> {
    fn place(&self, path: &str) -> Option<usize> {
        self.paths.get(path).copied()
    }

    /// Reads weight `path` as field `field`, if `whole.cu` can.
    fn take(&mut self, path: &str, field: usize) -> Option<&'a Input> {
        let at = self.place(path)?;
        let input = &self.inputs[at];
        let (c, trellis) = class(input)?;
        let k = trellis.map_or(0, |[k, _]| k);
        let n = input.shape.iter().product();
        let codes = CODES.contains(&field);
        // Rotated iff trellis outside the expert codes.
        let turned = c == TRELLIS && !codes;
        let fits = sniff(bytes(c, n, k), n) == c
            && turned == input.parts.is_some()
            && (!SIGNS.contains(&field) || c == 0);
        if !fits {
            return None;
        }
        if let Some(b) = trellis.map(|[_, b]| b)
            && *self.books[field].get_or_insert(b) != b
        {
            return None;
        }
        match turned {
            true => self
                .reads
                .extend((0..3).map(|p| Place::Part { part: [at, p] })),
            false => self.reads.push(Place::Read(at)),
        }
        Some(input)
    }

    /// Reads each of `fields` under `prefix`, from field `first`.
    fn group(&mut self, prefix: &str, fields: &[&str], first: usize) -> Option<Vec<&'a Input>> {
        let mut out = Vec::with_capacity(fields.len());
        for (i, f) in fields.iter().enumerate() {
            out.push(self.take(&format!("{prefix}{f}"), first + i)?);
        }
        Some(out)
    }
}

/// An axis' whole size, from its share or its shape.
fn whole(input: &Input, axis: usize) -> usize {
    match input.share {
        Some([a, _, _, w]) if a == axis => w,
        _ => input.shape[axis],
    }
}

/// The part along `axis` in `unit`s: first and count.
fn part(input: &Input, axis: usize, unit: usize) -> Option<Option<[usize; 2]>> {
    match input.share {
        None => Some(None),
        Some([a, first, count, _]) if a == axis && first % unit == 0 && count % unit == 0 => {
            Some(Some([first / unit, count / unit]))
        }
        Some(_) => None,
    }
}

/// One value for every entry of `parts`, else `None`.
fn same<T: PartialEq + Copy>(parts: impl IntoIterator<Item = T>) -> Option<T> {
    let mut parts = parts.into_iter();
    let first = parts.next()?;
    parts.all(|p| p == first).then_some(first)
}

/// The one value of `op`, else `None`.
fn only(scalars: &[Scalar], op: &str) -> Option<f32> {
    let mut found = scalars.iter().filter(|s| s.op == op).map(|s| s.value);
    let v = found.next()?;
    found.next().is_none().then_some(v)
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
    let [q, kv, o] = [whole(a[3], 0), whole(a[6], 0), whole(a[7], 1)];
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
    let q = part(a[3], 0, s.nope + s.rope)?;
    let kv = part(a[6], 0, s.nope + s.head_dim)?;
    let o = part(a[7], 1, s.head_dim)?;
    (q == kv && kv == o).then_some(o)
}

/// An MLP's part of its inner width, from all three.
fn inner(m: &[&Input]) -> Option<Option<[usize; 2]>> {
    let [g, u, d] = [part(m[0], 0, 1)?, part(m[1], 0, 1)?, part(m[2], 1, 1)?];
    (g == u && u == d).then_some(d)
}

/// The indexing layers: where each indexer's `wq` is read.
fn indexing(w: &Walk, layers: usize) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for i in 0.. {
        let Some(at) = w.place(&format!("indexers.{i}.wq")) else {
            break;
        };
        let before = |l: &usize| w.place(&format!("attn.{l}.q_a")).is_some_and(|q| q < at);
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
fn offer(site: &Site) -> Option<Vec<Offer>> {
    let step = site.step()?;
    let (inputs, kept, sums, scalars, len) =
        (step.inputs, step.kept, step.sums, step.scalars, step.len);
    let [tokens, positions, slots, picks] = step.ids;
    let first = &kept[0];
    let named = inputs.iter().enumerate();
    let paths = named.filter_map(|(i, input)| Some((input.weight.as_deref()?, i)));
    let mut w = Walk {
        inputs,
        paths: paths.collect(),
        reads: vec![tokens, positions, slots, picks]
            .into_iter()
            .map(Place::Read)
            .collect(),
        books: [None; FIELDS],
    };
    let global = w.group("", &GLOBAL, 0)?;
    let layers = (0..)
        .take_while(|l| w.place(&format!("attn.{l}.q_a")).is_some())
        .count();
    let dense = (0..)
        .take_while(|l| w.place(&format!("dense.{l}.gate")).is_some())
        .count();
    let index = indexing(&w, layers)?;
    let every = every(&index, dense, layers)?;
    let (mut attention, mut mlps, mut routed, mut indexers) = (vec![], vec![], vec![], vec![]);
    for l in 0..layers {
        let fields = ATTENTION.map(|f| f.replace("{l}", &l.to_string()));
        let fields: Vec<&str> = fields.iter().map(String::as_str).collect();
        attention.push(w.group("", &fields, GLOBAL.len())?);
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
        let held = same([4, 7, 10].map(|i| part(m[i], 0, 1)))??;
        let shared = inner(&m[13..])?;
        if held.is_some() != shared.is_some() {
            return None;
        }
        moe_parts.push(held.zip(shared));
    }
    let count = |parts: &[Option<[usize; 2]>], all: usize| match same(parts.iter().flatten()) {
        Some(p) => Some(p[1]),
        None if parts.iter().all(Option::is_none) => Some(all),
        None => None,
    };
    let attended = count(&attn, s.heads)?;
    let dense_all = mlps.first()?[2].shape[1];
    let dense_ffn = count(&dense_parts, dense_all)?;
    let ffn_all = r[15].shape[1];
    let shared: Vec<_> = moe_parts.iter().map(|p| p.map(|(_, s)| s)).collect();
    let ffn = count(&shared, ffn_all)?;
    let held = same(moe_parts.iter().flatten().map(|(h, _)| *h));
    if held.is_none() && moe_parts.iter().any(Option::is_some) {
        return None;
    }
    let [held_first, held] = held.unwrap_or([0, experts]);
    let mlp_split =
        (dense_parts.iter().map(Option::is_some)).chain(moe_parts.iter().map(Option::is_some));
    let bits = |split: &mut dyn Iterator<Item = bool>| {
        let mut words = [0u32; 3];
        for (l, on) in split.enumerate() {
            words[l / 32] |= u32::from(on) << (l % 32);
        }
        words
    };
    let attn_bits = bits(&mut attn.iter().map(Option::is_some));
    let mlp_bits = bits(&mut mlp_split.clone());
    // Only heads, inner widths and experts split.
    let whole = |group: &[Vec<&Input>], split: &[usize]| {
        let kept = |(i, _): &(usize, &&Input)| !split.contains(i);
        let rest = group.iter().flat_map(|g| g.iter().enumerate().filter(kept));
        rest.into_iter().all(|(_, i)| i.share.is_none())
    };
    let whole_rest = whole(std::slice::from_ref(&global), &[])
        && whole(&indexers, &[])
        && whole(&attention, &[3, 6, 7])
        && whole(&routed, &[4, 7, 10, 13, 14, 15]);
    let summed = attn.iter().flatten().count() + mlp_split.filter(|on| *on).count();
    let rows = inputs[tokens].shape[0];
    // Trimmed last layers sum picked rows. The grid sums all.
    let sums_fit = sums.len() == summed && sums.iter().all(|n| n % hidden == 0);
    let kept_fit = kept.len() == 2 * layers + index.len();
    if !(whole_rest && sums_fit && kept_fit && layers <= 96) {
        return None;
    }
    if hidden != s.hidden || hidden % 128 != 0 {
        return None;
    }
    let tops: Vec<f32> = scalars
        .iter()
        .filter(|s| s.op == "top_k")
        .map(|s| s.value)
        .collect();
    let keep = *tops.first()? as usize;
    let top_k = *tops.get(1).or(tops.first())? as usize;
    let eps = only(scalars, "rms_norm")?;
    let theta = only(scalars, "rope_adjacent")?;
    let route = only(scalars, "scale")?;
    let scale = only(scalars, "attend")?;
    let [code, page] = first.code()?;
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
    let fixed = |n: usize| Count::Fixed(n as u32);
    let dim = |i: usize| match inputs[i].dims.first() {
        Some(Some(dim)) => Count::Dim {
            dim: dim.clone(),
            mul: 1,
        },
        _ => fixed(inputs[i].shape[0]),
    };
    let mut counts = vec![dim(tokens), dim(picks)];
    let shape = [
        vocab,
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
    counts.extend(shape.map(fixed));
    counts.extend(attn_bits.into_iter().chain(mlp_bits).map(Count::Fixed));
    let mut books = [0u32; 2];
    for (f, b) in w.books.iter().enumerate() {
        books[f / 32] |= (b.unwrap_or(0) as u32) << (f % 32);
    }
    counts.extend(books.map(Count::Fixed));
    counts.extend([fixed(width), Count::Fixed(code), Count::Fixed(page)]);
    counts.extend([eps, theta, route, scale].map(|f| Count::Fixed(f.to_bits())));
    let mut reads = w.reads;
    reads.extend((0..kept.len()).map(|kept| Place::Kept { kept }));
    let floats =
        rows * (3 * hidden + s.q_latent + s.latent + s.rope + 6 * width + keep + 2 * top_k);
    Some(vec![Offer {
        kernel: NAME,
        reads,
        len,
        counts,
        shape: Shape::Line(1),
        work: (rows * layers * hidden * (s.q_latent + s.heads * s.head_dim)) as f64,
        guess: 1e-12,
        writes: None,
        resident: Some(Resident {
            threads: THREADS,
            shared: 0,
            scratch: floats * 4,
            lanes: vec![Lane::Sum],
        }),
    }])
}

tessera_guest::exports!(NAME, Kind::Model, "whole.cu", offer);
