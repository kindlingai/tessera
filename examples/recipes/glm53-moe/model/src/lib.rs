//! GLM-5.3's routed block as fused CUDA kernels.

use tessera_guest::site::{Input, Kind, Offer, Place, Shape, Site, Weights, fixed};

const NAME: &str = "glm53_moe";
const ROUTE: &str = "glm53_route";
const PLAN: &str = "glm53_plan";
const FC1: &str = "glm53_fc1";
const FC2: &str = "glm53_fc2";
const FINISH: &str = "glm53_finish";

/// The routed weights, by field.
const ROUTED: [&str; 13] = [
    "router",
    "router_bias",
    "signs_in",
    "gate_su",
    "gate",
    "gate_sv",
    "up_su",
    "up",
    "up_sv",
    "down_su",
    "down",
    "down_sv",
    "signs_out",
];
const SHARED: [&str; 3] = ["shared.gate", "shared.up", "shared.down"];
/// Expert codes: plain trellis, split by expert.
const CODES: [&str; 3] = ["gate", "up", "down"];
/// Weights a split of the inner width cuts.
const INNER: [&str; 6] = ["gate", "gate_sv", "up", "up_sv", "down_su", "down"];

/// Past these, the linked rules serve better or `moe.cu` cannot.
const MOST_ROWS: usize = 64;
const MOST_EXPERTS: usize = 2048;
const MAX_SLOTS: usize = 512;
const PICKS: usize = 8;
/// Words in `Plan` struct in `moe.cu`.
const PLAN_WORDS: usize = 1 + MAX_SLOTS * 6 + MOST_ROWS * PICKS * 2;

/// Experts each row takes, by expert count.
fn picks(experts: usize) -> Option<u32> {
    match experts {
        256 => Some(8),
        32 => Some(2),
        _ => None,
    }
}

/// Largest divisor of `blocks` up to `want`.
fn split_dim(blocks: usize, want: usize) -> usize {
    let mut d = want.min(blocks);
    while d > 1 {
        if blocks.is_multiple_of(d) {
            return d;
        }
        d -= 1;
    }
    1
}

/// A trellis form's width and codebook: 0 mcg, 1 mul1.
fn bits(input: &Input) -> Option<[u32; 2]> {
    let rest = input.form.strip_prefix("trellis_k")?;
    let (k, book) = match rest.strip_suffix("_mul1") {
        Some(k) => (k, 1),
        None => (rest, 0),
    };
    let k: u32 = k.parse().ok()?;
    (2..=4).contains(&k).then_some([k, book])
}

/// One width and codebook for all of `codes`.
fn width(codes: &[&Input]) -> Option<[u32; 2]> {
    let k = bits(codes.first()?)?;
    codes.iter().all(|c| bits(c) == Some(k)).then_some(k)
}

/// Experts this rank holds: first and count.
fn held(w: &Weights, experts: usize, ffn: usize) -> Option<[usize; 2]> {
    let share = |field: &str| w.get(field)?.share;
    let whole_but =
        |split: &[&str]| (ROUTED.iter()).all(|f| split.contains(f) || share(f).is_none());
    if whole_but(&CODES) {
        let held = share("gate");
        if share("up") != held || share("down") != held {
            return None;
        }
        return match held {
            None => Some([0, experts]),
            Some([0, first, count, whole]) if whole == experts => Some([first, count]),
            Some(_) => None,
        };
    }
    let Some([1, first, count, whole]) = share("gate") else {
        return None;
    };
    let same = |field: &str, axis: usize| share(field) == Some([axis, first, count, whole]);
    let cut = INNER[..5].iter().all(|f| same(f, 1)) && same("down", 2);
    (whole_but(&INNER) && count == ffn && cut).then_some([0, experts])
}

/// Route, plan, fc1, fc2, then finish over a routed block.
fn offer(site: &Site) -> Option<Vec<Offer>> {
    let r = site.region(Kind::Moe)?;
    let [len] = r.outputs[..] else {
        return None;
    };
    let x = r.inputs.first()?;
    let [rows, hidden] = *x.shape.as_slice() else {
        return None;
    };
    let w = Weights::new(r.inputs);
    let routed = w.all(ROUTED)?;
    let shared = w.all(SHARED)?;
    let [
        router,
        bias,
        signs_in,
        gate_su,
        gate,
        gate_sv,
        up_su,
        up,
        up_sv,
        down_su,
        down,
        down_sv,
        signs_out,
    ] = routed;
    let [shared_gate, shared_up, shared_down] = shared;
    let plain = routed.iter().all(|i| i.parts.is_none());
    let turned = shared.iter().all(|i| i.parts.is_some());
    let fits = r.inputs.len() == 1 + ROUTED.len() + SHARED.len()
        && w.len() == ROUTED.len() + SHARED.len()
        && r.sums.is_empty()
        && x.weight.is_none()
        && x.dims.iter().all(Option::is_none)
        && r.kept.is_empty();
    if !(plain && turned && fits) {
        return None;
    }
    let [experts, ffn] = *gate_sv.shape.as_slice() else {
        return None;
    };
    let [first, count] = held(&w, experts, ffn)?;
    let [inner, wide] = shared_gate.parts?;
    let sized = router.shape == [experts, hidden]
        && inner == hidden
        && shared_up.parts == shared_gate.parts
        && shared_down.parts == Some([wide, hidden])
        && rows <= MOST_ROWS
        && experts <= MOST_EXPERTS
        && hidden % 128 == 0
        && ffn % 128 == 0
        && wide % 128 == 0
        && wide <= ffn;
    if !sized {
        return None;
    }
    // Floats are f32, but the router may be bf16.
    let mut floats = ROUTED.iter().zip(&routed);
    let floats = floats.all(|(f, i)| *f == "router" || CODES.contains(f) || i.form == "f32");
    if x.form != "f32" || !floats {
        return None;
    }
    let bf16 = match router.form.as_str() {
        "f32" => 0,
        "bf16" => 1,
        _ => return None,
    };
    let [k1, book1] = width(&[gate.input, up.input])?;
    let [k2, book2] = bits(down.input)?;
    let [sh_k1, sh_book1] = width(&[shared_gate.input, shared_up.input])?;
    let [sh_k2, sh_book2] = bits(shared_down.input)?;
    let picks = picks(experts)? as usize;
    let [scores, plan, gpart, z] = [0, 1, 2, 3].map(|draft| Place::Draft { draft });
    let chunks = ffn / 128;
    let ibs = hidden / 128;
    let ks1 = split_dim(ibs, if rows == 1 { 8 } else { 4 });
    let ns2 = split_dim(ibs, if rows == 1 { 8 } else { 3 });
    let n_sh = rows.div_ceil(8);
    let assigns = (rows * picks).min(count * rows);
    let active = count.min(assigns);
    let max_routed = (count * n_sh).min(active + (assigns - active) / 8);
    let total_slots = n_sh + max_routed;
    let max_col = rows.min(8);
    let route = Offer {
        kernel: ROUTE,
        reads: vec![Place::Read(0), router.read()],
        len: rows * experts,
        counts: fixed(&[rows, hidden, experts, bf16]),
        shape: Shape::grid(experts.div_ceil(2), 1, 256),
        work: (rows * experts * hidden) as f64,
        guess: 1e-12,
        writes: None,
        resident: None,
    };
    let plan_offer = Offer {
        kernel: PLAN,
        reads: vec![scores, bias.read()],
        len: PLAN_WORDS,
        counts: fixed(&[rows, experts, picks, first, count]),
        shape: Shape::grid(1, 1, 256),
        work: (rows * experts) as f64,
        guess: 1e-12,
        writes: None,
        resident: None,
    };
    let codes1 = [k1, sh_k1, book1, sh_book1].map(|b| b as usize);
    let codes2 = [k2, sh_k2, book2, sh_book2].map(|b| b as usize);
    let fc1_counts = [
        &[rows, hidden, ffn, picks][..],
        &codes1,
        &[first, wide, ks1],
    ]
    .concat();
    let fc2_counts = [
        &[rows, hidden, ffn, picks][..],
        &codes2,
        &[first, wide, ks1, ns2],
    ]
    .concat();
    let mut reads = vec![Place::Read(0), plan];
    reads.extend([signs_in, gate_su, gate, up_su, up].map(|f| f.read()));
    reads.extend(
        [shared_gate, shared_up]
            .map(|f| [f.part(0), f.part(1)])
            .concat(),
    );
    let fc1 = Offer {
        kernel: FC1,
        reads,
        len: total_slots * chunks * 2 * ks1 * max_col * 128,
        counts: fixed(&fc1_counts),
        shape: Shape::grid(chunks * 2 * ks1, total_slots, 256),
        work: (total_slots * 2 * hidden * ffn) as f64,
        guess: 1e-15,
        writes: None,
        resident: None,
    };
    let mut reads = vec![plan, gpart];
    reads.extend([gate_sv, up_sv, down_su, down].map(|f| f.read()));
    reads.extend([shared_gate.part(2), shared_up.part(2)]);
    reads.extend([shared_down.part(0), shared_down.part(1)]);
    let fc2 = Offer {
        kernel: FC2,
        reads,
        len: rows * (picks + 1) * chunks * hidden / 2,
        counts: fixed(&fc2_counts),
        shape: Shape::grid(chunks * ns2, total_slots, 256),
        work: (total_slots * hidden * ffn) as f64,
        guess: 1e-15,
        writes: None,
        resident: None,
    };
    let finish = Offer {
        kernel: FINISH,
        reads: vec![
            plan,
            down_sv.read(),
            signs_out.read(),
            shared_down.part(2),
            z,
        ],
        len,
        counts: fixed(&[rows, hidden, ffn, picks, wide]),
        shape: Shape::grid(hidden / 128, rows, (picks + 1) * 32),
        work: len as f64,
        guess: 1e-12,
        writes: None,
        resident: None,
    };
    Some(vec![route, plan_offer, fc1, fc2, finish])
}

tessera_guest::exports! {
    rules: { NAME => Kind::Moe },
    kernels: { "moe.cu": [ROUTE, PLAN, FC1, FC2, FINISH] },
    propose: offer,
}

#[cfg(test)]
mod offer_test;
