//! GLM-5.3's routed block as fused CUDA kernels.

use tessera_guest::output;
use tessera_guest::site::{self, Count, Input, Kind, Manifest, Offer, Place, Shape, Site};

const NAME: &str = "glm53_moe";
const ROUTE: &str = "glm53_route";
const PLAN: &str = "glm53_plan";
const FC1: &str = "glm53_fc1";
const FC2: &str = "glm53_fc2";
const FINISH: &str = "glm53_finish";

/// The weights after the rows, in kernel argument order.
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

/// Past these, the linked rules serve better or `moe.cu` cannot.
const MOST_ROWS: usize = 16;
const MOST_EXPERTS: usize = 2048;
/// Words in `Plan` struct in `moe.cu`.
const PLAN_WORDS: usize = 1 + 128 + 128 + 128 * 8 + 128 * 8 + 16 * 8 + 16 * 8;

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

/// Whether `input` is the weight `field`.
fn named(input: &Input, field: &str) -> bool {
    let path = input.weight.as_deref().unwrap_or_default();
    path.ends_with(&format!(".{field}"))
}

/// One width and codebook for all of `codes`.
fn width(codes: &[&Input]) -> Option<[u32; 2]> {
    let k = bits(codes.first()?)?;
    codes.iter().all(|c| bits(c) == Some(k)).then_some(k)
}

/// Experts this rank holds: first and count.
fn held(routed: &[Input], experts: usize, ffn: usize) -> Option<[usize; 2]> {
    let ep = (routed.iter().enumerate()).all(|(i, r)| r.share.is_none() || [4, 7, 10].contains(&i));
    if ep {
        let share = routed[4].share;
        if [7, 10].iter().any(|&i| routed[i].share != share) {
            return None;
        }
        return match share {
            None => Some([0, experts]),
            Some([0, first, count, whole]) if whole == experts => Some([first, count]),
            Some(_) => None,
        };
    }
    let tp = (routed.iter().enumerate())
        .all(|(i, r)| r.share.is_none() || [4, 5, 7, 8, 9, 10].contains(&i));
    let Some([1, first, count, whole]) = routed[4].share else {
        return None;
    };
    let same = |i: usize, axis: usize| routed[i].share == Some([axis, first, count, whole]);
    let ok = tp && count == ffn && [4, 5, 7, 8, 9].into_iter().all(|i| same(i, 1)) && same(10, 2);
    ok.then_some([0, experts])
}

/// Route, plan, fc1, fc2, then finish over a routed block.
fn offer(site: &Site) -> Option<Vec<Offer>> {
    let Site::Region {
        kind: Kind::Moe,
        inputs,
        outputs,
        kept,
        sums,
        ..
    } = site
    else {
        return None;
    };
    if !sums.is_empty() {
        return None;
    }
    let [len] = outputs[..] else {
        return None;
    };
    let (x, rest) = inputs.split_first()?;
    let [rows, hidden] = *x.shape.as_slice() else {
        return None;
    };
    let (routed, shared) = rest.split_at_checked(ROUTED.len())?;
    let plain = routed.iter().all(|i| i.parts.is_none());
    let turned = shared.iter().all(|i| i.parts.is_some());
    let fields = ROUTED.iter().chain(&SHARED);
    let fits = rest.len() == ROUTED.len() + SHARED.len()
        && rest.iter().zip(fields).all(|(i, f)| named(i, f))
        && x.weight.is_none()
        && x.dims.iter().all(Option::is_none)
        && kept.is_empty();
    if !(plain && turned && fits) {
        return None;
    }
    let [experts, ffn] = *routed[5].shape.as_slice() else {
        return None;
    };
    let codes = [&routed[4], &routed[7], &routed[10]];
    let [first, count] = held(routed, experts, ffn)?;
    let [inner, wide] = shared[0].parts?;
    let sized = routed[0].shape == [experts, hidden]
        && inner == hidden
        && shared[1].parts == shared[0].parts
        && shared[2].parts == Some([wide, hidden])
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
    let floats = routed
        .iter()
        .enumerate()
        .filter(|(i, _)| ![0, 4, 7, 10].contains(i));
    if x.form != "f32" || floats.map(|(_, r)| &r.form).any(|f| f != "f32") {
        return None;
    }
    let bf16 = match routed[0].form.as_str() {
        "f32" => 0,
        "bf16" => 1,
        _ => return None,
    };
    let [k, book] = width(&codes)?;
    let [shared_k, shared_book] = width(&shared.iter().collect::<Vec<_>>())?;
    let picks = picks(experts)? as usize;
    let fixed = |n: &[usize]| n.iter().map(|&n| Count::Fixed(n as u32)).collect();
    let part = |i: usize, p: usize| Place::Part {
        part: [1 + ROUTED.len() + i, p],
    };
    let scores = Place::Draft { draft: 0 };
    let plan = Place::Draft { draft: 1 };
    let gpart = Place::Draft { draft: 2 };
    let z = Place::Draft { draft: 3 };
    let chunks = ffn / 128;
    let ibs = hidden / 128;
    let ks1 = split_dim(ibs, if rows == 1 { 8 } else { 4 });
    let ns2 = split_dim(ibs, if rows == 1 { 8 } else { 3 });
    let n_sh = rows.div_ceil(8);
    let max_routed = (count * n_sh).min(rows * picks);
    let total_slots = n_sh + max_routed;
    let max_col = rows.min(8);
    let route = Offer {
        kernel: ROUTE,
        reads: vec![Place::Read(0), Place::Read(1)],
        len: rows * experts,
        counts: fixed(&[rows, hidden, experts, bf16]),
        shape: Shape::Grid {
            across: experts.div_ceil(2),
            down: 1,
            threads: 256,
        },
        work: (rows * experts * hidden) as f64,
        guess: 1e-12,
        writes: None,
        resident: None,
    };
    let plan_offer = Offer {
        kernel: PLAN,
        reads: vec![scores, Place::Read(2)],
        len: PLAN_WORDS,
        counts: fixed(&[rows, experts, picks, first, count]),
        shape: Shape::Grid {
            across: 1,
            down: 1,
            threads: 256,
        },
        work: (rows * experts) as f64,
        guess: 1e-12,
        writes: None,
        resident: None,
    };
    let fc_counts = [
        rows,
        hidden,
        ffn,
        picks,
        k as usize,
        shared_k as usize,
        book as usize,
        shared_book as usize,
        first,
        wide,
        ks1,
    ];
    let mut fc2_counts = fc_counts.to_vec();
    fc2_counts.push(ns2);
    let fc1 = Offer {
        kernel: FC1,
        reads: vec![
            Place::Read(0),
            plan,
            Place::Read(3),
            Place::Read(4),
            Place::Read(5),
            Place::Read(7),
            Place::Read(8),
            part(0, 0),
            part(0, 1),
            part(1, 0),
            part(1, 1),
        ],
        len: total_slots * chunks * 2 * ks1 * max_col * 128,
        counts: fixed(&fc_counts),
        shape: Shape::Grid {
            across: chunks * 2 * ks1,
            down: total_slots,
            threads: 256,
        },
        work: (total_slots * 2 * hidden * ffn) as f64,
        guess: 1e-15,
        writes: None,
        resident: None,
    };
    let fc2 = Offer {
        kernel: FC2,
        reads: vec![
            plan,
            gpart,
            Place::Read(6),
            Place::Read(9),
            Place::Read(10),
            Place::Read(11),
            part(0, 2),
            part(1, 2),
            part(2, 0),
            part(2, 1),
        ],
        len: rows * (picks + 1) * chunks * hidden / 2,
        counts: fixed(&fc2_counts),
        shape: Shape::Grid {
            across: chunks * ns2,
            down: total_slots,
            threads: 256,
        },
        work: (total_slots * hidden * ffn) as f64,
        guess: 1e-15,
        writes: None,
        resident: None,
    };
    let finish = Offer {
        kernel: FINISH,
        reads: vec![plan, Place::Read(12), Place::Read(13), part(2, 2), z],
        len,
        counts: fixed(&[rows, hidden, ffn, picks, wide]),
        shape: Shape::Grid {
            across: hidden / 128,
            down: rows,
            threads: (picks + 1) * 32,
        },
        work: len as f64,
        guess: 1e-12,
        writes: None,
        resident: None,
    };
    Some(vec![route, plan_offer, fc1, fc2, finish])
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: i32) -> i32 {
    tessera_guest::alloc(len)
}

#[unsafe(no_mangle)]
pub extern "C" fn manifest() -> i64 {
    let manifest = Manifest {
        abi: tessera_guest::ABI,
        kernels: [ROUTE, PLAN, FC1, FC2, FINISH]
            .map(|k| site::imports(k, k, "moe.cu"))
            .into(),
        rules: vec![site::Rule {
            name: NAME,
            site: Kind::Moe,
        }],
        graph: false,
        files: vec![site::File {
            path: "moe.cu",
            text: include_str!("moe.cu"),
        }],
    };
    output(serde_json::to_vec(&manifest).unwrap_or_default())
}

/// One offer for a site that fits, else `null`.
#[unsafe(no_mangle)]
pub extern "C" fn propose(at: i32, len: i32) -> i64 {
    // SAFETY: the host passes back what `alloc` returned.
    unsafe { tessera_guest::answer(at, len, offer) }
}
