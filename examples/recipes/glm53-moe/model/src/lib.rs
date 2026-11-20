//! GLM-5.3's routed block as one CUDA kernel.

use sensai_guest::site::{self, Ask, Count, Input, Kind, Manifest, Offer, Place, Shape, Site};
use sensai_guest::{input, output};

const NAME: &str = "glm53_moe";
const ROUTE: &str = "glm53_route";
const EXPERTS: &str = "glm53_experts";
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

/// Rows one expert block takes, as `ROWS` in `moe.cu`.
const ROWS: usize = 4;
/// Past these, the linked rules serve better or `moe.cu` cannot.
const MOST_ROWS: usize = 16;
const MOST_EXPERTS: usize = 2048;

/// Experts each row takes, by expert count.
fn picks(experts: usize) -> Option<u32> {
    match experts {
        256 => Some(8),
        32 => Some(2),
        _ => None,
    }
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

/// The experts this rank holds: first and count.
fn held(codes: &[&Input], experts: usize) -> Option<[usize; 2]> {
    let share = codes.first()?.share;
    if codes.iter().any(|c| c.share != share) {
        return None;
    }
    match share {
        None => Some([0, experts]),
        Some([0, first, count, whole]) if whole == experts => Some([first, count]),
        Some(_) => None,
    }
}

/// Route, experts, then finish over a rotated routed block.
fn offer(site: &Site) -> Option<Vec<Offer>> {
    let Site::Region {
        kind: Kind::Moe,
        inputs,
        len,
        kept,
    } = site
    else {
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
    // Only the code sets split: by expert.
    let whole =
        (routed.iter().enumerate()).all(|(i, r)| r.share.is_none() || [4, 7, 10].contains(&i));
    let [first, count] = held(&codes, experts)?;
    let [inner, _] = shared[0].parts?;
    let wide = shared[0].parts?[1];
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
    if !(whole && sized) {
        return None;
    }
    let [k, book] = width(&codes)?;
    let [shared_k, shared_book] = width(&shared.iter().collect::<Vec<_>>())?;
    let picks = picks(experts)?;
    let fixed = |n: &[usize]| n.iter().map(|&n| Count::Fixed(n as u32)).collect();
    let part = |i: usize, p: usize| Place::Part {
        part: [1 + ROUTED.len() + i, p],
    };
    let scores = Place::Draft { draft: 0 };
    let route = Offer {
        kernel: ROUTE,
        reads: vec![Place::Read(0), Place::Read(1)],
        len: rows * experts,
        counts: fixed(&[rows, hidden, experts]),
        shape: Shape::Grid {
            across: experts.div_ceil(8),
            down: 1,
            threads: 256,
        },
        work: (rows * experts * hidden) as f64,
        guess: 1e-12,
    };
    let reads = [Place::Read(0), scores].into_iter();
    let reads = reads.chain((2..=11).map(Place::Read));
    let reads = reads.chain((0..2).flat_map(|i| (0..3).map(move |p| part(i, p))));
    let groups = rows.div_ceil(ROWS);
    let slots = count.min(ROWS * picks as usize) + 1;
    let experts_offer = Offer {
        kernel: EXPERTS,
        reads: reads.chain([part(2, 0), part(2, 1)]).collect(),
        len: rows * (picks as usize + 1) * (ffn / 128) * hidden,
        counts: fixed(&[
            rows,
            hidden,
            ffn,
            experts,
            picks as usize,
            k as usize,
            shared_k as usize,
            book as usize,
            shared_book as usize,
            first,
            count,
            wide,
        ]),
        shape: Shape::Grid {
            across: ffn / 128,
            down: groups * slots,
            threads: 256,
        },
        work: (groups * slots * 3 * hidden * ffn) as f64,
        guess: 1e-15,
    };
    let finish = Offer {
        kernel: FINISH,
        reads: vec![
            scores,
            Place::Read(2),
            Place::Read(12),
            Place::Read(13),
            part(2, 2),
            Place::Draft { draft: 1 },
        ],
        len: *len,
        counts: fixed(&[
            rows,
            hidden,
            ffn,
            experts,
            picks as usize,
            first,
            count,
            wide,
        ]),
        shape: Shape::Grid {
            across: hidden / 128,
            down: rows,
            threads: 128,
        },
        work: *len as f64,
        guess: 1e-12,
    };
    Some(vec![route, experts_offer, finish])
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: i32) -> i32 {
    sensai_guest::alloc(len)
}

#[unsafe(no_mangle)]
pub extern "C" fn manifest() -> i64 {
    let manifest = Manifest {
        abi: sensai_guest::ABI,
        kernels: [ROUTE, EXPERTS, FINISH]
            .map(|k| site::kernel(k, k, include_str!("moe.cu")))
            .into(),
        rules: vec![site::Rule {
            name: NAME,
            site: Kind::Moe,
        }],
        graph: false,
    };
    output(serde_json::to_vec(&manifest).unwrap_or_default())
}

/// One offer for a site that fits, else `null`.
#[unsafe(no_mangle)]
pub extern "C" fn propose(at: i32, len: i32) -> i64 {
    // SAFETY: the host passes back what `alloc` returned.
    let bytes = unsafe { input(at, len) };
    let offers = (serde_json::from_slice::<Ask>(bytes).ok()).and_then(|ask| offer(&ask.site));
    output(serde_json::to_vec(&offers).unwrap_or_else(|_| b"null".to_vec()))
}
