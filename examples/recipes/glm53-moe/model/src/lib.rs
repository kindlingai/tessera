//! GLM-5.3's routed block as one CUDA kernel.

use sensai_guest::site::{self, Ask, Count, Input, Kind, Manifest, Offer, Place, Shape, Site};
use sensai_guest::{input, output};

const NAME: &str = "glm53_moe";

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

/// The widest stream and inner width `moe.cu` holds.
const WIDE: usize = 6144;
const NARROW: usize = 2048;

/// Experts each row takes, by expert count.
fn picks(experts: usize) -> Option<u32> {
    match experts {
        256 => Some(8),
        32 => Some(2),
        _ => None,
    }
}

/// The code width a trellis form names.
fn bits(input: &Input) -> Option<u32> {
    let k: u32 = input.form.strip_prefix("trellis_k")?.parse().ok()?;
    (2..=4).contains(&k).then_some(k)
}

/// Whether `input` is the whole weight `field`.
fn named(input: &Input, field: &str) -> bool {
    let path = input.weight.as_deref().unwrap_or_default();
    path.ends_with(&format!(".{field}")) && input.share.is_none()
}

/// One width for all of `codes`.
fn width(codes: &[&Input]) -> Option<u32> {
    let k = bits(codes.first()?)?;
    codes.iter().all(|c| bits(c) == Some(k)).then_some(k)
}

/// One launch over a rotated routed block, else nothing.
fn offer(site: &Site) -> Option<Offer> {
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
    let [experts, ffn] = *routed[5].shape.as_slice() else {
        return None;
    };
    let sized = routed[0].shape == [experts, hidden]
        && shared[0].parts == Some([hidden, ffn])
        && hidden <= WIDE
        && ffn <= NARROW
        && experts <= NARROW
        && hidden % 128 == 0
        && ffn % 128 == 0;
    if !(plain && turned && fits && sized) {
        return None;
    }
    let k = width(&[&routed[4], &routed[7], &routed[10]])?;
    let shared_k = width(&shared.iter().collect::<Vec<_>>())?;
    let sizes = [rows, hidden, ffn, experts].map(|n| n as u32);
    let counts = [picks(experts)?, k, shared_k];
    let whole = (0..=ROUTED.len()).map(Place::Read);
    let parts = (0..SHARED.len()).flat_map(|i| {
        (0..3).map(move |p| Place::Part {
            part: [ROUTED.len() + 1 + i, p],
        })
    });
    Some(Offer {
        kernel: NAME,
        reads: whole.chain(parts).collect(),
        len: *len,
        counts: sizes.into_iter().chain(counts).map(Count::Fixed).collect(),
        shape: Shape::Grid {
            across: 1,
            down: rows,
            threads: 256,
        },
        work: *len as f64,
        guess: 1e-15,
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn alloc(len: i32) -> i32 {
    sensai_guest::alloc(len)
}

#[unsafe(no_mangle)]
pub extern "C" fn manifest() -> i64 {
    let manifest = Manifest {
        abi: sensai_guest::ABI,
        kernels: vec![site::kernel(NAME, NAME, include_str!("moe.cu"))],
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
    let offers = (serde_json::from_slice::<Ask>(bytes).ok())
        .and_then(|ask| offer(&ask.site))
        .map(|o| vec![o]);
    output(serde_json::to_vec(&offers).unwrap_or_else(|_| b"null".to_vec()))
}
