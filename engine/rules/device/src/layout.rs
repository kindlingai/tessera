//! The structs the host writes, laid out as the headers.

/// Limits and doorbell page offsets from `core.h`.
pub(crate) const ARENAS: usize = 16;
pub(crate) const MAX_GROUP: usize = 8;
pub(crate) const MAX_POOLS: usize = 4;
pub const MAX_DEPTH: usize = 4;
pub const VALUES: usize = 58;
pub const NONE: u32 = u32::MAX;
pub(crate) const DOORBELL_SLOTS: usize = 64;
pub(crate) const DOORBELL_ABORT: usize = 4032;
pub(crate) const DOORBELL_SENT: usize = 4040;
pub(crate) const DOORBELL_FAULT: usize = 4072;
/// `dtype` codes matching `core.h`.
pub(crate) const F32: u32 = 0;
pub(crate) const BF16: u32 = 1;
pub(crate) const F16: u32 = 2;
pub(crate) const E4M3: u32 = 3;
pub(crate) const U32: u32 = 4;
pub(crate) const FORM: u32 = 5;
pub(crate) const BYTES: u32 = 6;
/// `dtype` names in code order, with bytes per element.
pub(crate) const DTYPES: [(&str, u32); 7] = [
    ("F32", 4),
    ("BF16", 2),
    ("F16", 2),
    ("E4M3", 1),
    ("U32", 4),
    ("FORM", 1),
    ("BYTES", 1),
];
/// `item_kind` codes matching `core.h`.
pub(crate) const INPUT: u32 = 0;
pub(crate) const OUTPUT: u32 = 1;
pub(crate) const SCRATCH: u32 = 2;
/// `convention` code for KV tables.
pub(crate) const KV: u32 = 1;
/// `form_kind` header names and storage form names in code order.
pub(crate) const FORM_KINDS: [(&str, &str); 17] = [
    ("FORM_UNKNOWN", ""),
    ("FORM_Q8_0", "q8_0"),
    ("FORM_Q4_0", "q4_0"),
    ("FORM_NVFP4", "nvfp4"),
    ("FORM_MXFP8", "mxfp8"),
    ("FORM_MXFP4", "mxfp4"),
    ("FORM_INT4_G32", "int4_g32"),
    ("FORM_INT4_G128", "int4_g128"),
    ("FORM_TRELLIS_K2", "trellis_k2"),
    ("FORM_TRELLIS_K3", "trellis_k3"),
    ("FORM_TRELLIS_K4", "trellis_k4"),
    ("FORM_TRELLIS_K2_MUL1", "trellis_k2_mul1"),
    ("FORM_TRELLIS_K3_MUL1", "trellis_k3_mul1"),
    ("FORM_TRELLIS_K4_MUL1", "trellis_k4_mul1"),
    ("FORM_TRELLIS_K2_PMCGX", "trellis_k2_pmcgx"),
    ("FORM_TRELLIS_K3_PMCGX", "trellis_k3_pmcgx"),
    ("FORM_TRELLIS_K4_PMCGX", "trellis_k4_pmcgx"),
];

/// Plain form name as a `core.h` `dtype` code.
pub(crate) fn dtype_code(name: &str) -> u32 {
    match name {
        "f32" => F32,
        "bf16" => BF16,
        "f16" => F16,
        "e4m3" | "fp8" => E4M3,
        "u32" => U32,
        "form" => FORM,
        _ => BYTES,
    }
}

/// Packed form name as a `core.h` `form_kind` code.
pub(crate) fn form_kind(name: &str) -> u32 {
    let base = name.strip_suffix("_rot").unwrap_or(name);
    for (code, &(_, form)) in FORM_KINDS.iter().enumerate() {
        if !form.is_empty() && form == base {
            return code as u32;
        }
    }
    0
}
/// `status_code` names in code order.
pub(crate) const STATUS: [&str; 6] = [
    "OK",
    "ABI_MISMATCH",
    "PEER_LOST",
    "BAD_INPUT",
    "TIMEOUT",
    "PROTOCOL",
];
/// The page, and where the host puts all-to-all sizes.
pub const PAGE: usize = 4096;
pub const SIZES: usize = 1024;

/// One field: its name, byte offset and size.
pub type Field = (&'static str, usize, usize);

/// A struct the host and the headers share.
pub trait Layout {
    /// Its name in the headers.
    const C: &'static str;
    const FIELDS: &'static [Field];
}

macro_rules! layout {
    ($(#[$attr:meta])* $name:ident = $c:literal { $($field:ident: $ty:ty),* $(,)? }) => {
        #[repr(C)]
        $(#[$attr])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name {
            $(pub $field: $ty),*
        }

        impl Layout for $name {
            const C: &'static str = $c;
            const FIELDS: &'static [Field] = &[$((
                stringify!($field),
                std::mem::offset_of!($name, $field),
                size_of::<$ty>(),
            )),*];
        }
    };
}

layout!(Ref = "ref" {
    arena: u32,
    unused: u32,
    offset: u64,
});

impl Ref {
    pub const NONE: Ref = Ref::at(NONE, 0);

    pub const fn at(arena: u32, offset: u64) -> Ref {
        Ref {
            arena,
            unused: 0,
            offset,
        }
    }
}

layout!(Item = "item" {
    at: Ref,
    kind: u32,
    elem: u32,
    dims: u32,
    shape: [u32; 4],
    share: [u32; 4],
    form: u32,
    forms: u32,
    unused: u32,
});

layout!(Form = "form" {
    codes: Ref,
    suh: Ref,
    svh: Ref,
    kind: u32,
    rows: u32,
    cols: u32,
    unused: u32,
});

layout!(Launch = "launch" {
    seq: u64,
    deadline: u64,
    values: [u32; 58],
});

layout!(ConventionRef = "convention_ref" {
    at: Ref,
    version: u32,
    bytes: u32,
});

layout!(Fault = "fault" {
    code: u32,
    pool: u32,
    ordinal: u64,
});

layout!(Table = "table" {
    edition: u32,
    rank: u32,
    ranks: u32,
    items_len: u32,
    forms_len: u32,
    lanes_len: u32,
    values_len: u32,
    unused: u32,
    arenas: [u64; 16],
    now: Ref,
    items: Ref,
    forms: Ref,
    conventions: [ConventionRef; 16],
    lanes: Ref,
    pools: Ref,
    doorbell: Ref,
    barrier: Ref,
    status: Ref,
    marks: Ref,
});

layout!(Barrier = "barrier" {
    count: u32,
    round: u32,
});

layout!(#[repr(align(16))] Flag = "flag" {
    ordinal: u64,
    count: u64,
});

layout!(Lane = "lane" {
    what: u32,
    elem: u32,
    group_size: u32,
    group_rank: u32,
    block: u32,
    pool: u32,
    unused: [u32; 2],
});

layout!(Pool = "pool" {
    depth: u32,
    one_way: u32,
    send_bytes: u32,
    recv_bytes: u32,
    unused: [u32; 4],
    send: [Ref; 4],
    recv: [Ref; 32],
    arrived: Ref,
    credit: Ref,
    sizes: Ref,
});

layout!(Counters = "counters" {
    next: [u64; 4],
    writes: u64,
    arrived: [[u64; 4]; 4],
    free: [u64; 4],
});

layout!(PoolSet = "pool_set" {
    count: u32,
    unused: u32,
    state: Ref,
    pools: [Pool; 4],
});

layout!(Kept = "kept" {
    base: Ref,
    host: Ref,
    pages: Ref,
    len: Ref,
    pages_per_slot: u32,
    page_rows: u32,
    row_bytes: u32,
    elem: u32,
});

layout!(KvTables = "kv_tables" {
    count: u32,
    unused: u32,
    tables: Ref,
});

layout!(#[repr(align(16))] Request = "request" {
    kept: u32,
    slot: u32,
    pos: u32,
    unused: u32,
});

layout!(Held = "held" {
    base: Ref,
    fresh: Ref,
    slot_bytes: u32,
    elem: u32,
});

layout!(StateTables = "state_tables" {
    count: u32,
    unused: u32,
    tables: Ref,
});

layout!(LlmStep = "llm_step" {
    rows: u32,
    out_rows: u32,
    vocab: u32,
    unused: u32,
    token: Ref,
    slot: Ref,
    position: Ref,
    out_row: Ref,
    temperature: Ref,
    top_k: Ref,
    seed: Ref,
    logits: Ref,
    sampled: Ref,
});

layout!(ImageStep = "image_step" {
    batch: u32,
    height: u32,
    width: u32,
    channels: u32,
    timestep: Ref,
    guidance: Ref,
    seed: Ref,
});
