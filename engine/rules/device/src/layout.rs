//! The structs the host writes, laid out as the headers.

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
        #[derive(Clone, Copy, Debug)]
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
