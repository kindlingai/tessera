//! The structs, codes and limits the host and kernels share.

pub use crate::abi::Layout;
use crate::abi::{Body, Decl, Entry, Member, Named, Struct};
use std::sync::LazyLock;
use tessera_mem::form::Form as Storage;

/// This file, for the generated header to name.
pub(crate) const SOURCE: &str = file!();

/// Declares `u32` constants for kernels, and typed ones here.
macro_rules! numbers {
    ($(#[doc = $about:literal])* $list:ident {
        $($(#[doc = $doc:literal])* $name:ident: $ty:ty = $value:expr),* $(,)?
    }) => {
        $($(#[doc = $doc])* pub const $name: $ty = $value;)*

        $(#[doc = $about])*
        pub const $list: Decl = Decl {
            about: &[$($about),*],
            body: Body::Numbers(&[$((stringify!($name), $name as u32)),*]),
        };
    };
}

/// Declares enum `$c` for kernels, and its codes here.
macro_rules! codes {
    ($(#[doc = $about:literal])* $list:ident = $c:literal {
        $($name:ident = $value:literal),* $(,)?
    }) => {
        $(pub const $name: u32 = $value;)*

        $(#[doc = $about])*
        pub const $list: Decl = Decl {
            about: &[$($about),*],
            body: Body::Codes($c, &[$((stringify!($name), $name)),*]),
        };
    };
}

/// One field of `$owner`. It peels array lengths, outermost first.
macro_rules! member {
    ($owner:ident, $field:ident, [$($dims:expr),*], [$elem:tt; $len:expr]) => {
        member!($owner, $field, [$($dims,)* stringify!($len)], $elem)
    };
    ($owner:ident, $field:ident, [$($dims:expr),*], $elem:ty) => {
        Member {
            name: stringify!($field),
            ty: <$elem as Named>::C,
            dims: &[$($dims),*],
            offset: std::mem::offset_of!($owner, $field),
        }
    };
}

/// Declares struct `$c` for kernels, and `$name` here.
macro_rules! layout {
    ($(#[doc = $about:literal])* $name:ident = $c:literal $(align($align:literal))? {
        $($field:ident: $ty:tt),* $(,)?
    }) => {
        $(#[doc = $about])*
        #[repr(C $(, align($align))?)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name {
            $(pub $field: $ty),*
        }

        impl Named for $name {
            const C: &'static str = $c;
        }

        impl Layout for $name {
            const DECL: Decl = Decl {
                about: &[$($about),*],
                body: Body::Struct(Struct {
                    name: $c,
                    align: 0 $(+ $align)?,
                    size: size_of::<$name>(),
                    members: &[$(member!($name, $field, [], $ty)),*],
                }),
            };
        }
    };
}

numbers!(LIMITS {
    /// The edition every import path names.
    EDITION: u32 = 1,
    ARENAS: usize = 16,
    MAX_GROUP: usize = 8,
    MAX_POOLS: usize = 4,
    MAX_DEPTH: usize = 4,
    VALUES: usize = 58,
    CONVENTIONS: usize = 16,
    NONE: u32 = u32::MAX,
});

numbers!(
    /// Doorbell page offsets, in bytes.
    DOORBELL {
        /// A trigger word: valid bit, 15-bit lane, low ordinal, count.
        DOORBELL_SLOTS: usize = 64,
        DOORBELL_ABORT: usize = 4032,
        DOORBELL_SENT: usize = 4040,
        DOORBELL_FAULT: usize = 4072,
    }
);

/// The page, and where the host puts all-to-all sizes.
pub const PAGE: usize = 4096;
pub const SIZES: usize = 1024;

const _: () = assert!(
    DOORBELL_SLOTS + MAX_POOLS * MAX_DEPTH * 16 <= SIZES
        && SIZES + MAX_POOLS * MAX_DEPTH * MAX_GROUP * 4 <= DOORBELL_ABORT
        && DOORBELL_SENT + MAX_POOLS * 8 <= DOORBELL_FAULT
        && DOORBELL_FAULT + 8 <= PAGE
);

codes!(DTYPE = "dtype" {
    F32 = 0,
    BF16 = 1,
    F16 = 2,
    E4M3 = 3,
    U32 = 4,
    FORM = 5,
    BYTES = 6,
});

codes!(ITEM_KIND = "item_kind" {
    INPUT = 0,
    OUTPUT = 1,
    SCRATCH = 2,
});

codes!(CONVENTION = "convention" {
    LLM = 0,
    KV = 1,
    STATE = 2,
    IMAGE = 3,
});

codes!(STATUS = "status_code" {
    OK = 0,
    ABI_MISMATCH = 1,
    PEER_LOST = 2,
    BAD_INPUT = 3,
    TIMEOUT = 4,
    PROTOCOL = 5,
});

codes!(OP = "op" {
    SUM = 0,
    GATHER = 1,
    ALL_TO_ALL = 2,
    SEND = 3,
    RECV = 4,
    FETCH = 5,
});

/// A packed form: its header name, itself, its code.
type Kind = (&'static str, &'static dyn Storage, u32);

macro_rules! block_kinds {
    ($(($name:ident, $text:literal, $kind:literal, $($rest:tt)*),)*) => {
        const BLOCK_KINDS: &[Kind] = &[$((
            concat!("FORM_", stringify!($name)),
            &tessera_mem::block::$name,
            $kind,
        )),*];
    };
}

macro_rules! trellis_kinds {
    ($(($book:ident, $suffix:literal, $k:literal, $plain:ident, $turned:ident, $kind:literal),)*) => {
        const TRELLIS_KINDS: &[Kind] = &[$((
            concat!("FORM_", stringify!($plain)),
            &tessera_mem::trellis::$plain,
            $kind,
        )),*];
    };
}

tessera_mem::block_forms!(block_kinds);
tessera_mem::trellis_forms!(trellis_kinds);

fn kinds() -> impl Iterator<Item = &'static Kind> {
    BLOCK_KINDS.iter().chain(TRELLIS_KINDS)
}

/// `form_kind`, in code order. Zero is no packed form.
pub(crate) static FORM_KINDS: LazyLock<Vec<Entry>> = LazyLock::new(|| {
    let mut list = vec![("FORM_UNKNOWN", 0)];
    list.extend(kinds().map(|k| (k.0, k.2)));
    list.sort_by_key(|e| e.1);
    list
});

fn form_kinds() -> Decl {
    Decl {
        about: &[],
        body: Body::Codes("form_kind", &FORM_KINDS),
    }
}

/// Plain form name as a `dtype` code.
pub fn dtype_code(name: &str) -> u32 {
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

/// Packed form name as a `form_kind`. Rotated forms share one.
pub fn form_kind(name: &str) -> u32 {
    let base = name.strip_suffix("_rot").unwrap_or(name);
    let found = kinds().find(|k| k.1.name() == base);
    found.map_or(0, |k| k.2)
}

/// The declarations `core.h` reads.
pub(crate) fn core() -> Vec<Decl> {
    vec![
        LIMITS,
        Ref::DECL,
        DTYPE,
        ITEM_KIND,
        form_kinds(),
        Item::DECL,
        Form::DECL,
        Launch::DECL,
        CONVENTION,
        ConventionRef::DECL,
        STATUS,
        Fault::DECL,
        DOORBELL,
        Table::DECL,
    ]
}

pub(crate) fn grid() -> Vec<Decl> {
    vec![Barrier::DECL]
}

pub(crate) fn collectives() -> Vec<Decl> {
    vec![
        OP,
        Flag::DECL,
        Lane::DECL,
        Pool::DECL,
        Counters::DECL,
        PoolSet::DECL,
    ]
}

pub(crate) fn kept() -> Vec<Decl> {
    vec![Kept::DECL, KvTables::DECL, Request::DECL]
}

pub(crate) fn state() -> Vec<Decl> {
    vec![Held::DECL, StateTables::DECL]
}

pub(crate) fn llm() -> Vec<Decl> {
    vec![LlmStep::DECL]
}

pub(crate) fn image() -> Vec<Decl> {
    vec![ImageStep::DECL]
}

layout!(
    /// An arena and a byte offset, or arena NONE.
    Ref = "ref" {
        arena: u32,
        unused: u32,
        offset: u64,
    }
);

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

layout!(
    /// One item the module declared, in its order.
    Item = "item" {
        at: Ref,
        kind: u32,
        elem: u32,
        dims: u32,
        shape: [u32; 4],
        share: [u32; 4],
        form: u32,
        forms: u32,
        unused: u32,
    }
);

layout!(
    /// One packed weight matrix and its rotation parts.
    Form = "form" {
        codes: Ref,
        suh: Ref,
        svh: Ref,
        kind: u32,
        rows: u32,
        cols: u32,
        unused: u32,
    }
);

layout!(
    /// Values the engine writes before each launch.
    Launch = "launch" {
        seq: u64,
        deadline: u64,
        values: [u32; VALUES],
    }
);

layout!(ConventionRef = "convention_ref" {
    at: Ref,
    version: u32,
    bytes: u32,
});

layout!(
    /// The first failure. The engine reads it after the launch.
    Fault = "fault" {
        code: u32,
        pool: u32,
        ordinal: u64,
    }
);

layout!(Table = "table" {
    edition: u32,
    rank: u32,
    ranks: u32,
    items_len: u32,
    forms_len: u32,
    lanes_len: u32,
    values_len: u32,
    unused: u32,
    arenas: [u64; ARENAS],
    now: Ref,
    items: Ref,
    forms: Ref,
    conventions: [ConventionRef; CONVENTIONS],
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

layout!(
    /// In the layout the NIC or proxy writes.
    Flag = "flag" align(16) {
        ordinal: u64,
        count: u64,
    }
);

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
    send: [Ref; MAX_DEPTH],
    recv: [Ref; MAX_DEPTH * MAX_GROUP],
    arrived: Ref,
    credit: Ref,
    sizes: Ref,
});

layout!(
    /// Device memory. Block 0 copies host state here for others.
    Counters = "counters" {
        next: [u64; MAX_POOLS],
        writes: u64,
        arrived: [[u64; MAX_DEPTH]; MAX_POOLS],
        free: [u64; MAX_POOLS],
    }
);

layout!(PoolSet = "pool_set" {
    count: u32,
    unused: u32,
    state: Ref,
    pools: [Pool; MAX_POOLS],
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

layout!(
    /// One fetch request. Kept NONE means skip.
    Request = "request" align(16) {
        kept: u32,
        slot: u32,
        pos: u32,
        unused: u32,
    }
);

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
