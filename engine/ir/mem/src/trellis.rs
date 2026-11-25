//! The exlr8 trellis form: K-bit codes in tiles.

use crate::form::{Form, Read, form};
use crate::ops::{Bits, host_fns, round_e4m3, round_f16};
use tessera_graph::{DType, F16};

/// The mcg codebook's multiplier.
pub const MCG: u32 = 0xCBAC_1FED;

/// pmcgx's second multiplier.
pub const PMCG2: u32 = 0xBCA1_4C51;

/// The mul1 codebook's multiplier.
pub const MUL1: u32 = 0x83DC_D12D;

/// mul1's scale and offset, copied from the format.
pub const MUL1_SCALE: f32 = F16(0x1EEE).to_f32();
pub const MUL1_OFFSET: f32 = F16(0xC931).to_f32();

/// Header words: `in`, `out`, the sharded side, `1 / in`.
pub const HEAD: usize = 16;

/// Record field 4: entries hold 128 input channels.
pub const INPUT_SIDE: u32 = 2;

/// Channels one entry holds on its sharded side.
const ENTRY: usize = 128;

/// `K`-bit trellis codes. Rotated files append `suh`, `svh`.
pub struct Trellis<const K: u32> {
    pub name: &'static str,
    pub rotated: bool,
    pub codebook: Codebook,
}

/// Expands `$each` over all codebooks.
// Row: variant, name, suffix, state bits, pairs, weight function.
#[macro_export]
macro_rules! codebooks {
    ($each:ident) => {
        $each! {
            (Mcg, "mcg", "", 16, false, mcg),
            (Mul1, "mul1", ".mul1", 16, false, mul1),
            (Pmcgx, "pmcgx", ".pmcgx", 20, true, pmcgx),
        }
    };
}

/// Expands `$each` over the shared host and kernel functions.
// Bodies use only kernel operations. Strings name kernel helpers.
#[macro_export]
macro_rules! trellis_fns {
    ($each:ident) => {
        $each! {
            /// An mcg state's weight: one hash, its halves summed.
            "trellis_mcg"
            fn mcg(state: u32, _odd: u32) -> f32 {
                let x = ((state & 0xFFFF).wrapping_mul(MCG) & 0x8FFF_8FFF) ^ 0x3B60_3B60;
                round_e4m3(round_f16(x.f16(0) + x.f16(16)))
            }

            /// A mul1 state's weight.
            // Exact in f32: `round_f16` alone rounds, as fused f16 does.
            "trellis_mul1"
            fn mul1(state: u32, _odd: u32) -> f32 {
                let z = (state & 0xFFFF).wrapping_mul(MUL1);
                let sum = (z & 0xFF) + ((z >> 8) & 0xFF) + ((z >> 16) & 0xFF) + (z >> 24);
                round_e4m3(round_f16((1024 + sum) as f32 * MUL1_SCALE + MUL1_OFFSET))
            }

            /// Weight `odd` of a pmcgx pair: two hashes, halves crossed.
            "trellis_pmcgx"
            fn pmcgx(state: u32, odd: u32) -> f32 {
                let w = state & 0xF_FFFF;
                let x = (w.wrapping_mul(MCG) & 0x8FFF_8FFF) ^ 0x3B60_3B60;
                let y = (w.wrapping_mul(PMCG2) & 0x8FFF_8FFF) ^ 0x3B60_3B60;
                let raw = if odd != 0 {
                    x.f16(16) + y.f16(0)
                } else {
                    x.f16(0) + y.f16(16)
                };
                round_e4m3(round_f16(raw))
            }

            /// Stream position of tile element `(r, c)`.
            "trellis_slot"
            fn slot(r: u32, c: u32) -> u32 {
                8 * (4 * (c % 8) + (r % 8) / 2) + 4 * (c >= 8) as u32 + 2 * (r >= 8) as u32 + r % 2
            }

            /// The first bit of code `j`'s state, wrapping.
            "trellis_start_k{K}_s{bits}_p{pairs}"
            fn start<const K: u32>(#[build] bits: u32, #[build] pairs: bool, j: u32) -> u32 {
                if pairs {
                    // A pair's state ends at bit 2K(p+1).
                    (2 * K * (j / 2 + 1) + 256 * K - bits) % (256 * K)
                } else {
                    // A state spans `bits / K` codes, rounded up.
                    ((j + (256 - (bits - 1) / K)) % 256) * K + (((bits - 1) / K + 1) * K - bits)
                }
            }

            /// The tile of `T[row][col]` within one matrix.
            "trellis_local"
            fn local(inp: u32, out: u32, side: u32, row: u32, col: u32) -> u32 {
                // An entry: 128 channels by all of the other side.
                let c = (col % 128) / 16;
                if (side & 0xFFFF) == INPUT_SIDE {
                    (row / 128) * (out / 2) + 64 * (col / 128) + 8 * ((row % 128) / 16) + c
                } else {
                    (col / 128) * (inp / 2) + 8 * (row / 16) + c
                }
            }
        }
    };
}

crate::trellis_fns!(host_fns);

macro_rules! define_codebook {
    ($(($book:ident, $name:literal, $suffix:literal, $bits:literal, $pairs:literal,
        $weight:ident),)*) => {
        /// The hash from a trellis state to its weight.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Codebook {
            $($book,)*
        }

        impl Codebook {
            /// Every codebook, in the order a reader tries them.
            pub const ALL: [Codebook; [$($name),*].len()] = [$(Codebook::$book),*];

            /// The codebook a manifest calls `name`.
            pub fn named(name: &str) -> Option<Self> {
                match name {
                    $($name => Some(Codebook::$book),)*
                    _ => None,
                }
            }

            /// The suffix a release appends to its tensor names.
            pub fn suffix(self) -> &'static str {
                match self {
                    $(Codebook::$book => $suffix,)*
                }
            }

            /// Bits in a state.
            pub fn state_bits(self) -> u32 {
                match self {
                    $(Codebook::$book => $bits,)*
                }
            }

            /// Whether a state holds two weights.
            pub fn pairs(self) -> bool {
                match self {
                    $(Codebook::$book => $pairs,)*
                }
            }

            /// Weight `odd` of a state's pair, or its only weight.
            pub fn weight(self, state: u32, odd: bool) -> f32 {
                match self {
                    $(Codebook::$book => $weight(state, u32::from(odd)),)*
                }
            }
        }
    };
}

crate::codebooks!(define_codebook);

/// The matrix's shape and side, from the header.
struct Shape {
    inp: usize,
    out: usize,
    side: u32,
}

impl Shape {
    fn read(bytes: &[u8]) -> Self {
        let word =
            |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().expect("4 bytes"));
        Self {
            inp: word(0) as usize,
            out: word(1) as usize,
            side: word(2),
        }
    }

    fn write(&self) -> [u8; HEAD] {
        let mut head = [0u8; HEAD];
        let inverse = (1.0 / self.inp as f32).to_bits();
        for (i, v) in [self.inp as u32, self.out as u32, self.side, inverse]
            .iter()
            .enumerate()
        {
            head[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        head
    }

    /// The tile of `T[row][col]`. Later matrices follow at `out`.
    fn tile(&self, row: usize, col: usize) -> usize {
        let (m, col) = (col / self.out, col % self.out);
        m * (self.inp * self.out / 256) + self.local(row, col)
    }

    /// The tile within one matrix.
    fn local(&self, row: usize, col: usize) -> usize {
        let [inp, out, row, col] = [self.inp, self.out, row, col].map(|v| v as u32);
        local(inp, out, self.side, row, col) as usize
    }
}

/// Stream position of tile element `(r, c)`.
fn position(r: usize, c: usize) -> usize {
    slot(r as u32, c as u32) as usize
}

/// 32 stream bits of a `k`-bit tile from bit `s`.
fn window(k: u32, tile: &[u8], s: u32) -> u32 {
    // Little-endian words hold 32 stream bits, high first.
    let word = |q: u32| {
        let at = 4 * (q % (8 * k)) as usize;
        u64::from(u32::from_le_bytes(
            tile[at..at + 4].try_into().expect("4 bytes"),
        ))
    };
    let (q, sh) = (s >> 5, s & 31);
    (((word(q) << 32 | word(q + 1)) << sh) >> 32) as u32
}

impl<const K: u32> Trellis<K> {
    pub const fn named(name: &'static str, codebook: Codebook) -> Self {
        Self {
            name,
            rotated: false,
            codebook,
        }
    }

    /// Codes between rotations: `diag(suh) H T H diag(svh)`.
    pub const fn turned(name: &'static str, codebook: Codebook) -> Self {
        Self {
            name,
            rotated: true,
            codebook,
        }
    }

    /// A rotated matrix's codes, then its `suh` and `svh`.
    fn parts(bytes: &[u8]) -> (&[u8], Vec<f32>, Vec<f32>) {
        let shape = Shape::read(bytes);
        let end = HEAD + shape.inp * shape.out * K as usize / 8;
        let (suh, svh) = bytes[end..].split_at(2 * shape.inp);
        (
            &bytes[..end],
            F16::floats(suh),
            F16::floats(&svh[..2 * shape.out]),
        )
    }

    /// The size of one tile of 256 codes, in bytes.
    const TILE: usize = 32 * K as usize;

    /// Element `index` of a `[out][in]` matrix in `bytes`.
    fn decode(&self, bytes: &[u8], index: usize) -> f32 {
        let shape = Shape::read(bytes);
        let (col, row) = (index / shape.inp, index % shape.inp);
        let at = HEAD + shape.tile(row, col) * Self::TILE;
        let tile = &bytes[at..at + Self::TILE];
        let j = position(row % 16, col % 16) as u32;
        let (bits, pairs) = (self.codebook.state_bits(), self.codebook.pairs());
        let state = window(K, tile, start::<K>(bits, pairs, j)) >> (32 - bits);
        self.codebook.weight(state, pairs && j % 2 == 1)
    }

    /// The first `n` decoded elements, a tile at a time.
    fn decode_all(&self, bytes: &[u8], n: usize) -> Vec<f32> {
        let shape = Shape::read(bytes);
        let inp = shape.inp;
        let mut t = vec![0.0; n];
        // Sixteen outputs a piece: one column of tiles.
        spread(&mut t, 16 * inp, |c16, piece| {
            for r16 in 0..inp / 16 {
                let at = HEAD + shape.tile(16 * r16, 16 * c16) * Self::TILE;
                let pairs = self.codebook.pairs();
                let tile = &bytes[at..at + Self::TILE];
                let w = stream(K as usize, pairs, tile, |s, odd| {
                    self.codebook.weight(s, odd)
                });
                for c in 0..16 {
                    for r in 0..16 {
                        piece[c * inp + 16 * r16 + r] = w[position(r, c)];
                    }
                }
            }
        });
        t
    }
}

/// A tile's 256 weights at `k` bits, in stream order.
fn stream(k: usize, pairs: bool, tile: &[u8], weight: impl Fn(u32, bool) -> f32) -> [f32; 256] {
    // Little-endian words hold 32 stream bits, high first.
    let word = |w: usize| u32::from_le_bytes(tile[4 * w..4 * w + 4].try_into().expect("4 bytes"));
    let step = if pairs { 2 * k } else { k };
    // The ring wraps: its last bits lead the first state.
    let mut bits = u64::from(word(8 * k - 1));
    let (mut held, mut next, mut n) = (0, 0, 0);
    let mut out = [0.0; 256];
    while n < 256 {
        if held < step {
            bits = (bits << 32) | u64::from(word(next));
            (held, next) = (held + 32, next + 1);
        }
        held -= step;
        let state = (bits >> held) as u32;
        out[n] = weight(state, false);
        if pairs {
            out[n + 1] = weight(state, true);
        }
        n += 1 + usize::from(pairs);
    }
    out
}

/// Row trellis rows `suh[c] (H v_r)[c] svh[r]`, through `each`.
pub fn rows<T: Clone + Send>(
    k: u32,
    codebook: Codebook,
    codes: &[u8],
    [suh, svh]: [&[f32]; 2],
    each: impl Fn(&f32) -> T + Sync,
) -> Option<Vec<T>> {
    // None for a width or codebook this engine lacks.
    form_for(k, codebook)?;
    let (k, width) = (k as usize, suh.len());
    let (tile, pairs) = (32 * k, codebook.pairs());
    let bytes = width / 256 * tile;
    // Each row must hold whole tiles, else None.
    if !width.is_multiple_of(256) || codes.len() != svh.len() * bytes {
        return None;
    }
    // Every state's weights, so a value is one lookup.
    let mask = (1u32 << codebook.state_bits()) - 1;
    let both = |s: u32| [false, true].map(|odd| codebook.weight(s, odd));
    let table: Vec<[f32; 2]> = (0..=mask).map(both).collect();
    let weight = |s: u32, odd: bool| table[(s & mask) as usize][usize::from(odd)];
    let mut out = vec![each(&0.0); width * svh.len()];
    spread(&mut out, width, |r, row| {
        let mut values = vec![0.0; width];
        let codes = &codes[r * bytes..(r + 1) * bytes];
        for (t, part) in values.chunks_mut(256).enumerate() {
            part.copy_from_slice(&stream(k, pairs, &codes[t * tile..(t + 1) * tile], weight));
        }
        for block in 0..width / ENTRY {
            hadamard(&mut values, ENTRY * block, 1);
        }
        for ((o, v), s) in row.iter_mut().zip(values).zip(suh) {
            *o = each(&(v * s * svh[r]));
        }
    });
    Some(out)
}

/// Runs `work` on each `size` piece, over all cores.
fn spread<T: Send>(data: &mut [T], size: usize, work: impl Fn(usize, &mut [T]) + Sync) {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let count = data.len().div_ceil(size);
    let share = count.div_ceil(cores) * size;
    std::thread::scope(|s| {
        for (i, part) in data.chunks_mut(share.max(size)).enumerate() {
            let work = &work;
            s.spawn(move || {
                for (j, piece) in part.chunks_mut(size).enumerate() {
                    work(i * share / size + j, piece);
                }
            });
        }
    });
}

impl<const K: u32> Form for Trellis<K> {
    fn name(&self) -> &'static str {
        self.name
    }
    fn dtype(&self) -> DType {
        DType::F32
    }
    fn read(&self) -> Read {
        Read::Decode
    }
    fn bits(&self) -> u32 {
        K
    }
    fn narrow(&self) -> bool {
        true
    }
    /// A tile, so no float row copies in as is.
    fn block(&self) -> usize {
        256
    }
    fn packs(&self) -> bool {
        false
    }
    fn align(&self) -> usize {
        ENTRY
    }
    fn bytes(&self, elements: usize) -> usize {
        HEAD + elements * K as usize / 8
    }
    fn offset(&self, index: usize) -> usize {
        match index {
            0 => 0,
            _ => HEAD + index * K as usize / 8,
        }
    }
    fn elements(&self, offset: usize) -> usize {
        match offset {
            0 => 0,
            _ => (offset - HEAD) * 8 / K as usize,
        }
    }
    fn element_bytes(&self) -> f64 {
        f64::from(K) / 8.0
    }
    /// A rotated element needs the whole matrix decoded.
    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        match self.rotated {
            true => self.unpack_all(bytes)[index],
            false => self.decode(bytes, index),
        }
    }
    fn unpack_all(&self, bytes: &[u8]) -> Vec<f32> {
        let shape = Shape::read(bytes);
        let n = match self.rotated {
            true => shape.inp * shape.out,
            false => self.elements(bytes.len()),
        };
        let t = self.decode_all(bytes, n);
        if !self.rotated {
            return t;
        }
        let (_, suh, svh) = Self::parts(bytes);
        rotated(&t, shape.inp, shape.out, &suh, &svh)
    }
    /// Plain trellis codes, then `suh` and `svh` as f32.
    fn split(&self, bytes: &[u8]) -> Option<Vec<Vec<u8>>> {
        let floats = |v: Vec<f32>| v.iter().flat_map(|x| x.to_le_bytes()).collect();
        if !self.rotated {
            return None;
        }
        let (codes, suh, svh) = Self::parts(bytes);
        Some(vec![codes.to_vec(), floats(suh), floats(svh)])
    }
    fn pack(&self, _: &[f32]) -> Vec<u8> {
        panic!("{} codes are read from a release, never packed", self.name)
    }
    /// A set's whole matrices, or a matrix's whole entries.
    fn shares(&self, shape: &[usize], axis: usize, count: usize) -> bool {
        match shape.len() {
            2 => count.is_multiple_of(ENTRY),
            3 => axis == 0 || count.is_multiple_of(ENTRY),
            _ => axis == 0,
        }
    }
    /// Whole matrices of a set, or whole stored-side entries.
    fn share(
        &self,
        bytes: &[u8],
        [outer, _, inner]: [usize; 3],
        [first, count]: [usize; 2],
    ) -> Option<Vec<u8>> {
        if self.rotated {
            return None;
        }
        let shape = Shape::read(bytes);
        let matrix = shape.inp * shape.out * K as usize / 8;
        if (outer, inner) == (1, shape.inp * shape.out) {
            let at = HEAD + first * matrix;
            let set = &bytes[at..at + count * matrix];
            return Some(packed(shape.inp, shape.out, shape.side, set));
        }
        let input = shape.side == INPUT_SIDE;
        if !first.is_multiple_of(ENTRY) || !count.is_multiple_of(ENTRY) {
            return None;
        }
        let body = bytes.len().checked_sub(HEAD)?;
        // Outputs lead, so an output split has `matrices` outer parts.
        let (outputs, matrices) = match (outer, inner) {
            (m, i) if i == shape.inp && m * matrix == body => (true, m),
            (o, 1) if o.is_multiple_of(shape.out) && (o / shape.out) * matrix == body => {
                (false, o / shape.out)
            }
            _ => return None,
        };
        let part = match outputs {
            true => Shape {
                out: count,
                ..shape
            },
            false => Shape {
                inp: count,
                ..shape
            },
        };
        // `take` tiles from `at` in each run of `per`.
        let (entries, per, at, take) = match (outputs, input) {
            // Along the stored side: whole entries, in one run.
            (true, false) | (false, true) => {
                let tiles = if input { shape.out / 2 } else { shape.inp / 2 };
                (1, 0, first / ENTRY * tiles, count / ENTRY * tiles)
            }
            // Outputs of input entries: 64 tiles per 128 outputs.
            (true, true) => (shape.inp / ENTRY, shape.out / 2, first / 2, count / 2),
            // Inputs of output entries: 8 tiles per 16 inputs.
            (false, false) => (shape.out / ENTRY, shape.inp / 2, first / 2, count / 2),
        };
        let mut codes = Vec::with_capacity(matrices * entries * take * Self::TILE);
        for m in 0..matrices {
            let base = HEAD + m * matrix;
            for g in 0..entries {
                let from = base + (g * per + at) * Self::TILE;
                codes.extend_from_slice(&bytes[from..from + take * Self::TILE]);
            }
        }
        Some(packed(part.inp, part.out, part.side, &codes))
    }
}

/// The header, then the entries in the release's layout.
pub fn packed(inp: usize, out: usize, side: u32, entries: &[u8]) -> Vec<u8> {
    let shape = Shape { inp, out, side };
    let mut bytes = Vec::with_capacity(HEAD + entries.len());
    bytes.extend_from_slice(&shape.write());
    bytes.extend_from_slice(entries);
    bytes
}

/// All trellis forms for `$each`, each with its `form_kind`.
#[macro_export]
macro_rules! trellis_forms {
    ($each:ident) => {
        $each! {
            (Mcg, "", 2, TRELLIS_K2, TRELLIS_K2_ROT, 8),
            (Mcg, "", 3, TRELLIS_K3, TRELLIS_K3_ROT, 9),
            (Mcg, "", 4, TRELLIS_K4, TRELLIS_K4_ROT, 10),
            (Mul1, "_mul1", 2, TRELLIS_K2_MUL1, TRELLIS_K2_MUL1_ROT, 11),
            (Mul1, "_mul1", 3, TRELLIS_K3_MUL1, TRELLIS_K3_MUL1_ROT, 12),
            (Mul1, "_mul1", 4, TRELLIS_K4_MUL1, TRELLIS_K4_MUL1_ROT, 13),
            (Pmcgx, "_pmcgx", 2, TRELLIS_K2_PMCGX, TRELLIS_K2_PMCGX_ROT, 14),
            (Pmcgx, "_pmcgx", 3, TRELLIS_K3_PMCGX, TRELLIS_K3_PMCGX_ROT, 15),
            (Pmcgx, "_pmcgx", 4, TRELLIS_K4_PMCGX, TRELLIS_K4_PMCGX_ROT, 16),
        }
    };
}

macro_rules! lookup_trellis {
    ($(($book:ident, $suffix:literal, $k:literal, $plain:ident, $turned:ident, $kind:literal),)*) => {
        /// The form for `k` bits a weight in `codebook`.
        pub fn form_for(k: u32, codebook: Codebook) -> Option<&'static dyn Form> {
            match (k, codebook) {
                $(($k, Codebook::$book) => Some(&$plain),)*
                _ => None,
            }
        }

        /// The rotated form for `k` bits a weight in `codebook`.
        pub fn turned_for(k: u32, codebook: Codebook) -> Option<&'static dyn Form> {
            match (k, codebook) {
                $(($k, Codebook::$book) => Some(&$turned),)*
                _ => None,
            }
        }
    };
}

crate::trellis_forms!(lookup_trellis);

/// The scaled Walsh-Hadamard transform of 128 values at `stride`.
fn hadamard(x: &mut [f32], start: usize, stride: usize) {
    let mut len = 1;
    while len < ENTRY {
        for i in (0..ENTRY).step_by(2 * len) {
            for j in i..i + len {
                let (a, b) = (x[start + j * stride], x[start + (j + len) * stride]);
                x[start + j * stride] = a + b;
                x[start + (j + len) * stride] = a - b;
            }
        }
        len *= 2;
    }
    let scale = 1.0 / (ENTRY as f32).sqrt();
    for j in 0..ENTRY {
        x[start + j * stride] *= scale;
    }
}

/// `diag(suh) H T H diag(svh)` for a `[out][in]` matrix `t`.
pub fn rotated(t: &[f32], inp: usize, out: usize, suh: &[f32], svh: &[f32]) -> Vec<f32> {
    // Work on T as `(in, out)`.
    let mut w: Vec<f32> = (0..inp * out)
        .map(|i| t[(i % out) * inp + i / out])
        .collect();
    spread(&mut w, ENTRY * out, |_, rows| {
        for col in 0..out {
            hadamard(rows, col, out);
        }
    });
    spread(&mut w, out, |row, values| {
        values.iter_mut().for_each(|v| *v *= suh[row]);
        for block in 0..out / ENTRY {
            hadamard(values, ENTRY * block, 1);
        }
    });
    let mut back = vec![0.0; inp * out];
    for row in 0..inp {
        for (col, s) in svh.iter().enumerate() {
            back[col * inp + row] = w[row * out + col] * s;
        }
    }
    back
}

macro_rules! define_trellis {
    ($(($book:ident, $suffix:literal, $k:literal, $plain:ident, $turned:ident, $kind:literal),)*) => {$(
        #[doc = concat!("exlr8 at ", $k, " bits a weight, ", stringify!($book), ".")]
        pub const $plain: Trellis<$k> =
            Trellis::named(concat!("trellis_k", $k, $suffix), Codebook::$book);
        #[doc = concat!("exlr8 at ", $k, " bits, ", stringify!($book), ", between rotations.")]
        pub const $turned: Trellis<$k> =
            Trellis::turned(concat!("trellis_k", $k, $suffix, "_rot"), Codebook::$book);
        form!($plain);
        form!($turned);
    )*};
}

crate::trellis_forms!(define_trellis);
