//! The exlr8 trellis form: K-bit codes in tiles.

use crate::form::{Form, Read, form};
use sensai_graph::{DType, E4M3, F16};

/// The mcg codebook's multiplier.
pub const MCG: u32 = 0xCBAC_1FED;

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

/// The hash from a trellis state to its weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codebook {
    Mcg,
    Mul1,
}

impl Codebook {
    /// What a release appends to its tensor names.
    pub fn suffix(self) -> &'static str {
        match self {
            Codebook::Mcg => "",
            Codebook::Mul1 => ".mul1",
        }
    }

    /// A state's weight: the hash, then E4M3 rounding.
    pub fn value(self, state: u32) -> f32 {
        let state = state & 0xFFFF;
        let v = match self {
            Codebook::Mcg => {
                let x = state.wrapping_mul(MCG);
                let x = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60;
                let low = f32::from(F16((x & 0xFFFF) as u16));
                let high = f32::from(F16((x >> 16) as u16));
                f32::from(F16::from(low + high))
            }
            Codebook::Mul1 => {
                let x = state.wrapping_mul(MUL1);
                let sum: u32 = x.to_le_bytes().iter().map(|b| u32::from(*b)).sum();
                // Exact in f32: one rounding, like a fused multiply-add.
                let v = (1024 + sum) as f32 * MUL1_SCALE + MUL1_OFFSET;
                f32::from(F16::from(v))
            }
        };
        f32::from(E4M3::from(v.clamp(-448.0, 448.0)))
    }
}

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
        // An entry: 128 channels by all of the other side.
        match self.side {
            INPUT_SIDE => {
                let (entry, r) = (row / ENTRY, row % ENTRY / 16);
                entry * (self.out / 2) + 64 * (col / ENTRY) + 8 * r + col % ENTRY / 16
            }
            _ => {
                let (entry, c) = (col / ENTRY, col % ENTRY / 16);
                entry * (self.inp / 2) + 8 * (row / 16) + c
            }
        }
    }
}

/// Stream position of tile element `(r, c)`.
fn position(r: usize, c: usize) -> usize {
    let t = 4 * (c % 8) + (r % 8) / 2;
    let s = 4 * usize::from(c >= 8) + 2 * usize::from(r >= 8) + r % 2;
    8 * t + s
}

/// Bit `b` of a tile: word pairs swapped, high first.
fn bit(tile: &[u8], b: usize) -> u32 {
    let word = (b / 16) ^ 1;
    let w = u16::from_le_bytes([tile[2 * word], tile[2 * word + 1]]);
    u32::from(w >> (15 - b % 16)) & 1
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

    /// Bytes one tile of 256 codes takes.
    const TILE: usize = 32 * K as usize;

    /// The code at stream position `j` of a tile.
    fn code(tile: &[u8], j: usize) -> u32 {
        let (k, b) = (K as usize, j * K as usize);
        if 16 % k != 0 {
            return (0..k).fold(0, |v, i| (v << 1) | bit(tile, b + i));
        }
        // Codes that divide a word sit in one, high first.
        let word = (b / 16) ^ 1;
        let w = u16::from_le_bytes([tile[2 * word], tile[2 * word + 1]]);
        u32::from(w >> (16 - k - b % 16)) & ((1 << k) - 1)
    }

    /// Element `index` of a `[out][in]` matrix in `bytes`.
    fn decode(&self, bytes: &[u8], index: usize) -> f32 {
        let shape = Shape::read(bytes);
        let (col, row) = (index / shape.inp, index % shape.inp);
        let at = HEAD + shape.tile(row, col) * Self::TILE;
        let tile = &bytes[at..at + Self::TILE];
        let j = position(row % 16, col % 16);
        // A state: the 16 bits ending at its value, wrapping.
        let mut state = 0;
        for t in 0..16usize.div_ceil(K as usize) {
            state |= Self::code(tile, (j + 256 - t) % 256) << (t * K as usize);
        }
        self.codebook.value(state)
    }
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
        let t: Vec<f32> = (0..n).map(|i| self.decode(bytes, i)).collect();
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
        // Outputs lead, so an output split has one outer part.
        let outputs = match (outer, inner) {
            (1, i) if i == shape.inp => true,
            (o, 1) if o == shape.out => false,
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
        let mut codes = Vec::with_capacity(entries * take * Self::TILE);
        for g in 0..entries {
            let from = HEAD + (g * per + at) * Self::TILE;
            codes.extend_from_slice(&bytes[from..from + take * Self::TILE]);
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

/// The form for `k` bits a weight in `codebook`.
pub fn form_for(k: u32, codebook: Codebook) -> Option<&'static dyn Form> {
    match (k, codebook) {
        (2, Codebook::Mcg) => Some(&TRELLIS_K2),
        (3, Codebook::Mcg) => Some(&TRELLIS_K3),
        (4, Codebook::Mcg) => Some(&TRELLIS_K4),
        (2, Codebook::Mul1) => Some(&TRELLIS_K2_MUL1),
        (3, Codebook::Mul1) => Some(&TRELLIS_K3_MUL1),
        (4, Codebook::Mul1) => Some(&TRELLIS_K4_MUL1),
        _ => None,
    }
}

/// The rotated form for `k` bits a weight in `codebook`.
pub fn turned_for(k: u32, codebook: Codebook) -> Option<&'static dyn Form> {
    match (k, codebook) {
        (2, Codebook::Mcg) => Some(&TRELLIS_K2_ROT),
        (3, Codebook::Mcg) => Some(&TRELLIS_K3_ROT),
        (4, Codebook::Mcg) => Some(&TRELLIS_K4_ROT),
        (2, Codebook::Mul1) => Some(&TRELLIS_K2_MUL1_ROT),
        (3, Codebook::Mul1) => Some(&TRELLIS_K3_MUL1_ROT),
        (4, Codebook::Mul1) => Some(&TRELLIS_K4_MUL1_ROT),
        _ => None,
    }
}

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
    for block in 0..inp / ENTRY {
        for col in 0..out {
            hadamard(&mut w, ENTRY * block * out + col, out);
        }
    }
    for (row, s) in suh.iter().enumerate() {
        w[row * out..(row + 1) * out]
            .iter_mut()
            .for_each(|v| *v *= s);
    }
    for row in 0..inp {
        for block in 0..out / ENTRY {
            hadamard(&mut w, row * out + ENTRY * block, 1);
        }
    }
    let mut back = vec![0.0; inp * out];
    for row in 0..inp {
        for (col, s) in svh.iter().enumerate() {
            back[col * inp + row] = w[row * out + col] * s;
        }
    }
    back
}

/// The four trellis forms at width `$k`, mcg and mul1.
macro_rules! trellis {
    ($($k:literal: $plain:ident, $turned:ident, $mul1:ident, $mul1_turned:ident;)*) => {$(
        #[doc = concat!("exlr8 at ", $k, " bits a weight.")]
        pub const $plain: Trellis<$k> = Trellis::named(concat!("trellis_k", $k), Codebook::Mcg);
        #[doc = concat!("exlr8 at ", $k, " bits, between its rotations.")]
        pub const $turned: Trellis<$k> =
            Trellis::turned(concat!("trellis_k", $k, "_rot"), Codebook::Mcg);
        #[doc = concat!("exlr8 at ", $k, " bits a weight, mul1 codebook.")]
        pub const $mul1: Trellis<$k> =
            Trellis::named(concat!("trellis_k", $k, "_mul1"), Codebook::Mul1);
        #[doc = concat!("exlr8 at ", $k, " bits, mul1, between its rotations.")]
        pub const $mul1_turned: Trellis<$k> =
            Trellis::turned(concat!("trellis_k", $k, "_mul1_rot"), Codebook::Mul1);
        form!($plain);
        form!($turned);
        form!($mul1);
        form!($mul1_turned);
    )*};
}

trellis! {
    2: TRELLIS_K2, TRELLIS_K2_ROT, TRELLIS_K2_MUL1, TRELLIS_K2_MUL1_ROT;
    3: TRELLIS_K3, TRELLIS_K3_ROT, TRELLIS_K3_MUL1, TRELLIS_K3_MUL1_ROT;
    4: TRELLIS_K4, TRELLIS_K4_ROT, TRELLIS_K4_MUL1, TRELLIS_K4_MUL1_ROT;
}
