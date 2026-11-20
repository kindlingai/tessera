//! Storage forms: how buffers hold elements, and how bytes decode.

use sensai_graph::{Bf16 as Half, DType};

pub use scattered_collect::ScatteredSlice;
pub use scattered_collect::declarative::{gather, scatter};

/// How host code reads one element of a form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Read {
    F32,
    Bf16,
    U32,
    U8,
    /// Through the form's own `unpack` and `dot`, as an f32.
    Decode,
}

/// Packs and unpacks its elements on the host.
pub trait Form: Sync {
    fn name(&self) -> &'static str;
    /// The element type, for moving and converting bytes.
    fn dtype(&self) -> DType;
    /// The host reader for an element.
    fn read(&self) -> Read;
    /// Significant bits an element keeps.
    fn bits(&self) -> u32;
    /// Element `index` of `bytes`, as a float.
    fn unpack(&self, bytes: &[u8], index: usize) -> f32;
    /// `data` as the bytes of this form.
    fn pack(&self, data: &[f32]) -> Vec<u8>;

    /// Whether floats can become this form. Codes cannot.
    fn packs(&self) -> bool {
        true
    }

    /// Elements that share one scale, decoded together.
    fn block(&self) -> usize {
        1
    }

    /// Elements a row's length must be a multiple of.
    fn align(&self) -> usize {
        self.block()
    }

    /// Middle-axis parts `first..first + count` of `[outer, whole, inner]`.
    fn share(
        &self,
        bytes: &[u8],
        [outer, whole, inner]: [usize; 3],
        [first, count]: [usize; 2],
    ) -> Option<Vec<u8>> {
        let (row, at, len) = (whole * inner, first * inner, count * inner);
        if [row, at, len]
            .iter()
            .any(|n| !n.is_multiple_of(self.block()))
        {
            return None;
        }
        // A header before the first element stays.
        let mut out = bytes[..self.bytes(0)].to_vec();
        for o in 0..outer {
            let from = o * row + at;
            out.extend_from_slice(&bytes[self.bytes(from)..self.bytes(from + len)]);
        }
        Some(out)
    }

    /// Parts a file's bytes hold, each in its own form.
    fn split(&self, _bytes: &[u8]) -> Option<Vec<Vec<u8>>> {
        None
    }

    /// Every element `bytes` hold, in order.
    fn unpack_all(&self, bytes: &[u8]) -> Vec<f32> {
        (0..self.elements(bytes.len()))
            .map(|i| self.unpack(bytes, i))
            .collect()
    }

    /// `data` as this form keeps it.
    fn roundtrip(&self, data: &[f32]) -> Vec<f32> {
        let packed = self.pack(data);
        (0..data.len()).map(|i| self.unpack(&packed, i)).collect()
    }

    /// Bytes one element takes, averaged over a block.
    fn element_bytes(&self) -> f64 {
        self.dtype().bytes() as f64
    }

    /// The byte at which element `index` starts.
    fn offset(&self, index: usize) -> usize {
        index * self.dtype().bytes()
    }

    /// The element starting at byte `offset`.
    fn elements(&self, offset: usize) -> usize {
        offset / self.dtype().bytes()
    }

    /// Bytes `elements` take.
    fn bytes(&self, elements: usize) -> usize {
        elements * self.dtype().bytes()
    }

    /// `x` against the elements from `at`, one at a time.
    fn dot(&self, x: &[f32], bytes: &[u8], at: usize) -> f32 {
        let mut sum = 0.0;
        for (i, xi) in x.iter().enumerate() {
            sum += xi * self.unpack(bytes, at + i);
        }
        sum
    }

    /// Whether a float form needs widening before an f32 read.
    fn narrow(&self) -> bool {
        self.dtype().is_float() && self.dtype() != DType::F32
    }
}

/// Debug, equality, the registry `$all` and `named`, for `$form`.
#[macro_export]
macro_rules! registry {
    ($form:ident, $all:ident) => {
        impl std::fmt::Debug for dyn $form {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.name())
            }
        }

        impl PartialEq for dyn $form {
            fn eq(&self, other: &Self) -> bool {
                self.name() == other.name()
            }
        }

        impl Eq for dyn $form {}

        $crate::form::gather! {
            #[gather]
            pub static $all: $crate::form::ScatteredSlice<&'static dyn $form>;
        }

        /// The registered form called `name`.
        pub fn named(name: &str) -> Option<&'static dyn $form> {
            $all.iter().copied().find(|f| f.name() == name)
        }
    };
}

/// Adds `$form` to the registry `$all` of `$kind` forms.
#[macro_export]
macro_rules! register {
    ([$($all:tt)*] [$($kind:tt)*] $form:expr) => {
        const _: () = {
            $crate::form::scatter! {
                #[scatter($($all)*)]
                static FORM: &'static dyn $($kind)* = &$form;
            }
        };
    };
}

crate::registry!(Form, FORMS);

/// Registers a form, so files name it.
macro_rules! form {
    ($form:expr) => {
        $crate::register!([$crate::form::FORMS] [$crate::form::Form] $form);
    };
}

pub(crate) use form;

/// Plain f32.
pub struct F32;
/// bf16, widened to f32 on read.
pub struct Bf16;
/// Unsigned 32-bit ids.
pub struct U32;
/// Raw bytes, for a decode helper to walk.
pub struct U8;

/// The methods naming a form and its element.
macro_rules! element {
    ($name:literal, $dtype:ident, $bits:literal) => {
        fn name(&self) -> &'static str {
            $name
        }
        fn dtype(&self) -> DType {
            DType::$dtype
        }
        fn read(&self) -> Read {
            Read::$dtype
        }
        fn bits(&self) -> u32 {
            $bits
        }
    };
}

fn word(bytes: &[u8], index: usize) -> u32 {
    let at = index * 4;
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

impl Form for F32 {
    element!("f32", F32, 24);
    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        f32::from_bits(word(bytes, index))
    }
    fn pack(&self, data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}

impl Form for Bf16 {
    element!("bf16", Bf16, 8);
    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        let at = index * 2;
        Half(u16::from_le_bytes([bytes[at], bytes[at + 1]])).into()
    }
    fn pack(&self, data: &[f32]) -> Vec<u8> {
        data.iter()
            .flat_map(|x| Half::from(*x).0.to_le_bytes())
            .collect()
    }
}

impl Form for U32 {
    element!("u32", U32, 32);
    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        word(bytes, index) as f32
    }
    fn pack(&self, data: &[f32]) -> Vec<u8> {
        data.iter()
            .flat_map(|x| (*x as u32).to_le_bytes())
            .collect()
    }
}

impl Form for U8 {
    element!("u8", U8, 8);
    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        f32::from(bytes[index])
    }
    fn pack(&self, data: &[f32]) -> Vec<u8> {
        data.iter().map(|x| *x as u8).collect()
    }
}

form!(F32);
form!(Bf16);
form!(U32);
