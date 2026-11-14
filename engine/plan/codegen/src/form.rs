//! Storage forms: how buffers hold elements and reads decode them.

use crate::ir::Helper;
use graph::{Bf16 as Half, DType};
use std::fmt;
use std::sync::Mutex;

pub use scattered_collect::declarative::{gather, scatter};

/// Decodes its elements in kernel text and on the host.
pub trait Form: Sync {
    fn name(&self) -> &'static str;
    /// The element type, for moving and converting bytes.
    fn dtype(&self) -> DType;
    /// Significant bits an element keeps.
    fn bits(&self) -> u32;
    /// A generated kernel's type for one element.
    fn ty(&self) -> &'static str;
    /// Kernel text: element `index` of `name` as a float.
    fn load(&self, name: &str, index: &str) -> String;
    /// Element `index` of `bytes`, as a float.
    fn unpack(&self, bytes: &[u8], index: usize) -> f32;
    /// `data` as the bytes of this form.
    fn pack(&self, data: &[f32]) -> Vec<u8>;

    /// Whether floats can become this form. Codes cannot.
    fn packs(&self) -> bool {
        true
    }

    /// Whether a load decodes at length, so loops stay rolled.
    fn heavy(&self) -> bool {
        false
    }

    /// Distance between a lane's two wide loads, in elements.
    fn pairs(&self) -> usize {
        0
    }

    /// Elements that share one scale, decoded together.
    fn block(&self) -> usize {
        1
    }

    /// Elements a row's length must be a multiple of.
    fn align(&self) -> usize {
        self.block()
    }

    /// `data` as this form keeps it.
    fn roundtrip(&self, data: &[f32]) -> Vec<f32> {
        let packed = self.pack(data);
        (0..data.len()).map(|i| self.unpack(&packed, i)).collect()
    }

    /// The helpers `load` calls, each defined once.
    fn header(&self) -> Vec<Helper> {
        Vec::new()
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

    /// Kernel text: elements `index` to `index + 3` as a `wide()`.
    fn load4(&self, name: &str, index: &str) -> String {
        let each = |i: u32| self.load(name, &format!("({index} + {i})"));
        format!(
            "f_make4({}, {}, {}, {})",
            each(0),
            each(1),
            each(2),
            each(3)
        )
    }

    /// The kernel type of a `load4` result.
    fn wide(&self) -> &'static str {
        "float4"
    }

    /// Kernel text: part `h` of a `wide()`, as a float.
    fn part(&self, wide: &str, h: char) -> String {
        format!("{wide}.{h}")
    }

    /// Bytes `elements` take.
    fn bytes(&self, elements: usize) -> usize {
        elements * self.dtype().bytes()
    }

    /// Whether a float form needs widening before an f32 read.
    fn narrow(&self) -> bool {
        self.dtype().is_float() && self.dtype() != DType::F32
    }

    /// Whether `input` reads as `form`, when `weight` is this form.
    fn taken(&self, input: usize, weight: usize, form: &dyn Form) -> bool {
        !form.narrow() || (input == weight && form.name() == self.name())
    }

    /// Bytes a matmul reads: the weight, then the rows.
    fn matmul_bytes(&self, m: usize, n: usize, k: usize) -> Vec<f64> {
        vec![self.bytes(n * k) as f64, (4 * m * k) as f64]
    }

    /// `base` for f32, else `base` with the name appended.
    fn suffixed(&self, base: &'static str) -> &'static str {
        match self.name() {
            "f32" => base,
            name => interned(format!("{base}_{name}")),
        }
    }
}

impl fmt::Debug for dyn Form {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl PartialEq for dyn Form {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for dyn Form {}

/// Kernel names that outlive the forms that made them.
static NAMES: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

fn interned(name: String) -> &'static str {
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    match names.iter().find(|n| ***n == *name) {
        Some(n) => n,
        None => {
            let name: &'static str = Box::leak(name.into_boxed_str());
            names.push(name);
            name
        }
    }
}

gather! {
    #[gather]
    pub static FORMS: scattered_collect::ScatteredSlice<&'static dyn Form>;
}

/// Registers a form. Pins and recipes then name it.
#[macro_export]
macro_rules! form {
    ($form:expr) => {
        const _: () = {
            $crate::form::scatter! {
                #[scatter($crate::form::FORMS)]
                static FORM: &'static dyn $crate::form::Form = &$form;
            }
        };
    };
}

/// The registered form called `name`.
pub fn named(name: &str) -> Option<&'static dyn Form> {
    FORMS.iter().copied().find(|f| f.name() == name)
}

/// The form that stores `dtype` as is.
pub fn of(dtype: DType) -> &'static dyn Form {
    let name = format!("{dtype:?}").to_lowercase();
    named(&name).unwrap_or_else(|| panic!("no form stores {dtype:?}"))
}

/// Plain f32.
pub struct F32;
/// bf16, widened to f32 on read.
pub struct Bf16;
/// Unsigned 32-bit ids.
pub struct U32;
/// Raw bytes, as a decode helper walks them.
pub struct U8;

/// The methods naming a form and its element.
macro_rules! element {
    ($name:literal, $dtype:ident, $bits:literal, $ty:literal) => {
        fn name(&self) -> &'static str {
            $name
        }
        fn dtype(&self) -> DType {
            DType::$dtype
        }
        fn bits(&self) -> u32 {
            $bits
        }
        fn ty(&self) -> &'static str {
            $ty
        }
    };
}

fn word(bytes: &[u8], index: usize) -> u32 {
    let at = index * 4;
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

impl Form for F32 {
    element!("f32", F32, 24, "float");
    fn load(&self, name: &str, index: &str) -> String {
        format!("{name}[{index}]")
    }
    fn load4(&self, name: &str, index: &str) -> String {
        format!("f_f32x4({name} + {index})")
    }
    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        f32::from_bits(word(bytes, index))
    }
    fn pack(&self, data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}

impl Form for Bf16 {
    element!("bf16", Bf16, 8, "u16");
    fn load(&self, name: &str, index: &str) -> String {
        format!("f_bits((u32){name}[{index}] << 16)")
    }
    /// Four bf16 as two words, decoded as they are used.
    fn load4(&self, name: &str, index: &str) -> String {
        format!("f_u32x2({name} + {index})")
    }
    fn wide(&self) -> &'static str {
        "uint2"
    }
    /// Two bf16 share each word: low half first.
    fn part(&self, wide: &str, h: char) -> String {
        match h {
            'x' => format!("f_bits({wide}.x << 16)"),
            'y' => format!("f_bits({wide}.x & 0xffff0000u)"),
            'z' => format!("f_bits({wide}.y << 16)"),
            _ => format!("f_bits({wide}.y & 0xffff0000u)"),
        }
    }
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
    element!("u32", U32, 32, "u32");
    fn load(&self, name: &str, index: &str) -> String {
        format!("{name}[{index}]")
    }
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
    element!("u8", U8, 8, "u8");
    fn load(&self, name: &str, index: &str) -> String {
        format!("{name}[{index}]")
    }
    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        f32::from(bytes[index])
    }
    fn pack(&self, data: &[f32]) -> Vec<u8> {
        data.iter().map(|x| *x as u8).collect()
    }
}

crate::form!(F32);
crate::form!(Bf16);
crate::form!(U32);

/// The builtin forms, by name.
pub const F32_FORM: &dyn Form = &F32;
pub const BF16_FORM: &dyn Form = &Bf16;
pub const U32_FORM: &dyn Form = &U32;
pub const U8_FORM: &dyn Form = &U8;
