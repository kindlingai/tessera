//! Storage forms in kernel text: types, loads and decode helpers.

use crate::ir::Helper;
use std::sync::Mutex;

pub use tessera_mem::form::{Bf16, F16, F32, Form, Fp8, Read, U8, U32};
pub(crate) use tessera_mem::form::{gather, scatter};

/// Decodes its elements in kernel text.
pub trait Load: Form {
    /// The plain form of a rotated weight's codes, if rotated.
    fn codes(&self) -> Option<&'static dyn Load> {
        None
    }
    /// A generated kernel's type for one element.
    fn ty(&self) -> &'static str;
    /// Kernel text: element `index` of `name` as a float.
    fn load(&self, name: &str, index: &str) -> String;

    /// Whether a load decodes at length, so loops stay rolled.
    fn heavy(&self) -> bool {
        false
    }

    /// Distance between a lane's two wide loads, in elements.
    fn pairs(&self) -> usize {
        0
    }

    /// Its decode helper for `load`, or `load4` if `wide`.
    fn decoder(&self, _wide: bool) -> Option<Helper> {
        None
    }

    /// The helpers `load` calls, each defined once.
    fn header(&self) -> Vec<Helper> {
        Vec::new()
    }

    /// Kernel text: elements `index` to `index + 3` as a `wide()`.
    fn load4(&self, name: &str, index: &str) -> String {
        four("f_make4", |i| self.load(name, &format!("({index} + {i})")))
    }

    /// The kernel type of a `load4` result.
    fn wide(&self) -> &'static str {
        "float4"
    }

    /// Kernel text: part `h` of a `wide()`, as a float.
    fn part(&self, wide: &str, h: char) -> String {
        format!("{wide}.{h}")
    }

    /// Whether kernels can store directly into this form.
    fn stores(&self) -> bool {
        false
    }

    /// Kernel text: stores `value` at `index` of `name`.
    fn store(&self, name: &str, index: &str, value: &str) -> String {
        format!("{name}[{index}] = {value}")
    }

    /// Kernel text: stores four floats `value` from `index`.
    fn store4(&self, name: &str, index: &str, value: &str) -> String {
        format!("f_f32x4({name} + {index}) = {value}")
    }

    /// Its WGSL element type. None: WGSL cannot read it.
    fn wgsl_ty(&self) -> Option<&'static str> {
        None
    }

    /// WGSL text: element `index` of `name` as a float.
    fn wgsl_load(&self, name: &str, index: &str) -> String {
        self.load(name, index)
    }

    /// WGSL text: elements `index` to `index + 3` as a `wgsl_wide()`.
    fn wgsl_load4(&self, name: &str, index: &str) -> String {
        four("vec4<f32>", |i| {
            self.wgsl_load(name, &format!("({index} + {i}u)"))
        })
    }

    fn wgsl_wide(&self) -> &'static str {
        "vec4<f32>"
    }

    /// WGSL text: part `h` of a `wgsl_wide()`, as a float.
    fn wgsl_part(&self, wide: &str, h: char) -> String {
        self.part(wide, h)
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

/// `make(a, b, c, d)`, with `each(i)` giving element `i`.
fn four(make: &str, each: impl Fn(u32) -> String) -> String {
    format!("{make}({}, {}, {}, {})", each(0), each(1), each(2), each(3))
}

/// Interned kernel names. They outlive their forms.
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

tessera_mem::registry!(Load, LOADS);

/// Registers a form's loads for pins and recipes to name.
macro_rules! form {
    ($form:expr) => {
        tessera_mem::register!([$crate::form::LOADS] [$crate::form::Load] $form);
    };
}

/// Each shared form function as `name::body`.
macro_rules! kernel_fns {
    ($($(#[$doc:meta])* $($kernel:literal)?
        fn $name:ident $(<const $c:ident: $ct:ty>)?
            ($($(#[$build:meta])* $arg:ident: $ty:ident $(<$g:ident>)?),*) -> $ret:tt $body:block)*) => {$(
        $(#[$doc])*
        pub mod $name {
            use super::*;
            $crate::form::kernel_attr! { [$($kernel)?]
                pub fn body $(<const $c: $ct>)? ($($(#[$build])* $arg: $ty $(<$g>)?),*) -> $ret $body
            }
        }
    )*};
}

/// Named bodies become helpers. Others inline at each call.
macro_rules! kernel_attr {
    ([] $($f:tt)*) => { #[kernel] $($f)* };
    ([$name:literal] $($f:tt)*) => { #[helper($name)] $($f)* };
}

pub(crate) use {form, kernel_attr, kernel_fns};

/// The form that stores `dtype` as is.
pub fn of(dtype: tessera_graph::DType) -> &'static dyn Load {
    let name = format!("{dtype:?}").to_lowercase();
    named(&name).unwrap_or_else(|| panic!("unknown storage form for {dtype:?}"))
}

impl Load for F32 {
    fn ty(&self) -> &'static str {
        "float"
    }
    fn load(&self, name: &str, index: &str) -> String {
        format!("{name}[{index}]")
    }
    fn load4(&self, name: &str, index: &str) -> String {
        format!("f_f32x4({name} + {index})")
    }
    fn stores(&self) -> bool {
        true
    }
    fn wgsl_ty(&self) -> Option<&'static str> {
        Some("f32")
    }
}

/// Half-width floats: two to a word, low half first.
macro_rules! halves {
    ($($form:ident = $text:ident, $wgsl:ident { $($more:tt)* })*) => {$(
        impl Load for $form {
            fn ty(&self) -> &'static str {
                "u16"
            }
            fn load(&self, name: &str, index: &str) -> String {
                $text(&format!("(u32){name}[{index}]"), 0)
            }
            /// Four halves as two words, decoded as they are used.
            fn load4(&self, name: &str, index: &str) -> String {
                format!("f_u32x2({name} + {index})")
            }
            fn wide(&self) -> &'static str {
                "uint2"
            }
            fn part(&self, wide: &str, h: char) -> String {
                let (word, low) = half_at(h);
                $text(&format!("{wide}.{word}"), low)
            }
            fn wgsl_ty(&self) -> Option<&'static str> {
                Some("u32")
            }
            fn wgsl_load(&self, name: &str, index: &str) -> String {
                let word = format!("{name}[({index}) >> 1u]");
                $wgsl(&format!("({word} >> ((({index}) & 1u) << 4u))"), 0)
            }
            /// Four elements start a word, so two words hold them.
            fn wgsl_load4(&self, name: &str, index: &str) -> String {
                format!("vec2<u32>({name}[({index}) >> 1u], {name}[(({index}) >> 1u) + 1u])")
            }
            fn wgsl_wide(&self) -> &'static str {
                "vec2<u32>"
            }
            fn wgsl_part(&self, wide: &str, h: char) -> String {
                let (word, low) = half_at(h);
                $wgsl(&format!("{wide}.{word}"), low)
            }
            $($more)*
        }
    )*};
}

halves! {
    Bf16 = bf16_text, bf16_wgsl {
        fn stores(&self) -> bool {
            true
        }
        fn store(&self, name: &str, index: &str, value: &str) -> String {
            format!("{name}[{index}] = f_to_bf16({value})")
        }
        fn store4(&self, name: &str, index: &str, value: &str) -> String {
            format!("f_store4_bf16({name} + {index}, {value})")
        }
    }
    F16 = f16_text, f16_wgsl {}
}

/// Part `h`'s word and low bit, for four halves.
fn half_at(h: char) -> (char, u32) {
    match h {
        'x' => ('x', 0),
        'y' => ('x', 16),
        'z' => ('y', 0),
        _ => ('y', 16),
    }
}

/// The bf16 at bit `low` of word `w`, through `bits`.
fn bf16_bits(bits: &str, w: &str, low: u32) -> String {
    match low {
        0 => format!("{bits}({w} << 16)"),
        _ => format!("{bits}({w} & 0xffff0000u)"),
    }
}

fn bf16_text(w: &str, low: u32) -> String {
    bf16_bits("f_bits", w, low)
}

fn bf16_wgsl(w: &str, low: u32) -> String {
    bf16_bits("bitcast<f32>", w, low)
}

fn f16_text(w: &str, low: u32) -> String {
    format!("f_f16({w}, {low})")
}

fn f16_wgsl(w: &str, low: u32) -> String {
    format!("unpack2x16float({w} >> {low}u).x")
}

impl Load for Fp8 {
    fn ty(&self) -> &'static str {
        "u8"
    }
    fn load(&self, name: &str, index: &str) -> String {
        format!("f_e4m3((u32){name}[{index}], 0)")
    }
    /// Four e4m3 bytes as one word, decoded on use.
    fn load4(&self, name: &str, index: &str) -> String {
        format!("f_u32({name} + {index})")
    }
    fn wide(&self) -> &'static str {
        "u32"
    }
    fn part(&self, wide: &str, h: char) -> String {
        let shift = match h {
            'x' => 0,
            'y' => 8,
            'z' => 16,
            _ => 24,
        };
        format!("f_e4m3({wide}, {shift})")
    }
    fn stores(&self) -> bool {
        true
    }
    fn store(&self, name: &str, index: &str, value: &str) -> String {
        format!("{name}[{index}] = f_to_e4m3({value})")
    }
    fn store4(&self, name: &str, index: &str, value: &str) -> String {
        format!("f_store4_e4m3({name} + {index}, {value})")
    }
}

/// Integer forms: a plain index, under the kernel type `$ty`.
macro_rules! plain {
    ($($form:ident = $ty:literal, $writes:literal);*) => {$(
        impl Load for $form {
            fn ty(&self) -> &'static str {
                $ty
            }
            fn load(&self, name: &str, index: &str) -> String {
                format!("{name}[{index}]")
            }
            fn stores(&self) -> bool {
                $writes
            }
            fn wgsl_ty(&self) -> Option<&'static str> {
                Some("u32")
            }
            fn wgsl_load(&self, name: &str, index: &str) -> String {
                wgsl_plain($ty, name, index)
            }
        }
    )*};
}

plain!(U32 = "u32", true; U8 = "u8", false);

/// WGSL stores words, so a byte is a field.
fn wgsl_plain(ty: &str, name: &str, index: &str) -> String {
    match ty {
        "u8" => format!("(({name}[({index}) >> 2u] >> ((({index}) & 3u) << 3u)) & 0xffu)"),
        _ => format!("{name}[{index}]"),
    }
}

form!(F32);
form!(Bf16);
form!(F16);
form!(Fp8);
form!(U32);

/// The builtin forms, by name.
pub const F32_FORM: &dyn Load = &F32;
pub const BF16_FORM: &dyn Load = &Bf16;
pub const FP8_FORM: &dyn Load = &Fp8;
pub const U32_FORM: &dyn Load = &U32;
pub(crate) const U8_FORM: &dyn Load = &U8;
