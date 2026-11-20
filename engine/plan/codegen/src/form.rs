//! Storage forms in kernel text: types, loads and decode helpers.

use crate::ir::Helper;
use std::sync::Mutex;

pub use sensai_mem::form::{Bf16, F32, Form, Read, U8, U32};
pub(crate) use sensai_mem::form::{gather, scatter};

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

sensai_mem::registry!(Load, LOADS);

/// Registers a form's loads. Pins and recipes then name it.
macro_rules! form {
    ($form:expr) => {
        sensai_mem::register!([$crate::form::LOADS] [$crate::form::Load] $form);
    };
}

pub(crate) use form;

/// The form that stores `dtype` as is.
pub fn of(dtype: sensai_graph::DType) -> &'static dyn Load {
    let name = format!("{dtype:?}").to_lowercase();
    named(&name).unwrap_or_else(|| panic!("no form stores {dtype:?}"))
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
    fn wgsl_ty(&self) -> Option<&'static str> {
        Some("f32")
    }
}

impl Load for Bf16 {
    fn ty(&self) -> &'static str {
        "u16"
    }
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
        bf16_part("f_bits", wide, h)
    }
    fn wgsl_ty(&self) -> Option<&'static str> {
        Some("u32")
    }
    fn wgsl_load(&self, name: &str, index: &str) -> String {
        let word = format!("{name}[({index}) >> 1u]");
        format!("bitcast<f32>(({word} >> ((({index}) & 1u) << 4u)) << 16u)")
    }
    /// Four elements start a word, so two words hold them.
    fn wgsl_load4(&self, name: &str, index: &str) -> String {
        format!("vec2<u32>({name}[({index}) >> 1u], {name}[(({index}) >> 1u) + 1u])")
    }
    fn wgsl_wide(&self) -> &'static str {
        "vec2<u32>"
    }
    fn wgsl_part(&self, wide: &str, h: char) -> String {
        bf16_part("bitcast<f32>", wide, h)
    }
}

/// Part `h` of four bf16 in two words.
fn bf16_part(bits: &str, wide: &str, h: char) -> String {
    match h {
        'x' => format!("{bits}({wide}.x << 16)"),
        'y' => format!("{bits}({wide}.x & 0xffff0000u)"),
        'z' => format!("{bits}({wide}.y << 16)"),
        _ => format!("{bits}({wide}.y & 0xffff0000u)"),
    }
}

/// Integer forms: a plain index, under the kernel type `$ty`.
macro_rules! plain {
    ($($form:ident = $ty:literal),*) => {$(
        impl Load for $form {
            fn ty(&self) -> &'static str {
                $ty
            }
            fn load(&self, name: &str, index: &str) -> String {
                format!("{name}[{index}]")
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

plain!(U32 = "u32", U8 = "u8");

/// WGSL stores words, so a byte is a field.
fn wgsl_plain(ty: &str, name: &str, index: &str) -> String {
    match ty {
        "u8" => format!("(({name}[({index}) >> 2u] >> ((({index}) & 3u) << 3u)) & 0xffu)"),
        _ => format!("{name}[{index}]"),
    }
}

form!(F32);
form!(Bf16);
form!(U32);

/// The builtin forms, by name.
pub const F32_FORM: &dyn Load = &F32;
pub const BF16_FORM: &dyn Load = &Bf16;
pub const U32_FORM: &dyn Load = &U32;
pub(crate) const U8_FORM: &dyn Load = &U8;
