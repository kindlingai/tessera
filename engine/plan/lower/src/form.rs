//! Storage forms by name, and the pins that choose them.

use graph::{DType, PathSeg};
use serde::{Deserialize, Serialize};
use std::fmt;

pub use scattered_collect::declarative::{gather, scatter};

/// How a buffer stores elements and a kernel reads one.
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
                static FORM: &'static dyn $crate::Form = &$form;
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

impl Form for F32 {
    fn name(&self) -> &'static str {
        "f32"
    }
    fn dtype(&self) -> DType {
        DType::F32
    }
    fn bits(&self) -> u32 {
        24
    }
    fn ty(&self) -> &'static str {
        "float"
    }
    fn load(&self, name: &str, index: &str) -> String {
        format!("{name}[{index}]")
    }
}

impl Form for Bf16 {
    fn name(&self) -> &'static str {
        "bf16"
    }
    fn dtype(&self) -> DType {
        DType::Bf16
    }
    fn bits(&self) -> u32 {
        8
    }
    fn ty(&self) -> &'static str {
        "u16"
    }
    fn load(&self, name: &str, index: &str) -> String {
        format!("f_bits((u32){name}[{index}] << 16)")
    }
}

impl Form for U32 {
    fn name(&self) -> &'static str {
        "u32"
    }
    fn dtype(&self) -> DType {
        DType::U32
    }
    fn bits(&self) -> u32 {
        32
    }
    fn ty(&self) -> &'static str {
        "u32"
    }
    fn load(&self, name: &str, index: &str) -> String {
        format!("{name}[{index}]")
    }
}

crate::form!(F32);
crate::form!(Bf16);
crate::form!(U32);

/// The builtin forms, by name.
pub const F32_FORM: &dyn Form = &F32;
pub const BF16_FORM: &dyn Form = &Bf16;
pub const U32_FORM: &dyn Form = &U32;

/// The form of each weight, as the user says.
#[derive(Clone, Debug, PartialEq)]
pub struct Pins {
    /// The form of every weight no pattern names.
    pub weights: &'static dyn Form,
    /// Patterns over weight paths, the last match winning.
    pub named: Vec<(String, &'static dyn Form)>,
}

impl Default for Pins {
    fn default() -> Self {
        Self::all(F32_FORM)
    }
}

impl Pins {
    pub fn all(weights: &'static dyn Form) -> Self {
        Self {
            weights,
            named: Vec::new(),
        }
    }

    /// Pins weights whose path fits `pattern`. `*` is any run.
    pub fn pin(mut self, pattern: &str, form: &'static dyn Form) -> Self {
        self.named.push((pattern.into(), form));
        self
    }

    /// The form of the weight at `path`.
    pub fn form(&self, path: &[PathSeg]) -> &'static dyn Form {
        let seg = |s: &PathSeg| match s {
            PathSeg::Field(f) => f.clone(),
            PathSeg::Index(i) => i.to_string(),
        };
        let name = path.iter().map(seg).collect::<Vec<_>>().join(".");
        let hit = self.named.iter().rev().find(|(p, _)| matches(p, &name));
        hit.map_or(self.weights, |(_, form)| *form)
    }
}

/// Pins as names, for recipes.
#[derive(Serialize, Deserialize)]
struct PinNames {
    weights: String,
    #[serde(default)]
    named: Vec<(String, String)>,
}

impl Serialize for Pins {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let named = |(p, f): &(String, &dyn Form)| (p.clone(), f.name().to_string());
        PinNames {
            weights: self.weights.name().into(),
            named: self.named.iter().map(named).collect(),
        }
        .serialize(s)
    }
}

impl<'de> Deserialize<'de> for Pins {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = PinNames::deserialize(d)?;
        let form = |name: &str| {
            named(name).ok_or_else(|| serde::de::Error::custom(format!("no form called {name}")))
        };
        let mut pins = Pins::all(form(&text.weights)?);
        for (pattern, name) in text.named {
            pins = pins.pin(&pattern, form(&name)?);
        }
        Ok(pins)
    }
}

/// Whether `name` fits `pattern`. A star stands for anything.
fn matches(pattern: &str, name: &str) -> bool {
    let mut parts = pattern.split('*');
    let first = parts.next().unwrap_or("");
    let Some(mut rest) = name.strip_prefix(first) else {
        return false;
    };
    let mut parts = parts.peekable();
    while let Some(part) = parts.next() {
        let last = parts.peek().is_none();
        rest = match last {
            true => return rest.ends_with(part),
            false => match rest.find(part) {
                Some(at) => &rest[at + part.len()..],
                None => return false,
            },
        };
    }
    rest.is_empty()
}
