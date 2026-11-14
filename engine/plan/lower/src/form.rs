//! Pins: which registered form each weight takes.

use graph::PathSeg;
use mem::Tensor;
use serde::{Deserialize, Serialize};

pub use codegen::form::{BF16_FORM, Bf16, F32, F32_FORM, FORMS, Form, U32, U32_FORM, named, of};

/// A weight as a file holds it.
pub enum Weight {
    Floats(Tensor),
    /// One form's bytes, as `Form::pack` lays them out.
    Packed(&'static dyn Form, Vec<u8>),
}

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
