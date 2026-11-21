//! Pins: the registered form for each weight.

use serde::{Deserialize, Serialize};
use tessera_graph::{Graph, PathSeg};
use tessera_mem::Tensor;

pub use tessera_codegen::form::{BF16_FORM, F32_FORM, LOADS, Load, U32_FORM, named};

/// A weight in a file's layout.
pub enum Weight {
    Floats(Tensor),
    /// One form's bytes, in `Form::pack` layout.
    Packed(&'static dyn Load, Vec<u8>),
}

/// The form of each weight, as the user says.
#[derive(Clone, Debug, PartialEq)]
pub struct Pins {
    /// The form of every weight no pattern names.
    pub weights: &'static dyn Load,
    /// Patterns over weight paths, the last match winning.
    pub named: Vec<(String, &'static dyn Load)>,
}

impl Default for Pins {
    fn default() -> Self {
        Self::all(F32_FORM)
    }
}

impl Pins {
    pub fn all(weights: &'static dyn Load) -> Self {
        Self {
            weights,
            named: Vec::new(),
        }
    }

    /// Pins weights whose path fits `pattern`. `*` is any run.
    pub fn pin(mut self, pattern: &str, form: &'static dyn Load) -> Self {
        self.named.push((pattern.into(), form));
        self
    }

    /// Whether a weight of `g` shares a part as bytes.
    pub fn shares(&self, g: &Graph) -> impl Fn(u32, &[usize], usize, usize) -> bool {
        move |w, shape, axis, count| {
            let form = self.form(&g.weights[w as usize].path);
            form.shares(shape, axis, count)
        }
    }

    /// The form of the weight at `path`.
    pub fn form(&self, path: &[PathSeg]) -> &'static dyn Load {
        let name = PathSeg::dotted(path);
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
        let named = |(p, f): &(String, &dyn Load)| (p.clone(), f.name().to_string());
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
