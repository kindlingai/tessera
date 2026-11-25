//! Pins: registered storage forms for a lowered graph.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use tessera_engine_settings::Reads;
use tessera_graph::{Graph, PathSeg};
use tessera_mem::Tensor;

pub use tessera_codegen::form::{BF16_FORM, F32_FORM, FP8_FORM, LOADS, Load, U32_FORM, named};

/// A weight in a file's layout.
pub enum Weight {
    Floats(Tensor),
    /// One form's bytes, in `Form::pack` layout.
    Packed(&'static dyn Load, Vec<u8>),
    /// One form's bytes, read from their file when loaded.
    Filed(&'static dyn Load, Arc<dyn Filed>),
}

/// A weight's bytes where a file keeps them.
pub trait Filed: Send + Sync {
    fn bytes(&self) -> usize;
    /// Fills `into`, which is `bytes()` long.
    fn read(&self, into: &mut [u8], reads: Reads) -> std::io::Result<()>;
}

impl Weight {
    /// This weight with a filed one's bytes read in.
    pub fn held(&self, reads: Reads) -> std::io::Result<Option<Weight>> {
        let Weight::Filed(form, file) = self else {
            return Ok(None);
        };
        let mut bytes = vec![0u8; file.bytes()];
        file.read(&mut bytes, reads)?;
        Ok(Some(Weight::Packed(*form, bytes)))
    }
}

/// A site a pin override can target besides weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Site {
    /// RMS and layer normalization outputs.
    Norm,
    /// Softmax and unit-sum outputs.
    Softmax,
    /// Entry output logits.
    Logits,
    /// MoE router and gating ops.
    Router,
    /// Residual stream adds.
    Residual,
    /// Attention outputs.
    Attention,
    /// Matrix multiply outputs.
    Matmul,
}

/// The form of weights, activations, kept rows and outputs.
#[derive(Clone, Debug, PartialEq)]
pub struct Pins {
    /// The form of every weight no pattern names.
    pub weights: &'static dyn Load,
    /// The form of float activations no override names.
    pub activations: &'static dyn Load,
    /// The form of kept rows (KV cache).
    pub kept: &'static dyn Load,
    /// The form of float entry outputs.
    pub outputs: &'static dyn Load,
    /// Overrides for accuracy-sensitive sites.
    pub sites: BTreeMap<Site, &'static dyn Load>,
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
            activations: F32_FORM,
            kept: F32_FORM,
            outputs: F32_FORM,
            sites: BTreeMap::new(),
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

    /// Pins default float activations to `form`.
    pub fn activations(mut self, form: &'static dyn Load) -> Self {
        assert!(
            form.stores(),
            "activation form `{}` is not writable",
            form.name()
        );
        self.activations = form;
        self
    }

    /// Pins kept rows (KV cache) to `form`.
    pub fn kept(mut self, form: &'static dyn Load) -> Self {
        assert!(form.stores(), "kept form `{}` is not writable", form.name());
        self.kept = form;
        self
    }

    /// Pins float entry outputs to `form`.
    pub fn outputs(mut self, form: &'static dyn Load) -> Self {
        assert!(
            form.stores(),
            "output form `{}` is not writable",
            form.name()
        );
        self.outputs = form;
        self
    }

    /// Pins one accuracy-sensitive site to `form`.
    pub fn site(mut self, site: Site, form: &'static dyn Load) -> Self {
        assert!(form.stores(), "site form `{}` is not writable", form.name());
        self.sites.insert(site, form);
        self
    }

    /// The form of the weight at `path`.
    pub fn form(&self, path: &[PathSeg]) -> &'static dyn Load {
        let name = PathSeg::dotted(path);
        let hit = self.named.iter().rev().find(|(p, _)| matches(p, &name));
        hit.map_or(self.weights, |(_, form)| *form)
    }

    /// The override form for `site`, if pinned.
    pub fn site_form(&self, site: Site) -> Option<&'static dyn Load> {
        self.sites.get(&site).copied()
    }
}

/// `form` if a plain readable float, else `F32_FORM`.
pub fn plain(form: &'static dyn Load) -> &'static dyn Load {
    match form.dtype().is_float() && form.block() == 1 {
        true => form,
        false => F32_FORM,
    }
}

/// `form` if a writable float form, else `F32_FORM`.
pub fn writable(form: &'static dyn Load) -> &'static dyn Load {
    match form.stores() && form.dtype().is_float() {
        true => form,
        false => F32_FORM,
    }
}

fn f32_name() -> String {
    "f32".into()
}

fn is_f32(name: &str) -> bool {
    name == "f32"
}

/// Pins as names, for recipes.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PinNames {
    #[serde(default = "f32_name")]
    weights: String,
    #[serde(default = "f32_name", skip_serializing_if = "is_f32")]
    activations: String,
    #[serde(default = "f32_name", skip_serializing_if = "is_f32")]
    kept: String,
    #[serde(default = "f32_name", skip_serializing_if = "is_f32")]
    outputs: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    sites: BTreeMap<Site, String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    named: Vec<(String, String)>,
}

impl Serialize for Pins {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let named = |(p, f): &(String, &dyn Load)| (p.clone(), f.name().to_string());
        let sites = self
            .sites
            .iter()
            .map(|(k, f)| (*k, f.name().to_string()))
            .collect();
        PinNames {
            weights: self.weights.name().into(),
            activations: self.activations.name().into(),
            kept: self.kept.name().into(),
            outputs: self.outputs.name().into(),
            sites,
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
        let store = |name: &str| {
            let f = form(name)?;
            match f.stores() && f.dtype().is_float() {
                true => Ok(f),
                false => Err(serde::de::Error::custom(format!(
                    "form {name} is not a writable float"
                ))),
            }
        };
        let mut pins = Pins::all(form(&text.weights)?)
            .activations(store(&text.activations)?)
            .kept(store(&text.kept)?)
            .outputs(store(&text.outputs)?);
        for (site, name) in text.sites {
            pins = pins.site(site, store(&name)?);
        }
        for (pattern, name) in text.named {
            pins = pins.pin(&pattern, form(&name)?);
        }
        Ok(pins)
    }
}

/// Whether `name` fits `pattern`. A star stands for anything.
pub(crate) fn matches(pattern: &str, name: &str) -> bool {
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
