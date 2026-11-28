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
    /// Forms for some regions. The last match wins.
    pub spots: Vec<Spot>,
}

/// Forms for the nodes whose region label fits `at`.
#[derive(Clone, Debug, PartialEq)]
pub struct Spot {
    /// A pattern over labels such as `layer.3.attention`. `*` is any run.
    pub at: String,
    pub activations: Option<&'static dyn Load>,
    pub kept: Option<&'static dyn Load>,
    pub sites: BTreeMap<Site, &'static dyn Load>,
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
            spots: Vec::new(),
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

    /// These pins with the spots that fit `label` applied.
    pub fn at(&self, label: &str) -> Pins {
        let mut out = Pins {
            sites: self.sites.clone(),
            named: self.named.clone(),
            spots: Vec::new(),
            ..*self
        };
        for s in self.spots.iter().filter(|s| in_scope(&s.at, label)) {
            out.activations = s.activations.unwrap_or(out.activations);
            out.kept = s.kept.unwrap_or(out.kept);
            out.sites.extend(&s.sites);
        }
        out
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    spots: Vec<SpotNames>,
}

/// A spot's forms as names.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpotNames {
    at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    activations: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    kept: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    sites: BTreeMap<Site, String>,
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
            spots: self
                .spots
                .iter()
                .map(|s| SpotNames {
                    at: s.at.clone(),
                    activations: s.activations.map(|f| f.name().into()),
                    kept: s.kept.map(|f| f.name().into()),
                    sites: s.sites.iter().map(|(k, f)| (*k, f.name().into())).collect(),
                })
                .collect(),
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
        for s in text.spots {
            let mut sites = BTreeMap::new();
            for (site, name) in s.sites {
                sites.insert(site, store(&name)?);
            }
            pins.spots.push(Spot {
                at: s.at,
                activations: s.activations.as_deref().map(store).transpose()?,
                kept: s.kept.as_deref().map(store).transpose()?,
                sites,
            });
        }
        Ok(pins)
    }
}

/// Whether `label` fits `at`. Layer indices may be sets: `1-26`.
pub(crate) fn in_scope(at: &str, label: &str) -> bool {
    // The index after `layer.` and the rest, if any.
    fn parts(text: &str) -> Option<(&str, Option<&str>)> {
        let rest = text.strip_prefix("layer.")?;
        Some(
            rest.split_once('.')
                .map_or((rest, None), |(k, tail)| (k, Some(tail))),
        )
    }
    let (Some((set, at_tail)), Some((k, tail))) = (parts(at), parts(label)) else {
        return matches(at, label);
    };
    let plain = |c: char| c.is_ascii_digit() || c == ',' || c == '-';
    let Ok(k) = k.parse::<u32>() else {
        return false;
    };
    if set.is_empty() || !set.chars().all(plain) {
        return matches(at, label);
    }
    let holds = |part: &str| match part.split_once('-') {
        Some((a, b)) => {
            a.parse().is_ok_and(|a: u32| a <= k) && b.parse().is_ok_and(|b: u32| k <= b)
        }
        None => part.parse() == Ok(k),
    };
    let tails = match (at_tail, tail) {
        (None, _) => true,
        (Some(a), Some(t)) => matches(a, t),
        _ => false,
    };
    set.split(',').any(holds) && tails
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

/// Metal includes and the shared e4m3 decode table.
pub const METAL_PRELUDE: &str = r#"#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant float FP8_LUT[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, NAN,
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, -NAN,
};
"#;

/// The Metal dtype code of `form`, with fp8 if `fp8`.
pub fn metal_code(form: &dyn Load, fp8: bool) -> Option<usize> {
    match form.name() {
        "f32" => Some(0),
        "bf16" => Some(1),
        "f16" => Some(2),
        "fp8" if fp8 => Some(3),
        _ => None,
    }
}

/// The shared Metal prelude, then `body`.
#[macro_export]
macro_rules! metal_source {
    ($body:expr) => {{
        static TEXT: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
        *TEXT.get_or_init(|| {
            Box::leak(format!("{}{}", $crate::form::METAL_PRELUDE, $body).into_boxed_str())
        })
    }};
}
