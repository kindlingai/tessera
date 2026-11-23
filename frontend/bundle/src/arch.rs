//! Model architectures, one record each, registered in their own files.

use crate::names::Rule;
use crate::parts::{Cut, Part};
use crate::{Overrides, Weights};
use std::error::Error;
use std::path::Path;
use tessera_engine::decoder::Config;
use tessera_engine::{Graph, Pins, Steps};

pub(crate) use scattered_collect::declarative::{gather, scatter};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

/// How a `config.json` names an architecture.
pub(crate) enum Spot {
    /// Its `model_type`.
    Type(&'static str),
    /// Its `architectures[0]` or `_class_name`.
    Class(&'static str),
    /// A test on the config text.
    Probe(fn(&str) -> bool),
}

/// The inputs of a step graph load.
pub(crate) struct Ask<'a> {
    pub dir: &'a Path,
    pub sizes: [u64; 2],
    pub user: &'a [Rule],
    pub steps: Steps,
    pub overrides: &'a Overrides,
    pub manifest: Option<&'a str>,
    /// 1-based layers whose rows the graph also gives.
    pub taps: &'a [u64],
    /// `config.json` with the overrides applied.
    pub json: &'a str,
}

/// A step graph with its weights and their forms.
pub(crate) type Built = (Graph, Weights, Pins);

/// Graphs over the shared decoder config.
pub(crate) struct Family {
    pub step: fn(&Config) -> Graph,
    pub taps: fn(&Config, &[u64]) -> Graph,
}

pub(crate) struct Arch {
    pub name: &'static str,
    pub spot: &'static [Spot],
    /// The weight name rules, as JSON.
    pub rules: Option<&'static str>,
    pub step: Option<fn(&Arch, &Ask) -> Result<Built>>,
    pub part: Option<fn(&Arch, &Cut) -> Result<Part>>,
    pub family: Option<Family>,
}

impl Arch {
    pub const NONE: Arch = Arch {
        name: "",
        spot: &[],
        rules: None,
        step: None,
        part: None,
        family: None,
    };

    /// The weight name rules, or none.
    pub fn rules(&self) -> Vec<Rule> {
        let json = self.rules.unwrap_or("{}");
        crate::names::rules(json).expect("the repository's rules parse")
    }

    fn spots(&self, v: &serde_json::Value, json: &str) -> bool {
        let class = [&v["_class_name"], &v["architectures"][0]];
        for spot in self.spot {
            let hit = match spot {
                Spot::Type(t) => v["model_type"].as_str() == Some(t),
                Spot::Class(c) => class.iter().any(|s| s.as_str() == Some(c)),
                Spot::Probe(probe) => probe(json),
            };
            if hit {
                return true;
            }
        }
        false
    }
}

gather! {
    #[gather]
    pub(crate) static ARCHES: scattered_collect::ScatteredSlice<Arch>;
}

/// Registers an architecture: name, spots, then `field: value` pairs.
macro_rules! arch {
    ($name:literal, [$($spot:expr),* $(,)?] $(, $field:ident: $value:expr)* $(,)?) => {
        const _: () = {
            $crate::arch::scatter! {
                #[scatter($crate::arch::ARCHES)]
                static ARCH: $crate::arch::Arch = $crate::arch::Arch {
                    name: $name,
                    spot: &[$($spot),*],
                    $($field: Some($value),)*
                    ..$crate::arch::Arch::NONE
                };
            }
        };
    };
}
pub(crate) use arch;

/// The registered names, sorted, for error text.
pub(crate) fn known() -> String {
    let mut names: Vec<&str> = ARCHES.iter().map(|a| a.name).collect();
    names.sort_unstable();
    names.join(", ")
}

/// The error text for an unregistered `what`.
pub(crate) fn unknown(what: &str) -> String {
    format!(
        "{what} is not a model this engine traces. Known: {}",
        known()
    )
}

/// The architecture registered as `name`.
pub(crate) fn find(name: &str) -> Option<&'static Arch> {
    ARCHES.iter().find(|a| a.name == name)
}

/// The one architecture `json` describes.
pub(crate) fn spot(json: &str) -> Result<&'static Arch> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    let mut hits = Vec::new();
    for arch in ARCHES.iter() {
        if arch.spots(&v, json) {
            hits.push(arch);
        }
    }
    match hits.as_slice() {
        [one] => Ok(one),
        [] => Err(unknown(v["model_type"].as_str().unwrap_or("the config")).into()),
        many => {
            let names: Vec<&str> = many.iter().map(|a| a.name).collect();
            Err(format!("the config matches {}", names.join(" and ")).into())
        }
    }
}

/// The decoder family registered as `name`.
pub(crate) fn family(name: &str) -> Result<&'static Family> {
    let arch = find(name).ok_or_else(|| unknown(name))?;
    arch.family
        .as_ref()
        .ok_or_else(|| format!("{name} is not a decoder. Known: {}", known()).into())
}
