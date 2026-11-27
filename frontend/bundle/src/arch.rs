//! Model architectures, one record each, registered in their own files.

use crate::encoder::Embeds;
use crate::names::Rule;
use crate::parts::{Cut, Part};
use crate::{Overrides, Stream, Weights, numbered, pins};
use std::error::Error;
use std::path::Path;
use tessera_checkpoint::Offer;
use tessera_engine::Weight;
use tessera_engine::hf::MODEL_TYPE;
use tessera_engine::{Graph, Pins, PleConfig, Steps};

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

/// The inputs of a graph load.
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

impl<'a> Ask<'a> {
    /// A graph load with no rules, steps, taps or manifest.
    pub fn plain(dir: &'a Path, sizes: [u64; 2], overrides: &'a Overrides, json: &'a str) -> Self {
        Ask {
            dir,
            sizes,
            user: &[],
            steps: Steps::Plain,
            overrides,
            manifest: None,
            taps: &[],
            json,
        }
    }
}

/// A graph with its weights and their forms.
pub(crate) type Built = (Graph, Weights, Pins);

/// Builds a graph for an `Ask`.
type Build = fn(&Arch, &Ask) -> Result<Built>;

/// The host n-gram table a step graph reads, if any.
type Table = fn(&Ask, &Weights) -> Result<Option<PleConfig>>;

pub(crate) struct Arch {
    pub name: &'static str,
    pub spot: &'static [Spot],
    /// The weight name rules, as JSON.
    pub rules: Option<&'static str>,
    pub step: Option<Build>,
    /// The whole-prompt logits graph.
    pub forward: Option<Build>,
    pub part: Option<fn(&Arch, &Cut) -> Result<Part>>,
    pub embed: Option<Embeds>,
    pub ple: Option<Table>,
}

impl Arch {
    pub const NONE: Arch = Arch {
        name: "",
        spot: &[],
        rules: None,
        step: None,
        forward: None,
        part: None,
        embed: None,
        ple: None,
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
                Spot::Type(t) => v[MODEL_TYPE.name].as_str() == Some(t),
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
        [] => Err(unknown(v[MODEL_TYPE.name].as_str().unwrap_or("the config")).into()),
        many => {
            let names: Vec<&str> = many.iter().map(|a| a.name).collect();
            Err(format!("the config matches {}", names.join(" and ")).into())
        }
    }
}

/// `dir/config.json` with overrides, and the architecture it names.
pub(crate) fn read(dir: &Path, overrides: &Overrides) -> Result<(String, &'static Arch)> {
    let json = overrides.merged("config.json", &dir.join("config.json"))?;
    let arch = spot(&json)?;
    Ok((json, arch))
}

/// Streamed checkpoint release backed by an `Offer` and weight rules.
pub(crate) struct Streamed {
    pub graph: Graph,
    offer: Box<dyn Offer>,
    rules: Vec<Rule>,
}

impl Stream for Streamed {
    fn stream(
        &self,
        shares: &dyn Fn(u32) -> Option<(u32, usize, usize)>,
        take: &mut dyn FnMut(u32, Weight) -> Result<()>,
    ) -> Result<()> {
        numbered(&self.graph, &*self.offer, &[&self.rules], shares, take)
    }

    fn i64s(&self, name: &str) -> Result<Vec<i64>> {
        Ok(self.offer.i64s(name)?)
    }
}

/// Parses `config.json` through `Keys` and checks reads.
pub(crate) fn parse_keys<C>(
    json: &str,
    read: impl FnOnce(&mut tessera_engine::keys::Keys) -> C,
) -> Result<C> {
    let offered = [tessera_engine::keys::Offered::parse("config.json", json)?];
    let mut keys = tessera_engine::keys::Keys::new(&offered);
    let cfg = read(&mut keys);
    keys.finish()?;
    Ok(cfg)
}

/// Opens a streamed release with optional rotated-expert dispatch.
pub(crate) fn open_streamed<C>(
    ask: &Ask,
    is_step: bool,
    cfg: C,
    stack: impl FnOnce(&mut C) -> &mut tessera_engine::Stack,
    make_rules: impl FnOnce(&C, bool) -> Vec<Rule>,
    traces: [fn(&C) -> Graph; 4],
) -> Result<Built> {
    let offer = crate::offer(ask.dir, ask.manifest)?;
    streamed(offer, ask, is_step, cfg, stack, make_rules, traces)
}

/// `open_streamed` over an offer already open.
pub(crate) fn streamed<C>(
    offer: Box<dyn Offer>,
    ask: &Ask,
    is_step: bool,
    mut cfg: C,
    stack: impl FnOnce(&mut C) -> &mut tessera_engine::Stack,
    make_rules: impl FnOnce(&C, bool) -> Vec<Rule>,
    traces: [fn(&C) -> Graph; 4],
) -> Result<Built> {
    let rotated = crate::mla::has_rotated_experts(&offer.slices());
    let mut rules = make_rules(&cfg, rotated);
    let st = stack(&mut cfg);
    let layers = st.layers;
    if is_step {
        st.steps = ask.steps;
        st.taps = ask.taps.to_vec();
        rules.extend(ask.user.iter().cloned());
    }
    let pick = usize::from(rotated) * 2 + usize::from(is_step);
    let graph = traces[pick](&cfg);
    let pins = pins(&graph, &*offer, layers, &[&rules])?;
    let r = Streamed {
        graph: graph.clone(),
        offer,
        rules,
    };
    Ok((graph, Weights::Streamed(Box::new(r)), pins))
}

/// Shared and routed SwiGLU expert rules for one layer.
pub(crate) fn push_routed_mlp(
    rules: &mut Vec<(String, String)>,
    shared: (&str, &str),
    routed: (&str, &str, &str),
    rotated: bool,
) {
    let (shared_dst, shared_src) = shared;
    for (f, s) in MLP {
        rules.push((format!("{shared_dst}.{f}"), format!("{shared_src}.{s}")));
    }
    push_routed(rules, routed, rotated);
}

const MLP: [(&str, &str); 3] = [
    ("gate", "gate_proj"),
    ("up", "up_proj"),
    ("down", "down_proj"),
];

/// Routed SwiGLU expert rules for one layer.
pub(crate) fn push_routed(
    rules: &mut Vec<(String, String)>,
    (routed_dst, routed_pfx, experts_src): (&str, &str, &str),
    rotated: bool,
) {
    if rotated {
        for side in ["signs_in", "signs_out"] {
            rules.push((
                format!("{routed_dst}.{side}"),
                format!("{experts_src}.{side}"),
            ));
        }
    }
    for (f, s) in MLP {
        if rotated {
            let set = format!("{experts_src}.{f}");
            for end in ["su", "sv"] {
                rules.push((
                    format!("{routed_dst}.{routed_pfx}{f}_{end}"),
                    format!("{set}.{end}"),
                ));
            }
            rules.push((format!("{routed_dst}.{routed_pfx}{f}"), set));
        } else {
            rules.push((
                format!("{routed_dst}.{routed_pfx}{f}"),
                format!("{experts_src}.{{}}.{s}"),
            ));
        }
    }
}
