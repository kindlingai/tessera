//! Inkling-2 (`inkling2_loop`) directory loading and weight rules.

#[cfg(test)]
mod inkling2_test;

use crate::arch::{Arch, Ask, Built, Spot, arch, parse_keys, push_routed, streamed};
use crate::names::Rule;
use crate::parts::{Cut, Part, from_opened};
use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use tessera_checkpoint::{Offer, Slice, Stored};
use tessera_engine::Filed;
use tessera_engine::inkling2::{Coded, Config, Plain, program, stepper};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

arch!(
    "inkling2",
    [
        Spot::Type("inkling2_loop"),
        Spot::Class("Inkling2LoopForConditionalGeneration"),
    ],
    step: step,
    forward: forward,
    part: part,
);

fn open(dir: &Path, manifest: Option<&str>) -> Result<Box<dyn Offer>> {
    Ok(Box::new(Inkling2Offer::new(crate::offer(dir, manifest)?)))
}

fn build(ask: &Ask, is_step: bool) -> Result<Built> {
    let cfg = parse_config(ask.json, ask.sizes)?;
    let offer = open(ask.dir, ask.manifest)?;
    let plain = [program::<Plain>, stepper::<Plain>];
    let [forward, step] = [program::<Coded>, stepper::<Coded>];
    let traces = [plain[0], plain[1], forward, step];
    streamed(offer, ask, is_step, cfg, |c| &mut c.stack, rules, traces)
}

fn step(_: &Arch, ask: &Ask) -> Result<Built> {
    build(ask, true)
}

fn forward(_: &Arch, ask: &Ask) -> Result<Built> {
    build(ask, false)
}

fn part(_: &Arch, cut: &Cut) -> Result<Part> {
    let cfg = parse_config(cut.json, cut.sized(256))?;
    let layers = cfg.stack.layers;
    let traces = [stepper::<Plain>, stepper::<Coded>];
    from_opened(cut, cfg, layers, rules, traces, open)
}

pub(crate) fn parse_config(json: &str, [seqs, context]: [u64; 2]) -> Result<Config> {
    parse_keys(json, |keys| Config::read(keys, seqs, context))
}

/// Maps traced Inkling-2 weights to checkpoint tensor names.
pub(crate) fn rules(c: &Config, rotated: bool) -> Vec<Rule> {
    let top = "model.llm";
    let mut rules: Vec<Rule> = [
        ("embed", "embed"),
        ("embed_norm", "embed_norm"),
        ("norm", "norm"),
        ("head", "unembed"),
        ("output_query", "attn_res.output_query"),
        ("cross_pass_queries", "middle_cycle_cross_pass_queries"),
        (
            "cross_pass_kv_queries",
            "middle_cycle_cross_pass_kv_queries",
        ),
    ]
    .map(|(dst, src)| (dst.into(), format!("{top}.{src}")))
    .into();
    for l in 0..c.stack.layers as usize {
        let (dst, src) = (format!("layers.{l}"), format!("{top}.layers.{l}"));
        let queries = format!("{top}.attn_res");
        let mut push = |d: &str, s: String| rules.push((format!("{dst}.{d}"), s));
        if l > 0 {
            push("attn_query", format!("{queries}.attn_queries.{}", l - 1));
        }
        push("mlp_query", format!("{queries}.mlp_queries.{l}"));
        let global = c.global[l];
        let own: &[(&str, &str)] = if global {
            &[
                ("q_norm", "q_input_norm"),
                ("k_norm", "k_input_norm"),
                ("v_norm", "v_input_norm"),
                ("attn.k_static", "attn.k_sconv.static_w"),
                ("attn.v_static", "attn.v_sconv.static_w"),
                ("attn.k_wproj", "attn.k_sconv.weight_proj"),
                ("attn.v_wproj", "attn.v_sconv.weight_proj"),
            ]
        } else {
            &[
                ("attn_norm", "attn_norm"),
                ("attn.k_sconv", "attn.k_sconv"),
                ("attn.v_sconv", "attn.v_sconv"),
            ]
        };
        let shared = [
            ("attn.wq", "attn.wq_du"),
            ("attn.wk", "attn.wk_dv"),
            ("attn.wv", "attn.wv_dv"),
            ("attn.wo", "attn.wo_ud"),
            ("attn.wr", "attn.wr_du"),
            ("attn.gate", "attn.gate_attn"),
            ("attn.sink", "attn.attn_sink_proj"),
            ("attn.q_clip", "attn.q_l2_clip.scale"),
            ("attn.rel_clip", "attn.rel_l2_clip.scale"),
            ("attn.rel_proj", "attn.rel_logits_proj.proj"),
            ("attn_sconv", "attn_sconv"),
            ("mlp_sconv", "mlp_sconv"),
            ("mlp_norm", "mlp_norm"),
            ("moe.router", "mlp.gate"),
            ("moe.router_bias", "mlp.gate.bias"),
        ];
        for (d, s) in own.iter().chain(&shared) {
            push(d, format!("{src}.{s}"));
        }
        if c.is_looped() && (c.loop_start..c.loop_end).contains(&(l as u64)) {
            let gates = [("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp")];
            for ((d, s), part) in gates
                .into_iter()
                .flat_map(|g| ["down", "forget", "gain"].map(|p| (g, p)))
            {
                push(
                    &format!("ctrl.{d}.{part}"),
                    format!("{src}.controllers.{s}.{part}_weight"),
                );
            }
        }
        let experts = (format!("{dst}.moe.experts"), format!("{src}.mlp.experts"));
        push_routed(&mut rules, (&experts.0, "", &experts.1), rotated);
    }
    rules
}

/// How a stored matrix turns into the traced layout.
#[derive(Clone, Copy)]
enum Turn {
    /// `[a, b]` to `[b, a]`.
    Swap,
    /// `[taps, channels]` to `[channels, taps]`, taps reversed.
    SwapFlip,
    /// Rows reversed within each group of this many.
    FlipRows(usize),
}

impl Turn {
    fn shape(self, shape: &[usize]) -> Vec<usize> {
        match self {
            Turn::FlipRows(_) => shape.to_vec(),
            _ => vec![shape[1], shape[0]],
        }
    }

    fn apply(self, [a, b]: [usize; 2], x: &[f32]) -> Vec<f32> {
        let from = |i: usize| match self {
            Turn::Swap => (i % a) * b + i / a,
            Turn::SwapFlip => (a - 1 - i % a) * b + i / a,
            Turn::FlipRows(k) => {
                let row = i / b;
                (row - row % k + (k - 1 - row % k)) * b + i % b
            }
        };
        (0..x.len()).map(|i| x[from(i)]).collect()
    }
}

type Turns = HashMap<String, (Turn, [usize; 2])>;

/// Finds each lag-ordered matrix and its turn.
fn turns(slices: &[Slice]) -> Turns {
    let taps = |name: &str| {
        let fixed = name.replace(".weight_proj", ".static_w");
        let found = slices.iter().find(|s| s.name == fixed);
        found.map(|s| s.shape[0])
    };
    let turn = |s: &Slice| {
        let name = s.name.as_str();
        let conv = name.contains("_sconv.");
        if name.ends_with(".rel_logits_proj.proj") {
            Some(Turn::Swap)
        } else if conv && name.ends_with(".static_w") {
            Some(Turn::SwapFlip)
        } else if conv && name.ends_with(".weight_proj") {
            taps(name).map(Turn::FlipRows)
        } else {
            None
        }
    };
    let flat = slices.iter().filter(|s| s.shape.len() == 2);
    let turned = flat.filter_map(|s| {
        let shape = [s.shape[0], s.shape[1]];
        Some((s.name.clone(), (turn(s)?, shape)))
    });
    turned.collect()
}

/// Reports turned slices in their traced layout.
fn turn_slices(mut slices: Vec<Slice>, turned: &Turns) -> Vec<Slice> {
    for s in &mut slices {
        if let Some((turn, _)) = turned.get(&s.name) {
            s.shape = turn.shape(&s.shape);
            s.stored = Stored::F32;
        }
    }
    slices
}

/// An offer with Inkling-2's lag-ordered tensors turned.
pub(crate) struct Inkling2Offer {
    inner: Box<dyn Offer>,
    /// Each turned slice's turn and stored shape.
    turned: Turns,
}

impl Inkling2Offer {
    pub(crate) fn new(inner: Box<dyn Offer>) -> Self {
        let turned = turns(&inner.slices());
        Self { inner, turned }
    }

    fn refuse(&self, name: &str) -> std::result::Result<(), tessera_checkpoint::Error> {
        if self.turned.contains_key(name) {
            let why = format!("{name} is only read as floats");
            return Err(tessera_checkpoint::Error::DType(why));
        }
        Ok(())
    }
}

type Read<T> = std::result::Result<T, tessera_checkpoint::Error>;

impl Offer for Inkling2Offer {
    fn slices(&self) -> Vec<Slice> {
        turn_slices(self.inner.slices(), &self.turned)
    }

    fn floats(&self, name: &str) -> Read<Vec<f32>> {
        let x = self.inner.floats(name)?;
        Ok(match self.turned.get(name) {
            Some(&(turn, shape)) => turn.apply(shape, &x),
            None => x,
        })
    }

    fn packed(&self, name: &str) -> Read<Vec<u8>> {
        self.refuse(name)?;
        self.inner.packed(name)
    }

    fn i64s(&self, name: &str) -> Read<Vec<i64>> {
        self.inner.i64s(name)
    }

    fn filed(&self, name: &str) -> Option<Arc<dyn Filed>> {
        self.refuse(name).ok()?;
        self.inner.filed(name)
    }

    fn packed_rows(&self, name: &str, rows: (usize, usize)) -> Read<Vec<u8>> {
        self.refuse(name)?;
        self.inner.packed_rows(name, rows)
    }

    fn packed_share(&self, name: &str, rows: (usize, usize)) -> Read<Vec<u8>> {
        self.refuse(name)?;
        self.inner.packed_share(name, rows)
    }

    fn packed_axis(&self, name: &str, axis: u32, range: (usize, usize)) -> Read<Vec<u8>> {
        self.refuse(name)?;
        self.inner.packed_axis(name, axis, range)
    }
}
