//! Lowers an engine's graph under another recipe, without its runner.

use crate::ranks::{Variant, pick_graph};
use crate::{Engine, Error, Row, launch_of};
use serde::Deserialize;
use tessera_graph::DimName::{Context, Segments};
use tessera_graph::{Graph, Role};
use tessera_lower::{Cache, Device, Plan, Recipe, Report, Wants};

/// The launch a report lowers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Launch {
    /// One decode step of this many rows over the slots.
    Decode(usize),
    /// One prompt's prefill step of this many rows.
    Prefill(usize),
}

/// Rows and picks that stand for a launch of `rows`.
pub(crate) fn sample(
    rows: usize,
    prefill: bool,
    slots: usize,
    context: usize,
) -> (Vec<Row>, Vec<u32>) {
    let at = if prefill { rows } else { 1 };
    sample_at(rows, prefill, slots, context, at)
}

/// Rows and picks for `rows` reaching `at_context`.
pub(crate) fn sample_at(
    rows: usize,
    prefill: bool,
    slots: usize,
    max_context: usize,
    at_context: usize,
) -> (Vec<Row>, Vec<u32>) {
    let rows = match prefill {
        true => rows.clamp(1, max_context),
        false => rows.max(1),
    };
    let last = max_context.saturating_sub(1);
    let end = at_context.clamp(1, max_context);
    let base = match prefill {
        true => end.saturating_sub(rows),
        false => end.saturating_sub(1).saturating_sub((rows - 1) / slots),
    };
    let mut sample = Vec::with_capacity(rows);
    for i in 0..rows {
        sample.push(match prefill {
            true => Row {
                slot: 0,
                token: 0,
                position: (base + i).min(last) as u32,
            },
            false => Row {
                slot: (i % slots) as u32,
                token: 0,
                position: (base + i / slots).min(last) as u32,
            },
        });
    }
    let picks = match prefill {
        true => vec![rows as u32 - 1],
        false => (0..rows as u32).collect(),
    };
    (sample, picks)
}

/// The engine fields that lowering reads.
pub(crate) struct Lowering<'a> {
    pub graphs: (&'a Graph, &'a Variant, &'a Variant),
    pub full: usize,
    pub taps: bool,
    pub slots: usize,
    pub context: usize,
    pub device: &'a Device,
}

/// The graph a plan's nodes refer to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Picked {
    Base,
    Owned,
    Split,
}

impl Lowering<'_> {
    /// Lowers `launch` through `plans` for a report.
    pub(crate) fn explain(&self, plans: &mut Cache, launch: Launch) -> Result<Lowered, Error> {
        let (rows, prefill) = match launch {
            Launch::Decode(rows) => (rows, false),
            Launch::Prefill(rows) => (rows, true),
        };
        let (sample, picks) = sample(rows, prefill, self.slots, self.context);
        let (mut sizes, [filled, _]) = launch_of(&sample, &picks, self.slots, self.context);
        let (base, owned, split) = self.graphs;
        let wants = match self.taps {
            true => Wants::roles(base, self.full, &[Role::Tokens, Role::Taps]),
            false => Wants::role(base, self.full, Role::Tokens),
        };
        let upto = &[Context, Segments];
        let device = self.device;
        let graph = pick_graph(self.graphs, device, true, filled, &mut sizes);
        let mut report = plans.explain(graph, self.full, wants, &sizes, upto, device)?;
        let plan = plans.wanting(graph, self.full, wants, &sizes, upto, device)?;
        report.label = match prefill {
            true => format!("prefill rows={filled}"),
            false => format!("rows={filled}"),
        };
        let is = |v: &Variant| v.as_ref().is_some_and(|(g, _)| std::ptr::eq(graph, g));
        let picked = match (is(owned), is(split)) {
            (true, _) => Picked::Owned,
            (_, true) => Picked::Split,
            _ => Picked::Base,
        };
        Ok(Lowered {
            report,
            plan: plan.0.clone(),
            picked,
        })
    }
}

/// One launch lowered for a report.
pub struct Lowered {
    pub report: Report,
    pub plan: Plan,
    picked: Picked,
}

/// An engine's graphs and device, held apart from its runner.
pub struct Probe {
    graph: Graph,
    owned: Variant,
    split: Variant,
    full: usize,
    taps: bool,
    slots: usize,
    context: usize,
    device: Device,
    name: String,
}

impl Probe {
    /// Lowers `launch` as `recipe` would, on this device.
    pub fn explain(&self, recipe: &Recipe, launch: Launch) -> Result<Lowered, Error> {
        let device = self.device.clone().with(recipe);
        let view = Lowering {
            graphs: (&self.graph, &self.owned, &self.split),
            full: self.full,
            taps: self.taps,
            slots: self.slots,
            context: self.context,
            device: &device,
        };
        view.explain(&mut Cache::default(), launch)
    }

    /// The device's name.
    pub fn device(&self) -> &str {
        &self.name
    }

    /// The graph lowering reads, before any row split.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// The graph `lowered`'s plan nodes refer to.
    pub fn graph_of(&self, lowered: &Lowered) -> &Graph {
        let variant = match lowered.picked {
            Picked::Base => &None,
            Picked::Owned => &self.owned,
            Picked::Split => &self.split,
        };
        variant.as_ref().map_or(&self.graph, |(g, _)| g)
    }
}

impl Engine {
    /// A copy of the graphs and device, to lower elsewhere.
    pub fn probe(&self) -> Probe {
        Probe {
            graph: self.graph.clone(),
            owned: self.owned.clone(),
            split: self.split.clone(),
            full: self.full,
            taps: self.taps.is_some(),
            slots: self.slots,
            context: self.context,
            device: self.device.clone(),
            name: self.runner.device(),
        }
    }
}
