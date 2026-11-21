//! Each rank's lowering by stage, linked to source nodes.

use crate::model::Model;
use crate::ranks::{Spec, Wire, device};
use serde::Serialize;
use std::collections::HashMap;
use std::error::Error;
use tessera_lower::{Device, Pins, Source, Step, Tree, assign, peak, schedule, select};

/// One step of a rank's plan, for views.
#[derive(Clone, Debug, Serialize)]
pub struct StepView {
    pub kernel: &'static str,
    /// The kernel with every size.
    pub detail: String,
    pub rule: &'static str,
    pub zone: u32,
    /// Source nodes this launch computes, or helps compute.
    pub covers: Vec<u32>,
    pub bytes: usize,
    pub seconds: f64,
    /// Set when the step crosses to the other ranks.
    pub sum: bool,
    /// Index of its text in `Pass::texts`.
    pub text: usize,
}

/// A rank's part of one weight.
#[derive(Clone, Debug, Serialize)]
pub struct Held {
    pub weight: u32,
    pub bytes: usize,
    /// Axis, first, count and whole length, when split.
    pub share: Option<[usize; 4]>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RankView {
    pub name: String,
    pub backend: &'static str,
    pub arch: String,
    pub part: u32,
    /// Per sharded node, its shape on this rank.
    pub shapes: Vec<Option<Vec<usize>>>,
    pub ops: usize,
    pub contested: Vec<&'static str>,
    /// Bytes live at most at once, in schedule order.
    pub peak: usize,
    pub buffers: usize,
    pub steps: Vec<StepView>,
    pub held: Vec<Held>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Text {
    pub kernel: &'static str,
    pub dialect: &'static str,
    pub text: String,
}

/// Every rank's plan for one launch.
#[derive(Clone, Debug, Serialize)]
pub struct Pass {
    pub label: String,
    pub rows: u64,
    pub context: u64,
    pub ranks: Vec<RankView>,
    /// Kernel texts, each once.
    pub texts: Vec<Text>,
}

/// Inputs the views need to lower each rank.
pub struct Setup<'a> {
    pub model: &'a Model,
    pub specs: &'a [Spec],
    pub wire: Wire,
    pub pins: Pins,
}

/// The text a backend compiles, or the host's tree.
fn text_of(step: &Step, spec: &Spec) -> (&'static str, String) {
    let dialect = spec.dialect().unwrap_or(tessera_lower::Dialect::Cuda);
    let name = match (spec.dialect(), dialect) {
        (None, _) => "tree",
        (_, tessera_lower::Dialect::Metal) => "metal",
        _ if spec.backend == "rocm" => "hip",
        _ => "cuda",
    };
    let text = match step.kernel.source() {
        Source::Fixed {
            metal, cuda, hip, ..
        } => match name {
            "metal" => metal.to_string(),
            "hip" => hip.to_string(),
            _ => cuda.to_string(),
        },
        Source::Made(body) => body.text().render(dialect),
    };
    (name, text)
}

/// Each sharded node's shape on this rank, from the tree.
fn shapes(
    model: &Model,
    device: &Device,
    sizes: &[(tessera_graph::DimName, u64)],
) -> Result<Vec<Option<Vec<usize>>>, Box<dyn Error>> {
    let (g, parts) = (&model.sharded, device.division());
    let tree = Tree::upto(
        g,
        0,
        sizes,
        &[],
        &device.pins,
        device.kept,
        &parts,
        device.rank,
    )?;
    let mut out = vec![None; g.nodes.len()];
    for t in 0..tree.len() {
        let node = tree.node(tessera_lower::T(t as u32));
        out[node.from.0 as usize] = Some(node.shape.clone());
    }
    Ok(out)
}

impl Setup<'_> {
    /// Every rank lowered for `rows` new rows at `context`.
    pub fn pass(&self, label: String, rows: u64, context: u64) -> Result<Pass, Box<dyn Error>> {
        let sizes = self.model.sizes(rows, context);
        let mut texts: Vec<Text> = Vec::new();
        let mut seen: HashMap<(String, &'static str), usize> = HashMap::new();
        let mut ranks = Vec::new();
        for (r, spec) in self.specs.iter().enumerate() {
            let device = device(self.specs, r, self.wire, &self.pins);
            let picked = select(&self.model.sharded, 0, &sizes, &device)?;
            let order = schedule(&picked);
            let plan = assign(&picked, &order);
            let mut steps = Vec::new();
            for step in &plan.steps {
                let detail = format!("{:?}", step.kernel);
                let (dialect, text) = text_of(step, spec);
                let at = *seen.entry((detail.clone(), dialect)).or_insert_with(|| {
                    let kernel = step.kernel.name();
                    texts.push(Text {
                        kernel,
                        dialect,
                        text,
                    });
                    texts.len() - 1
                });
                let mut covers: Vec<u32> = Vec::new();
                for n in &step.covers {
                    let from = self.model.origin[n.0 as usize].0;
                    if !covers.contains(&from) {
                        covers.push(from);
                    }
                }
                steps.push(StepView {
                    kernel: step.kernel.name(),
                    detail,
                    rule: step.rule,
                    zone: step.zone,
                    covers,
                    bytes: step.write.bytes(),
                    seconds: device.step_seconds(&plan, step).unwrap_or(0.0),
                    sum: tessera_lower::kernels::crosses(step.kernel.name()),
                    text: at,
                });
            }
            let held = plan.weights.iter().map(|l| Held {
                weight: l.weight,
                bytes: l.slot.bytes(),
                share: l
                    .share
                    .map(|s| [s.axis as usize, s.first, s.count, s.whole]),
            });
            ranks.push(RankView {
                name: spec.name.clone(),
                backend: spec.backend,
                arch: spec.arch.clone(),
                part: spec.part,
                shapes: shapes(self.model, &device, &sizes)?,
                ops: picked.ops.len(),
                contested: picked.contested.iter().copied().collect(),
                peak: peak(&picked, &order),
                buffers: plan.buffers.iter().sum(),
                steps,
                held: held.collect(),
            });
        }
        Ok(Pass {
            label,
            rows,
            context,
            ranks,
            texts,
        })
    }
}
