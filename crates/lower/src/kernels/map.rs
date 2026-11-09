//! Generated elementwise kernels, one per chain of stages.

use crate::abi::{Arg, Launch, Shape, Source};
use crate::emit::{Text, float, hash};
use crate::kernel::Kernel;
use crate::plan::Step;
use crate::rule::{Draft, Site};
use graph::Act;
use std::fmt::Write;

/// One elementwise step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MapStage {
    /// Adds the next operand.
    Add,
    /// Multiplies by the next operand.
    Mul,
    Scale(f32),
    Act(Act),
}

/// `out = stages(x, operands..)`. No stages makes a copy.
#[derive(Clone, Debug)]
pub struct Map(pub Vec<MapStage>);

fn act(a: Act) -> &'static str {
    match a {
        Act::Silu => "v = v / (1.0f + f_exp(-v));",
        Act::Sigmoid => "v = 1.0f / (1.0f + f_exp(-v));",
        Act::Relu => "v = f_max(v, 0.0f);",
        Act::GeluTanh => {
            "v = 0.5f * v * (1.0f + f_tanh(f_clamp(\
             0.7978846f * (v + 0.044715f * v * v * v), -8.0f, 8.0f)));"
        }
    }
}

impl Kernel for Map {
    fn name(&self) -> &'static str {
        "map"
    }

    fn source(&self) -> Source {
        let mut buffers = vec![float("x", false)];
        let start = "    if (gid >= len) {\n        return;\n    }\n    float v = x[gid];\n";
        let mut body = start.to_string();
        for s in &self.0 {
            let line = match s {
                MapStage::Add | MapStage::Mul => {
                    let op = if *s == MapStage::Add { '+' } else { '*' };
                    let name = format!("a{}", buffers.len() - 1);
                    buffers.push(float(&name, false));
                    format!("v = v {op} {name}[gid];")
                }
                // Bits, so the constant survives the round trip.
                MapStage::Scale(by) => format!("v = v * f_bits(0x{:08x}u);", by.to_bits()),
                MapStage::Act(a) => act(*a).to_string(),
            };
            let _ = writeln!(body, "    {line}");
        }
        body.push_str("    out[gid] = v;\n");
        buffers.push(float("out", true));
        Source::Made(Text {
            entry: format!("map_{:x}", hash(&body)),
            buffers,
            scalars: vec![("len", "u32")],
            body,
        })
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let len = step.write.len;
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.extend([step.out(), Arg::count(len)]);
        vec![Launch {
            args,
            shape: Shape::Line(len),
        }]
    }

    fn work(&self, len: usize) -> f64 {
        (len * (1 + self.0.len())) as f64
    }
}

fn propose(site: &Site) -> Option<Vec<Draft>> {
    let Site::Map { stages, reads, len } = *site else {
        return None;
    };
    Some(vec![Draft::new(Map(stages.to_vec()), reads.to_vec(), len)])
}

crate::rule!("map", propose);
