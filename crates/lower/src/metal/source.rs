//! Metal source for the kernels a plan generates.

use crate::plan::MapStage;
use graph::Act;
use std::fmt::Write;

pub(crate) struct Source {
    pub entry: String,
    pub text: String,
}

const HEADER: &str = "#include <metal_stdlib>\nusing namespace metal;\n\n";

fn act(a: Act) -> &'static str {
    match a {
        Act::Silu => "v = v / (1.0f + exp(-v));",
        Act::Sigmoid => "v = 1.0f / (1.0f + exp(-v));",
        Act::Relu => "v = max(v, 0.0f);",
        Act::GeluTanh => {
            "v = 0.5f * v * (1.0f + tanh(clamp(\
             0.7978846f * (v + 0.044715f * v * v * v), -8.0f, 8.0f)));"
        }
    }
}

/// `out = stages(x, operands..)`, one element per thread.
pub(crate) fn map(stages: &[MapStage]) -> Source {
    let (mut body, mut operands) = (String::new(), 0);
    for s in stages {
        let line = match s {
            MapStage::Add | MapStage::Mul => {
                let op = if *s == MapStage::Add { '+' } else { '*' };
                operands += 1;
                format!("v = v {op} a{}[gid];", operands - 1)
            }
            // Bits, so the constant survives the round trip.
            MapStage::Scale(by) => format!("v = v * as_type<float>(0x{:08x}u);", by.to_bits()),
            MapStage::Act(a) => act(*a).to_string(),
        };
        let _ = writeln!(body, "    {line}");
    }
    let entry = format!("map_{:x}", gpukit::metal::source_hash(&body));
    let mut text = format!("{HEADER}kernel void {entry}(\n");
    let _ = writeln!(text, "    device const float *x [[buffer(0)]],");
    for a in 0..operands {
        let at = a + 1;
        let _ = writeln!(text, "    device const float *a{a} [[buffer({at})]],");
    }
    let (out, len) = (operands + 1, operands + 2);
    let _ = write!(
        text,
        "    device float *out [[buffer({out})]],\n    \
         constant uint &len [[buffer({len})]],\n    \
         uint gid [[thread_position_in_grid]]\n) {{\n    \
         if (gid >= len) {{\n        return;\n    }}\n    \
         float v = x[gid];\n{body}    out[gid] = v;\n}}\n"
    );
    Source { entry, text }
}

/// Rows of `a` and `b`, joined; `dims` holds both widths.
pub(crate) fn join() -> Source {
    let body = "kernel void plan_join(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint3 &dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint width = dims.x + dims.y;
    uint row = gid / width;
    uint col = gid % width;
    if (row >= dims.z) {
        return;
    }
    out[gid] = col < dims.x ? a[row * dims.x + col] : b[row * dims.y + col - dims.x];
}
";
    Source {
        entry: "plan_join".into(),
        text: format!("{HEADER}{body}"),
    }
}

/// Each thread sums one output column for every row.
pub(crate) fn thin() -> Source {
    let body = "kernel void plan_thin(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint3 &dims [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dims.y) {
        return;
    }
    float sums[32];
    for (uint row = 0; row < dims.x; row++) {
        sums[row] = 0.0f;
    }
    const device float *col = w + (ulong)gid * dims.z;
    for (uint i = 0; i < dims.z; i++) {
        float c = col[i];
        for (uint row = 0; row < dims.x; row++) {
            sums[row] += x[row * dims.z + i] * c;
        }
    }
    for (uint row = 0; row < dims.x; row++) {
        out[(ulong)row * dims.y + gid] = sums[row];
    }
}
";
    Source {
        entry: "plan_thin".into(),
        text: format!("{HEADER}{body}"),
    }
}

/// A cosine and sine per position and pair.
pub(crate) fn angles() -> Source {
    let body = "kernel void plan_angles(
    device const uint *positions [[buffer(0)]],
    device float *out [[buffer(1)]],
    constant uint2 &dims [[buffer(2)]],
    constant float &theta [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint pairs = dims.y / 2;
    uint at = gid / pairs;
    uint pair = gid % pairs;
    if (at >= dims.x) {
        return;
    }
    float turn = precise::pow(theta, -float(2 * pair) / float(dims.y));
    float angle = float(positions[at]) * turn;
    out[at * dims.y + 2 * pair] = precise::cos(angle);
    out[at * dims.y + 2 * pair + 1] = precise::sin(angle);
}
";
    Source {
        entry: "plan_angles".into(),
        text: format!("{HEADER}{body}"),
    }
}
