//! Source text for generated kernels, in each backend's dialect.

use crate::plan::MapStage;
use graph::Act;
use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    Metal,
    /// Also what hiprtc compiles for ROCm.
    Cuda,
}

/// One kernel: buffers, then scalars, then a body over `gid`.
pub struct Text {
    pub entry: String,
    /// Name, element type, and whether the kernel writes it.
    buffers: Vec<(String, &'static str, bool)>,
    scalars: Vec<(&'static str, &'static str)>,
    body: String,
}

/// Each dialect's spelling of the names a body may use.
const METAL: &str = "#include <metal_stdlib>
using namespace metal;
#define f_exp(x) exp(x)
#define f_tanh(x) tanh(x)
#define f_max(a, b) max(a, b)
#define f_clamp(x, lo, hi) clamp(x, lo, hi)
#define f_pow(a, b) precise::pow(a, b)
#define f_cos(x) precise::cos(x)
#define f_sin(x) precise::sin(x)
#define f_bits(x) as_type<float>(x)

";

const CUDA: &str = "typedef unsigned int uint;
typedef unsigned long long ulong;
#define f_exp(x) expf(x)
#define f_tanh(x) tanhf(x)
#define f_max(a, b) fmaxf(a, b)
#define f_clamp(x, lo, hi) fminf(fmaxf(x, lo), hi)
#define f_pow(a, b) powf(a, b)
#define f_cos(x) cosf(x)
#define f_sin(x) sinf(x)
#define f_bits(x) __uint_as_float(x)

";

impl Text {
    pub fn render(&self, dialect: Dialect) -> String {
        let metal = dialect == Dialect::Metal;
        let mut params = Vec::new();
        for (at, (name, ty, writes)) in self.buffers.iter().enumerate() {
            let fixed = if *writes { "" } else { "const " };
            params.push(match metal {
                true => format!("device {fixed}{ty} *{name} [[buffer({at})]]"),
                false => format!("{fixed}{ty} *{name}"),
            });
        }
        for (at, (name, ty)) in self.scalars.iter().enumerate() {
            let at = self.buffers.len() + at;
            params.push(match metal {
                true => format!("constant {ty} &{name} [[buffer({at})]]"),
                false => format!("{ty} {name}"),
            });
        }
        let (head, kind, index) = match metal {
            true => {
                params.push("uint gid [[thread_position_in_grid]]".into());
                (METAL, "kernel void", "")
            }
            false => {
                let index = "    uint gid = blockIdx.x * blockDim.x + threadIdx.x;\n";
                (CUDA, "extern \"C\" __global__ void", index)
            }
        };
        let (entry, params, body) = (&self.entry, params.join(",\n    "), &self.body);
        format!("{head}{kind} {entry}(\n    {params}\n) {{\n{index}{body}}}\n")
    }
}

fn float(name: &str, writes: bool) -> (String, &'static str, bool) {
    (name.into(), "float", writes)
}

/// FNV-1a, so equal bodies share one entry name.
fn hash(text: &str) -> u64 {
    let mix = |h: u64, b: u8| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    text.bytes().fold(0xcbf2_9ce4_8422_2325, mix)
}

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

/// `out = stages(x, operands..)`, one element per thread.
pub fn map(stages: &[MapStage]) -> Text {
    let mut buffers = vec![float("x", false)];
    let mut body =
        "    if (gid >= len) {\n        return;\n    }\n    float v = x[gid];\n".to_string();
    for s in stages {
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
    Text {
        entry: format!("map_{:x}", hash(&body)),
        buffers,
        scalars: vec![("len", "uint")],
        body,
    }
}

/// Rows of `a` and `b`, joined along the last axis.
pub fn join() -> Text {
    let body = "    uint width = left + right;
    uint row = gid / width;
    uint col = gid % width;
    if (row >= rows) {
        return;
    }
    out[gid] = col < left ? a[row * left + col] : b[row * right + col - left];
";
    Text {
        entry: "plan_join".into(),
        buffers: vec![float("a", false), float("b", false), float("out", true)],
        scalars: vec![("left", "uint"), ("right", "uint"), ("rows", "uint")],
        body: body.into(),
    }
}

/// Each thread sums one output column for every row.
pub fn thin() -> Text {
    let body = "    if (gid >= n) {
        return;
    }
    float sums[32];
    for (uint row = 0; row < m; row++) {
        sums[row] = 0.0f;
    }
    for (uint i = 0; i < k; i++) {
        float c = w[(ulong)gid * k + i];
        for (uint row = 0; row < m; row++) {
            sums[row] += x[row * k + i] * c;
        }
    }
    for (uint row = 0; row < m; row++) {
        out[(ulong)row * n + gid] = sums[row];
    }
";
    Text {
        entry: "plan_thin".into(),
        buffers: vec![float("x", false), float("w", false), float("out", true)],
        scalars: vec![("m", "uint"), ("n", "uint"), ("k", "uint")],
        body: body.into(),
    }
}

/// A cosine and sine per position and pair.
pub fn angles() -> Text {
    let body = "    uint pairs = width / 2;
    uint at = gid / pairs;
    uint pair = gid % pairs;
    if (at >= count) {
        return;
    }
    float turn = f_pow(theta, -(float)(2 * pair) / (float)width);
    float angle = (float)positions[at] * turn;
    out[at * width + 2 * pair] = f_cos(angle);
    out[at * width + 2 * pair + 1] = f_sin(angle);
";
    Text {
        entry: "plan_angles".into(),
        buffers: vec![("positions".into(), "uint", false), float("out", true)],
        scalars: vec![("count", "uint"), ("width", "uint"), ("theta", "float")],
        body: body.into(),
    }
}
