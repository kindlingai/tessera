//! Source text for generated kernels, in each backend's dialect.

use crate::ir::Intrinsic;

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
    pub buffers: Vec<(String, &'static str, bool)>,
    /// Name and type, bound after the buffers.
    pub scalars: Vec<(&'static str, &'static str)>,
    /// Functions the body calls, each once.
    pub headers: Vec<String>,
    pub body: String,
}

/// A pointer to a buffer's bytes.
pub const BYTES: &str = "f_const(u8)";

/// A helper's buffer parameter: a pointer to `ty`.
pub fn pointer(ty: &str, name: &str) -> String {
    format!("f_const({ty}) {name}")
}

/// A function defined once, called from kernel bodies.
pub fn function(ret: &str, name: &str, params: &[String], lines: &[String]) -> String {
    let (params, lines) = (params.join(", "), lines.join("\n"));
    format!("f_fn {ret} {name}({params}) {{\n{lines}\n}}\n")
}

/// The names a body may use, spelled per dialect.
const NAMES: &[(&str, &str, &str)] = &[
    ("u64", "unsigned long", "unsigned long long"),
    ("f_const(ty)", "device const ty *", "const ty *"),
    ("f_fn", "static inline", "static __device__ inline"),
    (
        "f_barrier()",
        "threadgroup_barrier(mem_flags::mem_threadgroup)",
        "__syncthreads()",
    ),
    (
        "f_group(ty, name, n)",
        "threadgroup ty name[n]",
        "__shared__ ty name[n]",
    ),
    (
        "f_u32x2(p)",
        "(*(device const uint2 *)(p))",
        "(*(const uint2 *)(p))",
    ),
];

/// Hopper on: wait for the last kernel, release the next.
const CUDA_PDL: &str = "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define f_pdl_wait() asm volatile(\"griddepcontrol.wait;\" ::: \"memory\")
#define f_pdl_go() asm volatile(\"griddepcontrol.launch_dependents;\")
#else
#define f_pdl_wait()
#define f_pdl_go()
#endif
";

/// `name` widens a word's low bits, `e` and `m` wide.
fn widen(name: &str, e: u32, m: u32) -> String {
    let bias = (1u32 << (e - 1)) - 1;
    let least = f32::from_bits((127 - bias - m + 1) << 23);
    format!(
        "f_fn float {name}(u32 x) {{\n    u32 ex = (x >> {m}u) & {}u;\n    u32 mn = x & {}u;\n    float v = ex == 0u ? (float)mn * {least:?}f : f_bits(((ex + {}u) << 23) | (mn << {}u));\n    return ((x >> {}u) & 1u) ? -v : v;\n}}\n",
        (1u32 << e) - 1,
        (1u32 << m) - 1,
        127 - bias,
        23 - m,
        e + m,
    )
}

/// `name` spelled one way per dialect: Metal, then CUDA.
fn define(name: &str, metal: &str, cuda: &str) -> [String; 2] {
    [metal, cuda].map(|s| format!("#define {name} {s}\n"))
}

fn spelled(metal: &str, cuda: &str) -> [String; 2] {
    [metal.into(), cuda.into()]
}

fn both(text: &str) -> [String; 2] {
    spelled(text, text)
}

fn none() -> [String; 2] {
    both("")
}

/// CUDA text with its own branch for HIP.
fn hip(hip: &str, cuda: &str) -> String {
    format!("#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)\n{hip}#else\n{cuda}#endif\n")
}

/// Each intrinsic's device text, from the table.
macro_rules! devices {
    ($($(#[$doc:meta])* $var:ident = $kind:ident $($name:ident)::+ ($($arg:tt)*) $(-> $ret:ident)?
        { device: $dev:expr, $($rest:tt)* })*) => {
        fn device(i: Intrinsic) -> [String; 2] {
            match i {
                $(Intrinsic::$var => $dev,)*
            }
        }
    };
}

crate::intrinsic::table!(devices);

/// The text before every kernel in `dialect`.
fn head(dialect: Dialect) -> String {
    let metal = dialect == Dialect::Metal;
    let mut text = String::new();
    if metal {
        text.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
    }
    text.push_str(
        "typedef unsigned int u32;\ntypedef unsigned short u16;\ntypedef unsigned char u8;\n",
    );
    for (name, for_metal, for_cuda) in NAMES {
        let spelling = if metal { for_metal } else { for_cuda };
        match *name {
            "u64" => text.push_str(&format!("typedef {spelling} u64;\n")),
            _ => text.push_str(&format!("#define {name} {spelling}\n")),
        }
    }
    for i in Intrinsic::ALL {
        let [for_metal, for_cuda] = device(*i);
        text.push_str(if metal { &for_metal } else { &for_cuda });
    }
    if !metal {
        text.push_str(CUDA_PDL);
    }
    text.push('\n');
    text
}

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
        // Bodies get `gid` and `lid`, the index in the group.
        let (head, kind, index) = match metal {
            true => {
                params.push("u32 gid [[thread_position_in_grid]]".into());
                params.push("u32 lid [[thread_index_in_threadgroup]]".into());
                (
                    format!("{}{}", head(dialect), crate::tile::METAL),
                    "kernel void",
                    "",
                )
            }
            false => {
                let index = "    f_pdl_wait();\n    f_pdl_go();\n    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;\n    u32 lid = threadIdx.x;\n";
                let head = format!("{}{}", head(dialect), crate::tile::cuda());
                (head, "extern \"C\" __global__ void", index)
            }
        };
        let (entry, params, body) = (&self.entry, params.join(",\n    "), &self.body);
        let headers = self.headers.concat();
        format!("{head}{headers}{kind} {entry}(\n    {params}\n) {{\n{index}{body}}}\n")
    }
}

/// A float buffer parameter.
pub fn float(name: &str, writes: bool) -> (String, &'static str, bool) {
    (name.into(), "float", writes)
}

/// FNV-1a, so equal bodies can share one entry name.
pub fn hash(text: &str) -> u64 {
    let mix = |h: u64, b: u8| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    text.bytes().fold(0xcbf2_9ce4_8422_2325, mix)
}
