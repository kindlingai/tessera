//! Source text for generated kernels, in each backend's dialect.

use crate::ir::Intrinsic;
use tessera_engine_settings::Backend;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    Metal,
    /// Also what hiprtc compiles for ROCm.
    Cuda,
}

impl Dialect {
    /// The text `backend` compiles.
    pub fn of(backend: Backend) -> Self {
        match backend {
            Backend::Metal => Dialect::Metal,
            Backend::Cuda | Backend::Rocm => Dialect::Cuda,
        }
    }

    /// The extension of a file in this dialect.
    pub fn extension(self) -> &'static str {
        match self {
            Dialect::Metal => "metal",
            Dialect::Cuda => "cu",
        }
    }
}

/// One kernel: buffers, then scalars, then a body over `gid`.
pub struct Text {
    pub entry: String,
    /// Name, element type, and whether the kernel writes it.
    pub buffers: Vec<(String, &'static str, bool)>,
    /// Name and type, bound after the buffers.
    pub scalars: Vec<(String, &'static str)>,
    /// Functions the body calls, each once.
    pub headers: Vec<String>,
    pub body: String,
}

/// A pointer to a buffer's bytes.
pub const BYTES: &str = "f_const(u8)";

/// A helper's buffer parameter: a pointer to `ty`.
pub(crate) fn pointer(ty: &str, name: &str) -> String {
    format!("f_const({ty}) {name}")
}

/// A function defined once, called from kernel bodies.
pub fn function(ret: &str, name: &str, params: &[String], lines: &[String]) -> String {
    let (params, lines) = (params.join(", "), lines.join("\n"));
    format!("f_fn {ret} {name}({params}) {{\n{lines}\n}}\n")
}

/// A WGSL function: `fn sig -> ret { lines }`.
pub(crate) fn wgsl_function(sig: &str, ret: &str, lines: &str) -> String {
    format!("fn {sig} -> {ret} {{\n{lines}\n}}\n")
}

/// A WGSL entry: `gid` counts threads over both group axes.
pub(crate) fn wgsl_entry(entry: &str, lines: &str) -> String {
    let groups = "@builtin(workgroup_id) f_group: vec3<u32>,\n    \
                  @builtin(num_workgroups) f_groups: vec3<u32>,\n    \
                  @builtin(local_invocation_index) lid: u32,";
    let gid = "let gid = (f_group.y * f_groups.x + f_group.x) * f_threads + lid;";
    format!(
        "@compute @workgroup_size(f_threads)\nfn {entry}(\n    {groups}\n) {{\n    {gid}\n{lines}\n}}\n"
    )
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
        "__shared__ __align__(16) ty name[n]",
    ),
    (
        "f_atomic_group(name, n)",
        "threadgroup atomic_uint name[n]",
        "__shared__ __align__(16) u32 name[n]",
    ),
    (
        "f_atomic_add(a, i, v)",
        "atomic_fetch_add_explicit(&(a)[i], v, memory_order_relaxed)",
        "atomicAdd(&(a)[i], v)",
    ),
    (
        "f_atomic_load(a, i)",
        "atomic_load_explicit(&(a)[i], memory_order_relaxed)",
        "((a)[i])",
    ),
    (
        "f_atomic_store(a, i, v)",
        "atomic_store_explicit(&(a)[i], v, memory_order_relaxed)",
        "((a)[i] = (v))",
    ),
    (
        "f_u32x2(p)",
        "(*(device const uint2 *)(p))",
        "(*(const uint2 *)(p))",
    ),
    (
        "f_u32(p)",
        "(*(device const u32 *)(p))",
        "(*(const u32 *)(p))",
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

/// `name` rounds to nearest even, `e` and `m` bits.
fn rounder(name: &str, [e, m]: [u32; 2], top: f32, [from, to]: [&str; 2]) -> String {
    let steps = (125 + (1u32 << (e - 1)) + m) << 23;
    let least = (129 - (1u32 << (e - 1))) << 23;
    let (keep, low) = ((1u32 << (22 - m)) - 1, (1u32 << (23 - m)) - 1);
    format!(
        "f_fn float {name}(float x) {{
    float a = x < 0.0f ? -x : x;
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float steps = {from}({steps}u);
    float small = (a * steps + 8388608.0f) - 8388608.0f;
    u32 bits = {to}(a);
    u32 nudge = {keep}u + ((bits >> {}u) & 1u);
    float r = {from}((bits + nudge) & ~{low}u);
    float big = r > {top:?}f ? {top:?}f : r;
    return sign * (a < {from}({least}u) ? small / steps : big);
}}
",
        23 - m
    )
}

/// Metal branchless E4M3 decode via native `half`.
const METAL_E4M3: &str = "f_fn float f_e4m3_bits(u32 x) {
    u32 s = (x & 0x80u) << 8;
    u32 m = x & 0x7fu;
    half norm = as_type<half>((u16)(s | ((m << 7) + 0x2000u)));
    half sub = as_type<half>((u16)(s | 0x1800u)) * (half)m;
    return (float)select(norm, sub, m < 8u);
}
#define f_e4m3(w, l) f_e4m3_bits(f_field(w, l, 8))
";

/// Metal 16-entry constant table for E2M1 nibbles.
const METAL_E2M1: &str = "constant float f_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
#define f_e2m1(w, l) f_e2m1_lut[f_field(w, l, 4)]
";

/// Metal native `half` cast for `round_f16`.
const METAL_ROUND_F16: &str = "f_fn float f_rounded_f16(float x) {
    return (float)(half)clamp(x, -65504.0f, 65504.0f);
}
#define f_round_f16(x) f_rounded_f16(x)
";

/// `f_{call}(x)` by soft rounding, or CUDA's own text from `arch`.
fn rounded(call: &str, eb: [u32; 2], top: f32, arch: u32, cuda: &str) -> [String; 2] {
    let name = format!("f_rounded_{call}");
    let metal = rounder(&name, eb, top, ["as_type<float>", "as_type<u32>"]);
    let soft = rounder(&name, eb, top, ["__uint_as_float", "__float_as_uint"]);
    let gated = format!(
        "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= {arch}\n{cuda}#else\n{soft}#endif\n"
    );
    let alias = format!("#define f_round_{call}(x) {name}(x)\n");
    [metal + &alias, hip(&soft, &gated) + &alias]
}

/// `f_pair_bf16` in integer steps, reading float bits with `bits`.
fn pair_bits(bits: &str) -> String {
    format!(
        "f_fn u32 f_pair_bf16(float hi, float lo) {{
    u32 h = {bits}(hi);
    u32 l = {bits}(lo);
    h = (h + 0x7fffu + ((h >> 16) & 1u)) & 0xffff0000u;
    l = (l + 0x7fffu + ((l >> 16) & 1u)) & 0xffff0000u;
    return h | (l >> 16);
}}
"
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

/// Portable float-to-e4m3 byte conversion, matching OCP e4m3.
const SOFT_TO_E4M3: &str = "f_fn u8 f_to_e4m3(float x) {
    u32 bits = f_asbits(x);
    u32 sign = (bits >> 24) & 0x80u;
    u32 mag = bits & 0x7fffffffu;
    if (mag > 0x7f800000u) { return (u8)(sign | 0x7fu); }
    if (mag >= 0x43e00000u) { return (u8)(sign | 0x7eu); }
    if (mag < 0x3c800000u) {
        float a = f_bits(mag);
        float s = (a * 512.0f + 8388608.0f) - 8388608.0f;
        return (u8)(sign | (u32)s);
    }
    u32 nudge = 0x7ffffu + ((mag >> 20) & 1u);
    u32 r = (mag + nudge) & ~0xfffffu;
    if (r >= 0x43e00000u) { return (u8)(sign | 0x7eu); }
    u32 exp = (r >> 23) - 120u;
    u32 man = (r >> 20) & 7u;
    return (u8)(sign | (exp << 3) | man);
}
";

/// Portable float-to-bf16 halfword conversion.
const SOFT_TO_BF16: &str = "f_fn u16 f_to_bf16(float x) {
    u32 b = f_asbits(x);
    return (u16)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
";

/// Metal branchless float-to-e4m3 byte conversion, matching OCP e4m3.
const METAL_TO_E4M3: &str = "f_fn u8 f_to_e4m3(float x) {
    u32 bits = f_asbits(x);
    u32 sign = (bits >> 24) & 0x80u;
    u32 mag = bits & 0x7fffffffu;
    u32 norm = (mag + 0xc407ffffu + ((mag >> 20) & 1u)) >> 20;
    u32 sub = f_asbits(fma(f_bits(mag), 512.0f, 8388608.0f)) - 0x4b000000u;
    u32 code = select(norm, sub, mag < 0x3c800000u);
    u32 over = select(0x7eu, 0x7fu, mag > 0x7f800000u);
    code = select(code, over, mag >= 0x43e00000u);
    return (u8)(sign | code);
}
";

/// Store helpers for bf16 and fp8 buffers.
fn narrow_stores(metal: bool) -> String {
    let (addr, make_u2) = match metal {
        true => ("device ", "uint2"),
        false => ("", "make_uint2"),
    };
    let store4_bf16 = format!(
        "f_fn void f_store4_bf16({addr}u16 *p, float4 v) {{
    *({addr}uint2 *)p = {make_u2}(f_bf16_pair(v.y, v.x), f_bf16_pair(v.w, v.z));
}}
"
    );
    let soft_store4_e4m3 = format!(
        "f_fn void f_store4_e4m3({addr}u8 *p, float4 v) {{
    *({addr}u32 *)p = (u32)f_to_e4m3(v.x) | ((u32)f_to_e4m3(v.y) << 8)
        | ((u32)f_to_e4m3(v.z) << 16) | ((u32)f_to_e4m3(v.w) << 24);
}}
"
    );
    if metal {
        let metal_bf16 = format!(
            "#if __METAL_VERSION__ >= 310
f_fn u16 f_to_bf16(float x) {{
    return as_type<u16>((bfloat)x);
}}
f_fn void f_store4_bf16(device u16 *p, float4 v) {{
    *(device bfloat4 *)p = bfloat4(v);
}}
#else
{SOFT_TO_BF16}{store4_bf16}#endif
"
        );
        return format!("{metal_bf16}{METAL_TO_E4M3}{soft_store4_e4m3}");
    }
    let cuda_bf16 = format!(
        "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
f_fn u16 f_to_bf16(float x) {{
    u32 d;
    asm(\"cvt.rn.bf16x2.f32 %0, %1, %2;\" : \"=r\"(d) : \"f\"(0.0f), \"f\"(x));
    return (u16)d;
}}
#else
{SOFT_TO_BF16}#endif
"
    );
    let cuda_e4m3 = format!(
        "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
f_fn u8 f_to_e4m3(float x) {{
    unsigned short e;
    asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\" : \"=h\"(e) : \"f\"(0.0f), \"f\"(x));
    return (u8)(e & 0xffu);
}}
f_fn void f_store4_e4m3(u8 *p, float4 v) {{
    unsigned short lo, hi;
    asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\" : \"=h\"(lo) : \"f\"(v.y), \"f\"(v.x));
    asm(\"cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\" : \"=h\"(hi) : \"f\"(v.w), \"f\"(v.z));
    *(u32 *)p = (u32)lo | ((u32)hi << 16);
}}
#else
{SOFT_TO_E4M3}{soft_store4_e4m3}#endif
"
    );
    let bf16 = hip(SOFT_TO_BF16, &cuda_bf16);
    let e4m3 = hip(&format!("{SOFT_TO_E4M3}{soft_store4_e4m3}"), &cuda_e4m3);
    format!("{bf16}{e4m3}{store4_bf16}")
}

/// The text before every kernel in `dialect`.
fn head(dialect: Dialect) -> String {
    let metal = dialect == Dialect::Metal;
    let mut text = String::new();
    if metal {
        text.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
        // Sums keep the printed order under fast math.
        text.push_str("#pragma clang fp reassociate(off)\n");
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
    text.push_str(&narrow_stores(metal));
    if !metal {
        text.push_str(CUDA_PDL);
    }
    text.push('\n');
    text
}

impl Text {
    pub fn render(&self, dialect: Dialect) -> String {
        self.bounded(dialect, None)
    }

    /// CUDA and HIP compile for at most `group` threads.
    pub fn bounded(&self, dialect: Dialect, group: Option<usize>) -> String {
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
                    "kernel void".into(),
                    "",
                )
            }
            false => {
                let index = "    f_pdl_wait();\n    f_pdl_go();\n    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;\n    u32 lid = threadIdx.x;\n";
                let head = format!("{}{}", head(dialect), crate::tile::cuda());
                let kind = match group {
                    Some(n) => format!("extern \"C\" __global__ void __launch_bounds__({n})"),
                    None => "extern \"C\" __global__ void".into(),
                };
                (head, kind, index)
            }
        };
        let (entry, params, body) = (&self.entry, params.join(",\n    "), &self.body);
        let headers = self.headers.concat();
        format!("{head}{headers}{kind} {entry}(\n    {params}\n) {{\n{index}{body}}}\n")
    }
}

/// FNV-1a, so equal bodies can share one entry name.
pub fn hash(text: &str) -> u64 {
    let mix = |h: u64, b: u8| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    text.bytes().fold(0xcbf2_9ce4_8422_2325, mix)
}
