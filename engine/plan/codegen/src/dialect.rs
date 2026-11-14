//! Source text for generated kernels, in each backend's dialect.

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
    ("f_exp(x)", "exp(x)", "expf(x)"),
    ("f_tanh(x)", "tanh(x)", "tanhf(x)"),
    ("f_max(a, b)", "max(a, b)", "fmaxf(a, b)"),
    (
        "f_clamp(x, lo, hi)",
        "clamp(x, lo, hi)",
        "fminf(fmaxf(x, lo), hi)",
    ),
    ("f_pow(a, b)", "precise::pow(a, b)", "powf(a, b)"),
    ("f_cos(x)", "precise::cos(x)", "cosf(x)"),
    ("f_sin(x)", "precise::sin(x)", "sinf(x)"),
    ("f_bits(x)", "as_type<float>(x)", "__uint_as_float(x)"),
    ("f_asbits(x)", "as_type<u32>(x)", "__float_as_uint(x)"),
    ("f_rsqrt(x)", "rsqrt(x)", "rsqrtf(x)"),
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
    ("f_f32x4(p)", "(*(device float4 *)(p))", "(*(float4 *)(p))"),
    (
        "f_u32(p)",
        "(*(device const u32 *)(p))",
        "(*(const u32 *)(p))",
    ),
    (
        "f_u16(p)",
        "(*(device const u16 *)(p))",
        "(*(const u16 *)(p))",
    ),
    (
        "f_u32x2(p)",
        "(*(device const uint2 *)(p))",
        "(*(const uint2 *)(p))",
    ),
    (
        "f_make4(a, b, c, d)",
        "float4(a, b, c, d)",
        "make_float4(a, b, c, d)",
    ),
];

/// The shuffle. HIP leaves out the mask.
const METAL_SHFL: &str = "#define f_shfl_xor(v, m) simd_shuffle_xor(v, m)\n";
/// Hopper on: wait for the last kernel, release the next.
const CUDA_PDL: &str = "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define f_pdl_wait() asm volatile(\"griddepcontrol.wait;\" ::: \"memory\")
#define f_pdl_go() asm volatile(\"griddepcontrol.launch_dependents;\")
#else
#define f_pdl_wait()
#define f_pdl_go()
#endif
";

/// Bit fields: one extract instruction on each backend.
const METAL_BITS: &str = "#define f_field(w, l, n) extract_bits((u32)(w), (l), (n))
#define f_sfield(w, l, n) ((float)extract_bits((int)(w), (l), (n)))
";
const CUDA_BITS: &str = "#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#define f_field(w, l, n) (((u32)(w) >> (l)) & ((1u << (n)) - 1u))
#define f_sfield(w, l, n) ((float)(((int)((u32)(w) << (32u - (l) - (n)))) >> (32u - (n))))
#else
f_fn u32 f_field(u32 w, u32 l, u32 n) {
    u32 r;
    asm(\"bfe.u32 %0, %1, %2, %3;\" : \"=r\"(r) : \"r\"(w), \"r\"(l), \"r\"(n));
    return r;
}
f_fn float f_sfield(u32 w, u32 l, u32 n) {
    int r;
    asm(\"bfe.s32 %0, %1, %2, %3;\" : \"=r\"(r) : \"r\"(w), \"r\"(l), \"r\"(n));
    return (float)r;
}
#endif
";

/// Narrow float fields, native where a backend has them.
const METAL_FLOATS: &str = "#define f_f16(w, l) ((float)as_type<half>((u16)f_field(w, l, 16)))
";
const CUDA_FLOATS: &str = "#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#define f_f16(w, l) ((float)__builtin_bit_cast(_Float16, (unsigned short)f_field(w, l, 16)))
#else
f_fn float f_f16(u32 w, u32 l) {
    float r;
    unsigned short h = (unsigned short)f_field(w, l, 16);
    asm(\"{.reg .b16 t; mov.b16 t, %1; cvt.f32.f16 %0, t;}\" : \"=f\"(r) : \"h\"(h));
    return r;
}
#endif
";
const CUDA_E4M3: &str = "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
f_fn float f_e4m3(u32 w, u32 l) {
    unsigned short b = (unsigned short)f_field(w, l, 8);
    u32 h;
    asm(\"cvt.rn.f16x2.e4m3x2 %0, %1;\" : \"=r\"(h) : \"h\"(b));
    return f_f16(h, 0);
}
#else
#define f_e4m3(w, l) f_e4m3_bits(f_field(w, l, 8))
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

const CUDA_SHFL: &str = "#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#define f_shfl_xor(v, m) __shfl_xor(v, m)
#else
#define f_shfl_xor(v, m) __shfl_xor_sync(0xffffffffu, v, m)
#endif
";

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
    text.push_str(if metal { METAL_SHFL } else { CUDA_SHFL });
    text.push_str(if metal { METAL_BITS } else { CUDA_BITS });
    text.push_str(if metal { METAL_FLOATS } else { CUDA_FLOATS });
    text.push_str(&widen("f_e4m3_bits", 4, 3));
    text.push_str(&widen("f_e2m1_bits", 2, 1));
    text.push_str(if metal {
        "#define f_e4m3(w, l) f_e4m3_bits(f_field(w, l, 8))\n"
    } else {
        CUDA_E4M3
    });
    text.push_str("#define f_e2m1(w, l) f_e2m1_bits(f_field(w, l, 4))\n");
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
