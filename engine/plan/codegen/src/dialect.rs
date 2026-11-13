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
    pub body: String,
}

/// Each dialect's spelling of the names a body may use.
const METAL: &str = "#include <metal_stdlib>
using namespace metal;
typedef unsigned int u32;
typedef unsigned long u64;
typedef unsigned short u16;
#define f_exp(x) exp(x)
#define f_tanh(x) tanh(x)
#define f_max(a, b) max(a, b)
#define f_clamp(x, lo, hi) clamp(x, lo, hi)
#define f_pow(a, b) precise::pow(a, b)
#define f_cos(x) precise::cos(x)
#define f_sin(x) precise::sin(x)
#define f_bits(x) as_type<float>(x)
#define f_rsqrt(x) rsqrt(x)
#define f_shfl_xor(v, m) simd_shuffle_xor(v, m)
#define f_barrier() threadgroup_barrier(mem_flags::mem_threadgroup)
#define f_group(ty, name, n) threadgroup ty name[n]
#define f_f32x4(p) (*(device float4 *)(p))
#define f_u32x2(p) (*(device const uint2 *)(p))

";

const CUDA: &str = "typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
#define f_exp(x) expf(x)
#define f_tanh(x) tanhf(x)
#define f_max(a, b) fmaxf(a, b)
#define f_clamp(x, lo, hi) fminf(fmaxf(x, lo), hi)
#define f_pow(a, b) powf(a, b)
#define f_cos(x) cosf(x)
#define f_sin(x) sinf(x)
#define f_bits(x) __uint_as_float(x)
#define f_rsqrt(x) rsqrtf(x)
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#define f_shfl_xor(v, m) __shfl_xor(v, m)
#else
#define f_shfl_xor(v, m) __shfl_xor_sync(0xffffffffu, v, m)
#endif
#define f_barrier() __syncthreads()
#define f_group(ty, name, n) __shared__ ty name[n]
#define f_f32x4(p) (*(float4 *)(p))
#define f_u32x2(p) (*(const uint2 *)(p))

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
        // Bodies get `gid` and `lid`, the index in the group.
        let (head, kind, index) = match metal {
            true => {
                params.push("u32 gid [[thread_position_in_grid]]".into());
                params.push("u32 lid [[thread_index_in_threadgroup]]".into());
                (format!("{METAL}{}", crate::tile::METAL), "kernel void", "")
            }
            false => {
                let index = "    u32 gid = blockIdx.x * blockDim.x + threadIdx.x;\n    u32 lid = threadIdx.x;\n";
                let head = format!("{CUDA}{}", crate::tile::cuda());
                (head, "extern \"C\" __global__ void", index)
            }
        };
        let (entry, params, body) = (&self.entry, params.join(",\n    "), &self.body);
        format!("{head}{kind} {entry}(\n    {params}\n) {{\n{index}{body}}}\n")
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
