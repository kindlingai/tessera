//! The matrix unit's tile, spelled for each dialect.

/// Metal: 8x8 f32 simdgroup matrices, which multiply in f32.
pub const METAL: &str = "#define TILE_M 8
#define TILE_N 8
#define TILE_K 8
typedef simdgroup_float8x8 tile_acc;
typedef simdgroup_float8x8 tile_a;
typedef simdgroup_float8x8 tile_b;
#define TILE_ZERO(acc) acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f)
#define TILE_LOAD_A(f, p, s) simdgroup_load(f, p, s)
#define TILE_LOAD_B(f, p, s) simdgroup_load(f, p, s, ulong2(0, 0), true)
#define TILE_MMA(c, a, b) simdgroup_multiply_accumulate(c, a, b, c)
#define TILE_COUNT 2
#define TILE_ROW(e) (((lane / 4) & 4) + ((lane / 2) % 4))
#define TILE_COL(e) ((((lane / 4) & 2) * 2) + ((lane % 2) * 2) + (e))
#define TILE_VAL(acc, e) ((acc).thread_elements()[e])

";

/// CUDA: tf32 `mma.sync`, split three ways to keep f32.
const CUDA: &str = "#define TILE_M 16
#define TILE_N 8
#define TILE_K 8
typedef struct { float v[4]; } tile_acc;
typedef struct { u32 hi[4]; u32 lo[4]; } tile_a;
typedef struct { u32 hi[2]; u32 lo[2]; } tile_b;
#define TILE_ZERO(acc) (acc).v[0] = (acc).v[1] = (acc).v[2] = (acc).v[3] = 0.0f
__device__ __forceinline__ void tile_split(float x, u32 &hi, u32 &lo) {
    asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(hi) : \"f\"(x));
    float rest = x - __uint_as_float(hi);
    asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(lo) : \"f\"(rest));
}
__device__ __forceinline__ void tile_load_a(tile_a &f, const float *p, u32 s, u32 lane) {
    u32 g = lane / 4, t = lane % 4;
    tile_split(p[g * s + t], f.hi[0], f.lo[0]);
    tile_split(p[(g + 8) * s + t], f.hi[1], f.lo[1]);
    tile_split(p[g * s + t + 4], f.hi[2], f.lo[2]);
    tile_split(p[(g + 8) * s + t + 4], f.hi[3], f.lo[3]);
}
__device__ __forceinline__ void tile_load_b(tile_b &f, const float *p, u32 s, u32 lane) {
    u32 g = lane / 4, t = lane % 4;
    tile_split(p[g * s + t], f.hi[0], f.lo[0]);
    tile_split(p[g * s + t + 4], f.hi[1], f.lo[1]);
}
__device__ __forceinline__ void tile_one(tile_acc &c, const u32 *a, const u32 *b) {
    asm volatile(\"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 \"
                 \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"
                 : \"+f\"(c.v[0]), \"+f\"(c.v[1]), \"+f\"(c.v[2]), \"+f\"(c.v[3])
                 : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"r\"(b[0]), \"r\"(b[1]));
}
#define TILE_LOAD_A(f, p, s) tile_load_a(f, p, s, lane)
#define TILE_LOAD_B(f, p, s) tile_load_b(f, p, s, lane)
#define TILE_MMA(c, a, b) do { tile_one(c, (a).lo, (b).hi); tile_one(c, (a).hi, (b).lo); tile_one(c, (a).hi, (b).hi); } while (0)
#define TILE_COUNT 4
#define TILE_ROW(e) ((lane / 4) + ((e) / 2) * 8)
#define TILE_COL(e) ((lane % 4) * 2 + ((e) % 2))
#define TILE_VAL(acc, e) ((acc).v[e])
";

/// HIP: f16 wmma, split three ways to keep 22 bits.
const HIP: &str = "#define TILE_M 16
#define TILE_N 16
#define TILE_K 16
typedef float tile_acc __attribute__((ext_vector_type(8)));
typedef _Float16 tile_half __attribute__((ext_vector_type(16)));
typedef struct { tile_half hi; tile_half lo; } tile_a;
typedef tile_a tile_b;
#define TILE_ZERO(acc) acc = (tile_acc){0, 0, 0, 0, 0, 0, 0, 0}
__device__ __forceinline__ void tile_load_row(tile_a &f, const float *p, u32 s, u32 lane) {
    const float *row = p + (lane % 16) * s;
    for (u32 i = 0; i < 16; i++) {
        _Float16 hi = (_Float16)row[i];
        f.hi[i] = hi;
        f.lo[i] = (_Float16)(row[i] - (float)hi);
    }
}
#define TILE_LOAD_A(f, p, s) tile_load_row(f, p, s, lane)
#define TILE_LOAD_B(f, p, s) tile_load_row(f, p, s, lane)
#define TILE_MMA(c, a, b) do { \\
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a).lo, (b).hi, c); \\
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a).hi, (b).lo, c); \\
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a).hi, (b).hi, c); } while (0)
#define TILE_COUNT 8
#define TILE_ROW(e) (2 * (e) + lane / 16)
#define TILE_COL(e) (lane % 16)
#define TILE_VAL(acc, e) ((acc)[e])
";

/// HIP's tile text alone, for compilers other than hiprtc.
pub fn hip() -> &'static str {
    HIP
}

/// The CUDA text, with HIP's tile when hiprtc compiles it.
pub fn cuda() -> String {
    format!("#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)\n{HIP}#else\n{CUDA}#endif\n\n")
}
