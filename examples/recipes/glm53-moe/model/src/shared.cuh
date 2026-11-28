// GLM-5.3 routed block: shared types, sizes and helpers.
using tessera::u8; using tessera::u16; using tessera::u32; using tessera::u64;
namespace prim = tessera::prim; namespace trellis = tessera::trellis;
using trellis::word; typedef const u8* bytes;

#define PICKS 8
#ifndef MAX_ROWS
#define MAX_ROWS 2048
#endif
#ifndef MAX_SLOTS
#define MAX_SLOTS 640
#endif
#define ROW_WORDS (MAX_ROWS / 32u)
#define ROUTE_SCALE 2.5f
#define PLAN_WORDS (1u + MAX_SLOTS * 50u + MAX_ROWS * PICKS * 2u)
#define SCORES_OFF ((PLAN_WORDS + 3u) & ~3u)

// Compact routing plan for one layer launch.
struct Plan {
  u32 n_slots;
  u32 slot_expert[MAX_SLOTS];
  u32 slot_count[MAX_SLOTS];
  u32 slot_tok[MAX_SLOTS][32];
  u8 slot_pick[MAX_SLOTS][64];
  u32 tok_expert[MAX_ROWS][PICKS];
  float tok_w[MAX_ROWS][PICKS];
};
static_assert(sizeof(Plan) == PLAN_WORDS * 4u, "Plan size");

// Last-CTA completion counter, reset to zero by winner.
static __device__ u32 route_done;

static __device__ inline u16 f32_to_f16(float v) { u16 o; asm("{.reg .b16 t; cvt.rn.f16.f32 t, %1; mov.b16 %0, t;}" : "=h"(o) : "f"(v)); return o; }
static __device__ inline float sigmoid(float v) { return 1.f / (1.f + __expf(-v)); }
static __device__ inline float4 sigmoid4(float4 v) { return make_float4(sigmoid(v.x), sigmoid(v.y), sigmoid(v.z), sigmoid(v.w)); }
static __device__ inline u32 f32_to_tf32(float v) { u32 o; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(o) : "f"(v)); return o; }
static __device__ inline void mma_tf32_m16n8k8(float d[4], const u32 a[4], const u32 b[2]) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
static __device__ inline void mma_bf16_m16n8k16(float d[4], const u32 a[4], const u32 b[2]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
static __device__ inline uint4 load_bf16x8(const u16* p, bool aligned) {
  if (aligned) { u32 d[4]; trellis::ld_nc_v4(reinterpret_cast<const u8*>(p), d); return make_uint4(d[0], d[1], d[2], d[3]); }
  return make_uint4(u32(p[0]) | (u32(p[1]) << 16u), u32(p[2]) | (u32(p[3]) << 16u), u32(p[4]) | (u32(p[5]) << 16u), u32(p[6]) | (u32(p[7]) << 16u));
}
struct Mat { bytes base; u32 inp, out; bool input_side; };
static __device__ inline Mat matrix(bytes p) { return {p, word(p, 0), word(p, 4), (word(p, 8) & trellis::SIDE_MASK) == trellis::INPUT_SIDE}; }
