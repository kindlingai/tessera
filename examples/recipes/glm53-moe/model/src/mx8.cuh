// Blackwell block-scaled FP8 MMA helpers for MoE prefill.
using tessera::u8; using tessera::u16; using tessera::u32; using tessera::u64;
namespace prim = tessera::prim;
namespace trellis = tessera::trellis;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define GLM53_USE_MX8 1
#else
#define GLM53_USE_MX8 0
#endif

#if GLM53_USE_MX8
constexpr u32 MX8_TILE = trellis::MX8_TILE;

static __device__ __forceinline__ void pf_store_bfrag(
    u32* btab, u32 ti, u32 row, u32 lane, bool valid, const float v[4]) {
  (void)valid;
  trellis::store_bfrag_mx8(btab + ti * MX8_TILE, row, lane, v);
}

template <u32 K, u32 BOOK, u32 NT>
static __device__ __forceinline__ void pf_blk_mma(
    const u32* wtiles, const u32* btab, u32 j, u32 lane, const trellis::word_map& wm,
    float acc[NT][4]) {
  trellis::blk_mma_mx8<K, BOOK, NT>(wtiles, btab, j, lane, wm, acc);
}
#else
constexpr u32 MX8_TILE = 512u;

static __device__ __forceinline__ void pf_store_bfrag(
    u32* btab, u32 ti, u32 row, u32 lane, bool valid, const float v[4]) {
  if (valid) trellis::store_bfrag(btab + ti * 512u, row, lane, v);
}

template <u32 K, u32 BOOK, u32 NT>
static __device__ __forceinline__ void pf_blk_mma(
    const u32* wtiles, const u32* btab, u32 j, u32 lane, const trellis::word_map& wm,
    float acc[NT][4]) {
  trellis::blk_mma<K, BOOK, NT>(wtiles, btab + (lane << 1u), NT > 1u ? 512u : 0u, j, lane, wm, acc);
}
#endif

#if GLM53_USE_MX8
constexpr u32 MX8_V4 = MX8_TILE >> 2u;
static __device__ __forceinline__ void pf_cp_async16(void* dst, const void* src) {
  u32 addr = static_cast<u32>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(src));
}
union alignas(16) Pf1Smem { struct { u32 btab_g[8u * MX8_TILE], btab_u[8u * MX8_TILE], wtiles[2][2048]; } k; float out[64][128]; };
struct alignas(16) Pf1TurnSmem { u32 btab_g[8u * MX8_TILE], btab_u[8u * MX8_TILE]; };

// Rotates and packs gate/up activations for one slot block.
template <typename TokFn>
static __device__ __forceinline__ void pf_turn_slot_blk(
    const float* x, const float* signs, const float* gsu, const float* usu, u32* dst_blk,
    Pf1TurnSmem& sm, u32 c0, u32 hidden, u32 n, TokFn get_tok) {
  u32 tid = threadIdx.x, j = tid >> 5u, lane = tid & 31u, at = (c0 << 7u) + (lane << 2u);
  float4 gsv4 = *(const float4*)(gsu + at), usv4 = *(const float4*)(usu + at);
  float4 sg = signs ? *(const float4*)(signs + at) : make_float4(1.f, 1.f, 1.f, 1.f);
  float scg[4] = {gsv4.x * sg.x, gsv4.y * sg.y, gsv4.z * sg.z, gsv4.w * sg.w};
  float scu[4] = {usv4.x * sg.x, usv4.y * sg.y, usv4.z * sg.z, usv4.w * sg.w};
  u32 nt = (n + 7u) >> 3u;
  for (u32 ti = 0u; ti < nt; ti++) {
    u32 r = (ti << 3u) + j;
    float vg[4] = {}, vu[4] = {};
    if (r < n) {
      u32 tok = get_tok(r);
      float4 xv = *(const float4*)(x + (u64)tok * hidden + at);
      vg[0] = xv.x * scg[0]; vg[1] = xv.y * scg[1]; vg[2] = xv.z * scg[2]; vg[3] = xv.w * scg[3];
      vu[0] = xv.x * scu[0]; vu[1] = xv.y * scu[1]; vu[2] = xv.z * scu[2]; vu[3] = xv.w * scu[3];
      trellis::wht128(vg, trellis::INV_SQRT128); trellis::wht128(vu, trellis::INV_SQRT128);
    }
    pf_store_bfrag(sm.btab_g, ti, j, lane, r < n, vg);
    pf_store_bfrag(sm.btab_u, ti, j, lane, r < n, vu);
  }
  prim::block_sync();
  uint4* dg = reinterpret_cast<uint4*>(dst_blk);
  uint4* du = dg + 8u * MX8_V4;
  const uint4* sg4 = reinterpret_cast<const uint4*>(sm.btab_g);
  const uint4* su4 = reinterpret_cast<const uint4*>(sm.btab_u);
  for (u32 i = tid; i < nt * MX8_V4; i += 256u) { dg[i] = sg4[i]; du[i] = su4[i]; }
}

template <u32 NT>
static __device__ __forceinline__ void pf_load_turned(
    const u32* x_blk, Pf1Smem& sm, u32 tid, u32 nv4) {
  const uint4* xg = reinterpret_cast<const uint4*>(x_blk);
  const uint4* xu = xg + 8u * MX8_V4;
  uint4* dg4 = reinterpret_cast<uint4*>(sm.k.btab_g);
  uint4* du4 = reinterpret_cast<uint4*>(sm.k.btab_u);
#pragma unroll
  for (u32 i = tid; i < NT * MX8_V4; i += 256u) {
    if (i < nv4) { pf_cp_async16(dg4 + i, xg + i); pf_cp_async16(du4 + i, xu + i); }
  }
  asm volatile("cp.async.commit_group;");
}
#else
union alignas(16) Pf1Smem { struct { u32 btab_g[8u * MX8_TILE], btab_u[8u * MX8_TILE], wtiles[1][2048]; } k; float out[64][128]; };
#endif
struct alignas(16) Pf2Smem { u32 btab[8u * MX8_TILE], wtiles[2048]; };

// Transposes MMA accumulators via smem into per-warp row vectors.
template <u32 NT>
static __device__ __forceinline__ void transpose_acc(
    const float acc[NT][4], float sm_out[64][128], float dst[NT][4], u32 n) {
  u32 tid = threadIdx.x, j = tid >> 5u, lane = tid & 31u;
  u32 col = (j << 4u) + (lane >> 2u), rc = (lane & 3u) << 1u;
  prim::block_sync();
#pragma unroll
  for (u32 ti = 0u; ti < NT; ti++) {
    u32 lr = (ti << 3u) + rc;
    if (lr < n) { sm_out[lr][col] = acc[ti][0]; sm_out[lr][col + 8u] = acc[ti][2]; }
    if (lr + 1u < n) { sm_out[lr + 1u][col] = acc[ti][1]; sm_out[lr + 1u][col + 8u] = acc[ti][3]; }
  }
  prim::block_sync();
#pragma unroll
  for (u32 ti = 0u; ti < NT; ti++) {
    u32 r = (ti << 3u) + j;
    if (r < n) trellis::load4(&sm_out[r][lane << 2u], dst[ti]);
  }
}
