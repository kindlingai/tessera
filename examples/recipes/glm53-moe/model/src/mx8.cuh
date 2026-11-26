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

union alignas(16) Pf1Smem { struct { u32 btab_g[8u * MX8_TILE], btab_u[8u * MX8_TILE], wtiles[2048]; } k; float out[64][128]; };
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
