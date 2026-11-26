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
constexpr u32 MX8_TILE = 272u;

static __device__ __forceinline__ u32 mx_scale(float amax, float& inv) {
  u32 u = __float_as_uint(__fmul_rn(amax, 0.002232142857f));
  int e = int(u >> 23u) - 127 + ((u & 0x7FFFFFu) != 0u ? 1 : 0);
  if (amax == 0.0f) e = 0;
  if (e < -126) e = -126;
  if (e > 127) e = 127;
  inv = __uint_as_float(u32(127 - e) << 23u);
  return u32(e + 127);
}

static __device__ __forceinline__ u16 f2_e4m3x2(float a, float b) {
  u16 r;
  asm("{.reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, %2, %1; mov.b16 %0, t;}"
      : "=h"(r) : "f"(a), "f"(b));
  return r;
}

static __device__ __forceinline__ u32 h2_e4m3x2(u32 h2) {
  u32 r;
  asm("{.reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.u32.u16 %0, t;}"
      : "=r"(r) : "r"(h2));
  return r;
}

static __device__ __forceinline__ void mma_mx8(
    float d[4], const u32 a[4], const u32 b[2], u32 sa, u32 sb) {
  const u16 z = 0u;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
}

template <u32 BOOK>
static __device__ __forceinline__ u32 decode_pair_e4m3(u32 a, u32 b) {
  if constexpr (BOOK == trellis::BOOK_MUL1) {
    u32 sx, sy, v;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(sx) : "r"(a * trellis::MUL1), "r"(0x01010101u), "r"(0x6400u));
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(sy) : "r"(b * trellis::MUL1), "r"(0x01010101u), "r"(0x6400u));
    asm("fma.rn.f16x2 %0, %1, %2, %3;"
        : "=r"(v) : "r"(trellis::prmt(sx, sy, 0x5410u)), "r"(0x1EEE1EEEu), "r"(0xC931C931u));
    return h2_e4m3x2(v);
  } else {
    u32 x = a * trellis::MCG, y = b * trellis::MCG, sum;
    asm("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(x) : "r"(x), "r"(0x8FFF8FFFu), "r"(0x3B603B60u));
    asm("lop3.b32 %0, %1, %2, %3, 0x6A;" : "=r"(y) : "r"(y), "r"(0x8FFF8FFFu), "r"(0x3B603B60u));
    asm("{.reg .b16 xl, xh, yl, yh; .reg .b32 xhh, yll, yhh;\n"
        " mov.b32 {xl, xh}, %1; mov.b32 {yl, yh}, %2;\n"
        " mov.b32 xhh, {xh, xh}; mov.b32 yll, {yl, yl}; mov.b32 yhh, {yh, yh};\n"
        " fma.rn.f16x2 %0, xhh, %3, %1;\n"
        " fma.rn.f16x2 %0, yll, %4, %0;\n"
        " fma.rn.f16x2 %0, yhh, %4, %0;}"
        : "=r"(sum) : "r"(x), "r"(y), "r"(0xBC003C00u), "r"(0x3C000000u));
    return h2_e4m3x2(sum);
  }
}

template <u32 K, u32 BOOK>
static __device__ __forceinline__ void decode_word_e4m3(
    u32 raw, const trellis::word_map& wm, u32& r_lo, u32& r_hi) {
  u32 a = __shfl_sync(0xFFFFFFFFu, raw, wm.i0);
  u32 b = (K == 4u) ? raw : __shfl_sync(0xFFFFFFFFu, raw, wm.i1);
  u32 s0 = (K == 4u) ? 20u : wm.sh;
  u32 hi = trellis::shf_l(b, a, s0), lo = b << s0, st[8];
#pragma unroll
  for (u32 s = 0u; s < 8u; s++) st[s] = trellis::shf_l(lo, hi, K * s) >> 16u;
  r_lo = decode_pair_e4m3<BOOK>(st[0], st[1]) | (decode_pair_e4m3<BOOK>(st[2], st[3]) << 16u);
  r_hi = decode_pair_e4m3<BOOK>(st[4], st[5]) | (decode_pair_e4m3<BOOK>(st[6], st[7]) << 16u);
}

// Quantizes one rotated 128-vector into permuted-K MXFP8 B-fragments.
static __device__ __forceinline__ void store_bfrag_mx8(
    u32* tile, u32 row, u32 lane, const float v[4]) {
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
  for (u32 off = 4u; off > 0u; off >>= 1u) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off, 8));
  }
  float inv;
  u32 sb = mx_scale(amax, inv);
  u16 q0 = f2_e4m3x2(v[0] * inv, v[1] * inv);
  u16 q1 = f2_e4m3x2(v[2] * inv, v[3] * inv);
  u32 ks32 = lane >> 3u, sub = lane & 7u;
  u32 reg = sub >> 2u, hi_half = (sub >> 1u) & 1u, t0 = (sub & 1u) << 1u;
  u16* p = reinterpret_cast<u16*>(tile + ((ks32 * 32u + row * 4u + t0) * 2u + reg));
  p[hi_half] = q0;
  p[4u + hi_half] = q1;
  if (sub == 0u) reinterpret_cast<u8*>(tile + 256u)[ks32 * 8u + row] = u8(sb);
}

// Decodes one staged 64-tile block with Blackwell m16n8k32 MXFP8 MMA.
template <u32 K, u32 BOOK, u32 NT>
static __device__ __forceinline__ void blk_mma_mx8(
    const u32* wtiles, const u32* btab, u32 j, u32 lane, const trellis::word_map& wm,
    float acc[NT][4]) {
  constexpr u32 TW = 8u * K;
  u32 g = lane >> 2u;
#pragma unroll
  for (u32 p = 0u; p < 4u; p++) {
    u32 u0 = p << 1u, u1 = u0 + 1u;
    u32 raw0 = (K == 4u || lane < TW) ? wtiles[(u0 * 8u + j) * TW + lane] : 0u;
    u32 raw1 = (K == 4u || lane < TW) ? wtiles[(u1 * 8u + j) * TW + lane] : 0u;
    u32 fa[4];
    decode_word_e4m3<K, BOOK>(raw0, wm, fa[0], fa[1]);
    decode_word_e4m3<K, BOOK>(raw1, wm, fa[2], fa[3]);
#pragma unroll
    for (u32 ti = 0u; ti < NT; ti++) {
      const u32* tile = btab + ti * MX8_TILE;
      uint2 b2 = *reinterpret_cast<const uint2*>(tile + p * 64u + (lane << 1u));
      u32 fb[2] = {b2.x, b2.y};
      u32 sb = reinterpret_cast<const u8*>(tile + 256u)[p * 8u + g];
      mma_mx8(acc[ti], fa, fb, 127u, sb);
    }
  }
}

static __device__ __forceinline__ void pf_store_bfrag(
    u32* btab, u32 ti, u32 row, u32 lane, bool valid, const float v[4]) {
  (void)valid;
  store_bfrag_mx8(btab + ti * MX8_TILE, row, lane, v);
}

template <u32 K, u32 BOOK, u32 NT>
static __device__ __forceinline__ void pf_blk_mma(
    const u32* wtiles, const u32* btab, u32 j, u32 lane, const trellis::word_map& wm,
    float acc[NT][4]) {
  blk_mma_mx8<K, BOOK, NT>(wtiles, btab, j, lane, wm, acc);
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
