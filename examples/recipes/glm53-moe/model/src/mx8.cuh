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
union alignas(16) Pf1Smem { struct { u32 btab[2][8u * MX8_TILE], wtiles[2][2048]; } k; float out[32][128]; };
struct alignas(16) Pf1TurnSmem { u32 btab_g[8u * MX8_TILE], btab_u[8u * MX8_TILE]; };
union alignas(16) Pf2Smem { struct { u32 btab[2][8u * MX8_TILE], wtiles[2048]; } k; float out[32][128]; };

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

template <tessera::u32 NT>
static __device__ __forceinline__ void pf_cp_one(
    const u32* src, u32* dst, u32 tid, u32 nv4) {
  const uint4* s4 = reinterpret_cast<const uint4*>(src);
  uint4* d4 = reinterpret_cast<uint4*>(dst);
#pragma unroll
  for (u32 i = tid; i < NT * MX8_V4; i += 256u) {
    if (i < nv4) pf_cp_async16(d4 + i, s4 + i);
  }
  asm volatile("cp.async.commit_group;");
}
#else
union alignas(16) Pf1Smem { struct { u32 btab[2][8u * MX8_TILE], wtiles[1][2048]; } k; float out[32][128]; };
union alignas(16) Pf2Smem { struct { u32 btab[1][8u * MX8_TILE], wtiles[2048]; } k; float out[32][128]; };
#endif
constexpr u32 ACT_CHUNK = 8u * MX8_TILE;

template <u32 SPAN>
static __device__ __forceinline__ void pf_scatter_acc(
    const float acc[][4], u32 t0, float sm_out[][128], u32 n, u32 col, u32 rc) {
  prim::block_sync();
#pragma unroll
  for (u32 ti = 0u; ti < SPAN; ti++) {
    u32 r = ((t0 + ti) << 3u) + rc, lr = (ti << 3u) + rc;
    if (r < n) { sm_out[lr][col] = acc[t0 + ti][0]; sm_out[lr][col + 8u] = acc[t0 + ti][2]; }
    if (r + 1u < n) { sm_out[lr + 1u][col] = acc[t0 + ti][1]; sm_out[lr + 1u][col + 8u] = acc[t0 + ti][3]; }
  }
  prim::block_sync();
}

// Transposes MMA accumulators via smem into per-warp row vectors.
template <u32 NT>
static __device__ __forceinline__ void transpose_acc(
    const float acc[NT][4], float sm_out[32][128], float dst[NT][4], u32 n) {
  u32 tid = threadIdx.x, j = tid >> 5u, lane = tid & 31u;
  u32 col = (j << 4u) + (lane >> 2u), rc = (lane & 3u) << 1u;
#pragma unroll
  for (u32 t0 = 0u; t0 < NT; t0 += 4u) {
    constexpr u32 SPAN = (NT < 4u) ? NT : 4u;
    pf_scatter_acc<SPAN>(acc, t0, sm_out, n, col, rc);
#pragma unroll
    for (u32 ti = 0u; ti < SPAN; ti++) {
      u32 r = ((t0 + ti) << 3u) + j;
      if (r < n) trellis::load4(&sm_out[(ti << 3u) + j][lane << 2u], dst[t0 + ti]);
    }
  }
}

template <u32 NT>
static __device__ __forceinline__ void pf_stage_fc2_act(
    const u32* act_src, u32* btab, u32 tid) {
  const uint4* bsrc4 = reinterpret_cast<const uint4*>(act_src);
  uint4* bdst4 = reinterpret_cast<uint4*>(btab);
#if GLM53_USE_MX8
  for (u32 i = tid; i < NT * MX8_V4; i += 256u) pf_cp_async16(bdst4 + i, bsrc4 + i);
  asm volatile("cp.async.commit_group;");
#else
  for (u32 i = tid; i < NT * (MX8_TILE >> 2u); i += 256u) bdst4[i] = bsrc4[i];
#endif
}

// Applies output turn and writes coalesced fp16 partials per pick.
template <u32 NT, typename TokFn, typename PickFn, typename WeightFn>
static __device__ __forceinline__ void pf_store_fc2(
    const float acc[NT][4], Pf2Smem& sm, const float* sv_p, const float* sg_p, u16* z,
    u32 ob, u32 hidden, u32 picks, u32 n, TokFn get_tok, PickFn get_pick, WeightFn get_w) {
  u32 tid = threadIdx.x, j = tid >> 5u, lane = tid & 31u;
  u32 col = (j << 4u) + (lane >> 2u), rc = (lane & 3u) << 1u, base = (ob << 7u) + (lane << 2u);
  float4 sv = *reinterpret_cast<const float4*>(sv_p + base);
  float4 sg = sg_p ? *reinterpret_cast<const float4*>(sg_p + base) : make_float4(1.f, 1.f, 1.f, 1.f);
  float sc[4] = {sv.x * sg.x, sv.y * sg.y, sv.z * sg.z, sv.w * sg.w};
#pragma unroll
  for (u32 t0 = 0u; t0 < NT; t0 += 4u) {
    constexpr u32 SPAN = (NT < 4u) ? NT : 4u;
    pf_scatter_acc<SPAN>(acc, t0, sm.out, n, col, rc);
#pragma unroll
    for (u32 ti = 0u; ti < SPAN; ti++) {
      u32 r = ((t0 + ti) << 3u) + j;
      if (r < n) {
        float v[4];
        trellis::load4(&sm.out[(ti << 3u) + j][lane << 2u], v);
        trellis::wht128(v, trellis::INV_SQRT128);
        u32 tok = get_tok(r), pk = get_pick(r);
        float gw = get_w(tok, pk);
        v[0] *= sc[0] * gw; v[1] *= sc[1] * gw; v[2] *= sc[2] * gw; v[3] *= sc[3] * gw;
        u32 h0 = trellis::f32x2_to_f16x2(v[0], v[1]), h1 = trellis::f32x2_to_f16x2(v[2], v[3]);
        *reinterpret_cast<uint2*>(z + ((u64)tok * (picks + 1u) + pk) * hidden + base) = make_uint2(h0, h1);
      }
    }
  }
}

// Runs pipelined MXFP8 fc2 chunk loop across all ffn blocks.
template <u32 K, u32 BOOK, u32 NT>
static __device__ __forceinline__ void pf_fc2_loop(
    const u32* act_slot, const u8* gb, u64 sG, bool aligned, u32 upto, u32 tid, u32 j, u32 lane,
    const trellis::word_map& wm, Pf2Smem& sm, float acc[NT][4]) {
  u32 nw[8];
  trellis::blk_load<K>(gb, tid, aligned, nw);
#if GLM53_USE_MX8
  pf_stage_fc2_act<NT>(act_slot, sm.k.btab[0], tid);
  for (u32 c = 0u; c < upto; c++) {
    u32 st = c & 1u;
    if (c + 1u < upto) pf_stage_fc2_act<NT>(act_slot + (u64)(c + 1u) * ACT_CHUNK, sm.k.btab[st ^ 1u], tid);
    trellis::blk_store<K>(nw, tid, sm.k.wtiles);
    if (c + 1u < upto) asm volatile("cp.async.wait_group 1;"); else asm volatile("cp.async.wait_group 0;");
    prim::block_sync();
    if (c + 1u < upto) { gb += sG; trellis::blk_load<K>(gb, tid, aligned, nw); }
    pf_blk_mma<K, BOOK, NT>(sm.k.wtiles, sm.k.btab[st], j, lane, wm, acc);
    if (c + 1u < upto) prim::block_sync();
  }
#else
  for (u32 c = 0u; c < upto; c++) {
    if (c > 0u) prim::block_sync();
    pf_stage_fc2_act<NT>(act_slot + (u64)c * ACT_CHUNK, sm.k.btab[0], tid);
    trellis::blk_store<K>(nw, tid, sm.k.wtiles);
    prim::block_sync();
    if (c + 1u < upto) { gb += sG; trellis::blk_load<K>(gb, tid, aligned, nw); }
    pf_blk_mma<K, BOOK, NT>(sm.k.wtiles, sm.k.btab[0], j, lane, wm, acc);
  }
#endif
}

// Sums weighted fp16 partials in pick order per warp.
template <typename ValidFn>
static __device__ __forceinline__ void pf_finish_row(
    const u16* z, float* y, u32 row, u32 base, u32 hidden, u32 picks, ValidFn has_pick) {
  const u16* rb = z + (u64)row * (picks + 1u) * hidden + base;
  float acc[4] = {};
  for (u32 p = 0u; p <= picks; p++) {
    if (p < picks && !has_pick(p)) continue;
    uint2 h = *reinterpret_cast<const uint2*>(rb + (u64)p * hidden);
    acc[0] += trellis::f16_to_f32(h.x & 0xFFFFu); acc[1] += trellis::f16_to_f32(h.x >> 16u);
    acc[2] += trellis::f16_to_f32(h.y & 0xFFFFu); acc[3] += trellis::f16_to_f32(h.y >> 16u);
  }
  *reinterpret_cast<float4*>(y + (u64)row * hidden + base) = make_float4(acc[0], acc[1], acc[2], acc[3]);
}
