// GLM-5.3 routed block: route+plan, fc1, fc2, finish.
#include <tessera/1/core>
#include <tessera/1/trellis>
#include "mx8.cuh"

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

#include "route.cuh"

// Builds compact deduplicated expert slots across all rows.
static __device__ __noinline__ void route_plan(
    const float* scores, const float* bias, float* plan_out, PlanSmem& sm,
    u32 rows, u32 experts, u32 picks, u32 first, u32 count, u32 rks, bool fused, bool have_picks, u32 cap) {
  Plan* plan = reinterpret_cast<Plan*>(plan_out);
  u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
  if (!have_picks) {
    for (u32 r = warp; r < rows; r += 8u) {
      if (experts == 256u && picks == PICKS) route_256(scores, bias, r, rows, rks, fused, sm.ids[warp], sm.gates[warp]);
      else route(scores + (u64)r * experts, bias, experts, picks, rks == 1u, sm.ids[warp], sm.gates[warp]);
      if (lane < picks) {
        u32 e = sm.ids[warp][lane];
        plan->tok_expert[r][lane] = (e >= first && e < first + count) ? e : 0xFFFFFFFFu;
        plan->tok_w[r][lane] = sm.gates[warp][lane];
      }
      __syncwarp();
    }
  }
  if (tid == 0u) sm.total = 0u;
  prim::block_sync();
  u32 rw = (rows + 31u) >> 5u, cap_sh = (cap == 64u) ? 6u : ((cap == 32u) ? 5u : 3u);
  for (u32 base = 0u; base < count; base += 128u) {
    u32 chunk_n = min(128u, count - base), le = tid, cnt = 0u;
    for (u32 idx = tid; idx < rw * 128u; idx += 256u) sm.emask[idx >> 7u][idx & 127u] = 0u;
    prim::block_sync();
    for (u32 i = tid; i < rows * picks; i += 256u) {
      u32 r = i / picks, p = i % picks;
      u32 e = have_picks ? prim::load_fresh(&plan->tok_expert[r][p]) : plan->tok_expert[r][p];
      if (e >= first + base && e < first + base + chunk_n) atomicOr(&sm.emask[r >> 5u][e - first - base], 1u << (r & 31u));
    }
    prim::block_sync();
    if (le < chunk_n) for (u32 w = 0u; w < rw; w++) cnt += __popc(sm.emask[w][le]);
    u32 ns = (cnt + cap - 1u) >> cap_sh, incl = ns;
    for (u32 d = 1u; d < 32u; d <<= 1u) { u32 v = __shfl_up_sync(0xFFFFFFFFu, incl, d); if (lane >= d) incl += v; }
    if (lane == 31u) sm.wsum[warp] = incl;
    prim::block_sync();
    u32 chunk_base = sm.total;
    if (warp == 0u) {
      u32 wv = lane < 8u ? sm.wsum[lane] : 0u, wi = wv;
      for (u32 d = 1u; d < 8u; d <<= 1u) { u32 v = __shfl_up_sync(0xFFFFFFFFu, wi, d); if (lane >= d) wi += v; }
      if (lane < 8u) sm.wsum[lane] = wi - wv;
      if (lane == 7u) sm.total = chunk_base + wi;
    }
    prim::block_sync();
    if (ns > 0u) {
      u32 s0 = chunk_base + sm.wsum[warp] + (incl - ns), sub = 0u, take = 0u, w = 0u, bits = sm.emask[0][le], pair = 0u;
      sm.slot_of[le] = (u16)s0;
      while (w < rw) {
        if (bits == 0u) { if (++w < rw) bits = sm.emask[w][le]; continue; }
        u32 r = (w << 5u) + (u32)(__ffs((int)bits) - 1);
        bits &= bits - 1u;
        if ((take & 1u) == 0u) pair = r; else plan->slot_tok[s0 + sub][take >> 1u] = pair | (r << 16u);
        if (++take == cap) { plan->slot_expert[s0 + sub] = first + base + le; plan->slot_count[s0 + sub] = cap; sub++; take = 0u; }
      }
      if (take > 0u) {
        u32 s = s0 + sub;
        if (take & 1u) plan->slot_tok[s][take >> 1u] = pair;
        plan->slot_expert[s] = first + base + le; plan->slot_count[s] = take;
      }
    }
    prim::block_sync();
    for (u32 i = tid; i < rows * picks; i += 256u) {
      u32 r = i / picks, p = i % picks;
      u32 e = have_picks ? prim::load_fresh(&plan->tok_expert[r][p]) : plan->tok_expert[r][p];
      if (e < first + base || e >= first + base + chunk_n) continue;
      u32 le2 = e - first - base, wi = r >> 5u, rank = 0u;
      for (u32 w = 0u; w < wi; w++) rank += __popc(sm.emask[w][le2]);
      rank += __popc(sm.emask[wi][le2] & ((1u << (r & 31u)) - 1u));
      plan->slot_pick[(u32)sm.slot_of[le2] + (rank >> cap_sh)][rank & (cap - 1u)] = (u8)p;
    }
    prim::block_sync();
  }
  if (tid == 0u) plan->n_slots = sm.total;
}

// Scores via sigmoid(x router), then last CTA builds plan.
extern "C" __global__ __launch_bounds__(256, ROUTE_MIN_BLOCKS) void glm53_route(
    const float* x, const void* router, const float* bias, float* plan_out,
    u32 rows, u32 hidden, u32 experts, u32 bf16, u32 picks, u32 first, u32 count) {
  __shared__ RouteSmem sm;
  __shared__ u32 is_last;
  float* scores = plan_out + SCORES_OFF;
  u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
  bool use_gemv = rows == 1u && bf16 && (experts & 1u) == 0u && (hidden & 7u) == 0u;
  bool use_pf_bf = rows >= 64u && bf16 && experts == 256u && picks == PICKS && (hidden & 63u) == 0u;
  bool use_pf = !use_pf_bf && rows >= 64u && (experts & 63u) == 0u && (hidden & 31u) == 0u;
  bool use_tc = !use_pf_bf && !use_pf && rows > 1u && (experts & 15u) == 0u && (hidden & 255u) == 0u;
  u32 nb = use_pf ? (experts >> 6u) : (experts >> 4u);
  u32 rks = (use_tc || use_pf) ? (gridDim.x / nb) : 1u;
  if (use_gemv) route_gemv(x, reinterpret_cast<const u16*>(router), scores, sm.wsum, hidden, experts);
  else if (use_pf_bf) route_pf_bf16(x, reinterpret_cast<const u16*>(router), bias, plan_out, sm.bf, rows, hidden, picks, first, count);
  else if (use_pf && rows >= ROUTE_PF64_MIN_ROWS) route_pf<64u>(x, router, scores, sm.pf, rows, hidden, experts, bf16, nb, rks);
  else if (use_pf) route_pf<32u>(x, router, scores, sm.pf, rows, hidden, experts, bf16, nb, rks);
  else if (use_tc) route_tc(x, router, scores, sm.tc, rows, hidden, experts, bf16, nb, rks);
  else if (blockIdx.x * 2u < experts) {
    u32 e0 = blockIdx.x * 2u;
    bool has1 = e0 + 1u < experts;
    for (u32 r0 = 0u; r0 < rows; r0 += 8u) {
      u32 nr = min(rows - r0, 8u);
      float s0[8] = {}, s1[8] = {};
      for (u32 i = tid * 4u; i < hidden; i += 1024u) {
        float4 a0 = weights4(router, (u64)e0 * hidden + i, bf16);
        float4 a1 = has1 ? weights4(router, (u64)(e0 + 1u) * hidden + i, bf16) : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (u32 r = 0u; r < 8u; r++) {
          if (r >= nr) break;
          float4 b = *(const float4*)(x + (u64)(r0 + r) * hidden + i);
          s0[r] += a0.x * b.x + a0.y * b.y + a0.z * b.z + a0.w * b.w;
          if (has1) s1[r] += a1.x * b.x + a1.y * b.y + a1.z * b.z + a1.w * b.w;
        }
      }
#pragma unroll
      for (u32 r = 0u; r < 8u; r++) {
        if (r >= nr) break;
        for (u32 off = 16u; off > 0u; off >>= 1u) {
          s0[r] += __shfl_xor_sync(0xFFFFFFFFu, s0[r], off);
          if (has1) s1[r] += __shfl_xor_sync(0xFFFFFFFFu, s1[r], off);
        }
        if (lane == 0u) { sm.wsum[0][r][warp] = s0[r]; if (has1) sm.wsum[1][r][warp] = s1[r]; }
      }
      prim::block_sync();
      if (warp < 2u && (warp == 0u || has1) && lane < nr) {
        float tot = sm.wsum[warp][lane][0] + sm.wsum[warp][lane][1] + sm.wsum[warp][lane][2] + sm.wsum[warp][lane][3] +
                    sm.wsum[warp][lane][4] + sm.wsum[warp][lane][5] + sm.wsum[warp][lane][6] + sm.wsum[warp][lane][7];
        scores[(u64)(r0 + lane) * experts + e0 + warp] = sigmoid(tot);
      }
      if (r0 + 8u < rows) prim::block_sync();
    }
  }
  __threadfence();
  prim::block_sync();
  if (tid == 0u) {
    u32 done = prim::fetch_add_gpu(&route_done, 1u) + 1u;
    is_last = done == gridDim.x * gridDim.y ? 1u : 0u;
    if (is_last) prim::store_relaxed_gpu(&route_done, 0u);
  }
  prim::block_sync();
  if (!is_last) return;
  bool split_k = (use_tc || use_pf) && rks > 1u;
  bool fused = split_k && experts == 256u && picks == PICKS;
  if (split_k && !fused) {
    u64 re = (u64)rows * experts;
    for (u64 idx = tid * 4ull; idx < re; idx += 1024ull) {
      float4 tot = make_float4(0.f, 0.f, 0.f, 0.f);
      for (u32 sp = 0u; sp < rks; sp++) {
        float4 v = prim::load_fresh4(scores + (u64)sp * re + idx);
        tot.x += v.x; tot.y += v.y; tot.z += v.z; tot.w += v.w;
      }
      *reinterpret_cast<float4*>(scores + idx) = sigmoid4(tot);
    }
    prim::block_sync();
  }
  route_plan(scores, bias, plan_out, sm.plan, rows, experts, picks, first, count, rks, fused,
             use_pf_bf, GLM53_SLOT_CAP);
}

struct alignas(16) Fc1Smem { u32 btab[2048], wtiles[2048]; };
struct alignas(16) Fc2Smem { u32 btab[512], wtiles[2048]; };

static __device__ __forceinline__ u32 tok_at(uint4 t, u32 r) {
  u32 w = (r < 2u) ? t.x : (r < 4u) ? t.y : (r < 6u) ? t.z : t.w;
  return (w >> ((r & 1u) << 4u)) & 0xFFFFu;
}

static __device__ __forceinline__ u32 slot_tok_at(const Plan* plan, bool is_sh, u32 r0_sh, u32 rs, u32 r) {
  if (is_sh) return r0_sh + r;
  u32 w = plan->slot_tok[rs][r >> 1u];
  return (w >> ((r & 1u) << 4u)) & 0xFFFFu;
}

#define DISPATCH_KB(HOT, COLD, KVAR, BVAR, ...)                                           \
  do {                                                                                     \
    if ((BVAR) && (KVAR) == 3u) { HOT<3u, trellis::BOOK_MUL1>(__VA_ARGS__); }              \
    else if ((BVAR) && (KVAR) == 4u) { COLD<4u, trellis::BOOK_MUL1>(__VA_ARGS__); }        \
    else if ((BVAR) && (KVAR) == 2u) { COLD<2u, trellis::BOOK_MUL1>(__VA_ARGS__); }        \
    else if ((KVAR) == 3u) { COLD<3u, trellis::BOOK_MCG>(__VA_ARGS__); }                   \
    else if ((KVAR) == 4u) { COLD<4u, trellis::BOOK_MCG>(__VA_ARGS__); }                   \
    else { COLD<2u, trellis::BOOK_MCG>(__VA_ARGS__); }                                     \
  } while (0)

// Hot path for MUL1 K 2 to 4, else cold.
#define DISPATCH_KB_HOT(HOT, COLD, KVAR, BVAR, ...)                                        \
  do {                                                                                     \
    if ((BVAR) && (KVAR) == 2u) { HOT<2u, trellis::BOOK_MUL1>(__VA_ARGS__); }              \
    else if ((BVAR) && (KVAR) == 3u) { HOT<3u, trellis::BOOK_MUL1>(__VA_ARGS__); }         \
    else if ((BVAR) && (KVAR) == 4u) { HOT<4u, trellis::BOOK_MUL1>(__VA_ARGS__); }         \
    else DISPATCH_KB(COLD, COLD, KVAR, BVAR, __VA_ARGS__);                                 \
  } while (0)

// Picks the tile count for NVAR rows.
#define DISPATCH_NT(BODY, NVAR, ...)                                                       \
  do {                                                                                     \
    if ((NVAR) <= 8u) { BODY<K, BOOK, 1u>(__VA_ARGS__); }                                  \
    else if ((NVAR) <= 16u) { BODY<K, BOOK, 2u>(__VA_ARGS__); }                            \
    else if ((NVAR) <= 32u) { BODY<K, BOOK, 4u>(__VA_ARGS__); }                            \
    else { BODY<K, BOOK, 8u>(__VA_ARGS__); }                                               \
  } while (0)

// Noinline wrapper that calls the callee given last.
#define DEFINE_COLD(NAME, PARAMS, CALL, ...)                                               \
  template <u32 K, u32 BOOK> static __device__ __noinline__ void NAME PARAMS { __VA_ARGS__ CALL; }

// Streams one rotated trellis fc1 matrix across token slots.
template <u32 K, u32 BOOK, u32 NT, typename TokFn>
static __device__ __forceinline__ void fc1_stream(
    const float* x, const float* su, const float* signs, bytes wbytes, u32* wtiles, u32* btab,
    u32 m, u32 c, u32 blk0, u32 nblk, u32 hidden, u32 n, TokFn get_tok, float acc[NT][4]) {
  constexpr u32 TB = 32u * K, CHUNK = (NT >= 4u) ? 1u : (4u / NT);
  u32 tid = threadIdx.x, j = tid >> 5u, lane = tid & 31u;
  Mat W = matrix(wbytes);
  u64 sG = (W.input_side ? u64(W.out >> 1u) : 64ull) * u64(TB);
  bytes gb = trellis::tile_ptr<K>(W.base, W.inp, W.out, m, blk0 << 3u, c, W.input_side);
  bool aligned = (reinterpret_cast<u64>(gb) & 15ull) == 0ull;
  u32 nw[8];
  trellis::blk_load<K>(gb, tid, aligned, nw);
  trellis::word_map wm = trellis::map_word<K>(lane);
#pragma unroll
  for (u32 ti = 0u; ti < NT; ti++) acc[ti][0] = acc[ti][1] = acc[ti][2] = acc[ti][3] = 0.f;
  for (u32 c0 = 0u; c0 < nblk; c0 += CHUNK) {
    u32 nb = min(CHUNK, nblk - c0);
    if (c0 > 0u) prim::block_sync();
    u32 wpb = (nb <= 2u) ? (8u / nb) : 2u;
    u32 b = j / wpb, r_base = j - b * wpb;
    if (b < nb && r_base < n) {
      u32 at = ((blk0 + c0 + b) << 7u) + (lane << 2u);
      float4 sv = *(const float4*)(su + at);
      float4 sg = signs ? *(const float4*)(signs + at) : make_float4(1.f, 1.f, 1.f, 1.f);
      float sc[4] = {sv.x * sg.x, sv.y * sg.y, sv.z * sg.z, sv.w * sg.w};
      for (u32 r = r_base; r < n; r += wpb) {
        u32 tok = get_tok(r);
        float4 xv = *(const float4*)(x + (u64)tok * hidden + at);
        float v[4] = {xv.x * sc[0], xv.y * sc[1], xv.z * sc[2], xv.w * sc[3]};
        trellis::wht128(v, trellis::INV_SQRT128);
        trellis::store_bfrag(btab + ((r >> 3u) * nb + b) * 512u, r & 7u, lane, v);
      }
    }
    trellis::blk_stream<K, BOOK, NT>(
        gb, sG, c0, nb, nblk, tid, j, lane, aligned, wm, nw, wtiles, btab + (lane << 1u),
        NT > 1u ? nb * 512u : 0u, acc);
  }
}

// Warp-per-tile split-K fc1 in 16 KiB shared memory.
template <u32 K, u32 BOOK>
static __device__ __forceinline__ void fc1_body(
    const float* x, const float* su, const float* signs, bytes wbytes, float* gpart,
    Fc1Smem& sm, u32 m, u32 c, u32 h, u32 mat, u32 slot, u32 chunks, u32 ks1, u32 max_col,
    u32 hidden, u32 n, uint4 toks) {
  u32 tid = threadIdx.x, j = tid >> 5u, lane = tid & 31u;
  u32 nblk = (hidden >> 7u) / ks1;
  float acc[1][4];
  fc1_stream<K, BOOK, 1u>(x, su, signs, wbytes, sm.wtiles, sm.btab, m, c, h * nblk, nblk, hidden, n,
                          [&](u32 r) { return tok_at(toks, r); }, acc);
  float* gp = gpart + ((((u64)slot * chunks + c) * 2u + mat) * ks1 + h) * max_col * 128u;
  u32 col = (j << 4u) + (lane >> 2u), r2 = (lane & 3u) << 1u;
  if (r2 < n) { gp[(u64)r2 * 128u + col] = acc[0][0]; gp[(u64)r2 * 128u + col + 8u] = acc[0][2]; }
  if (r2 + 1u < n) {
    gp[(u64)(r2 + 1u) * 128u + col] = acc[0][1]; gp[(u64)(r2 + 1u) * 128u + col + 8u] = acc[0][3];
  }
}

DEFINE_COLD(fc1_cold,
    (const float* x, const float* su, const float* signs, bytes wbytes, float* gpart, Fc1Smem& sm, u32 m, u32 c,
     u32 h, u32 mat, u32 slot, u32 chunks, u32 ks1, u32 max_col, u32 hidden, u32 n, uint4 toks),
    (x, su, signs, wbytes, gpart, sm, m, c, h, mat, slot, chunks, ks1, max_col, hidden, n, toks),
    fc1_body<K, BOOK>)

// Split-K gate and up projections for routed and shared slots.
extern "C" __global__ __launch_bounds__(256, FC1_MIN_BLOCKS) void glm53_fc1(
    const float* x, const float* plan_in, const float* signs_in, const float* gate_su, bytes gate,
    const float* up_su, bytes up, bytes shared_gate, const float* shared_gate_su, bytes shared_up,
    const float* shared_up_su, float* gpart, u32 rows, u32 hidden, u32 ffn, u32 picks, u32 k,
    u32 shared_k, u32 book, u32 shared_book, u32 first, u32 shared_ffn, u32 ks1) {
  (void)picks;
  const Plan* plan = reinterpret_cast<const Plan*>(plan_in);
  u32 n_sh = (rows + 7u) >> 3u, slot = blockIdx.y, chunks = ffn >> 7u;
  bool is_shared = slot < n_sh;
  u32 c = blockIdx.x % chunks, hm = blockIdx.x / chunks, h = hm % ks1, mat = hm / ks1;
  if (is_shared ? (c >= (shared_ffn >> 7u)) : (slot - n_sh >= plan->n_slots)) return;
  __shared__ Fc1Smem sm;
  uint4 toks = make_uint4(0u, 0u, 0u, 0u);
  u32 n, m; bytes wbytes; const float *su, *signs;
  if (is_shared) {
    u32 r0 = slot * 8u, tw[4] = {}; n = min(rows - r0, 8u);
    for (u32 r = 0u; r < n; r++) tw[r >> 1u] |= (r0 + r) << ((r & 1u) << 4u);
    toks = make_uint4(tw[0], tw[1], tw[2], tw[3]);
    m = 0u; wbytes = mat ? shared_up : shared_gate; su = mat ? shared_up_su : shared_gate_su; signs = 0; k = shared_k; book = shared_book;
  } else {
    u32 rs = slot - n_sh, e = plan->slot_expert[rs]; n = plan->slot_count[rs]; m = e - first;
    toks = make_uint4(plan->slot_tok[rs][0], plan->slot_tok[rs][1], plan->slot_tok[rs][2], plan->slot_tok[rs][3]);
    wbytes = mat ? up : gate; su = (mat ? up_su : gate_su) + (u64)e * hidden; signs = signs_in;
    if (c == 0u && h == 0u && threadIdx.x == 0u) trellis::note_hit(wbytes, m);
  }
  DISPATCH_KB(fc1_body, fc1_cold, k, book,
      x, su, signs, wbytes, gpart, sm, m, c, h, mat, slot, chunks, ks1, min(rows, 8u), hidden, n, toks);
}

// Warp-per-tile split-N fc2 with double-buffered 128-bit block prefetch.
template <u32 K, u32 BOOK>
static __device__ __forceinline__ void fc2_body(
    const float* gpart, const float* gate_sv, const float* up_sv, const float* down_su,
    bytes dbytes, u16* z, Fc2Smem& sm, u32 m, u32 c, u32 ns, u32 ns2, u32 slot, u32 chunks, u32 ks1,
    u32 max_col, u32 hidden, u32 n, u64 z_off0, u64 z_off1) {
  constexpr u32 TB = 32u * K;
  u32 tid = threadIdx.x, j = tid >> 5u, lane = tid & 31u;
  Mat D = matrix(dbytes);
  u32 obs = hidden >> 7u, ob_per = obs / ns2, ob0 = ns * ob_per;
  u64 sO = (D.input_side ? 64ull : u64(D.inp >> 1u)) * u64(TB);
  bytes gb = trellis::tile_ptr<K>(D.base, D.inp, D.out, m, c << 3u, ob0, D.input_side);
  bool aligned = (reinterpret_cast<u64>(gb) & 15ull) == 0ull;
  u32 nw[8];
  trellis::blk_load<K>(gb, tid, aligned, nw);
  if (j < n) {
    float g[4] = {}, u[4] = {};
    u64 base_gp = ((u64)slot * chunks + c) * 2u * ks1 * max_col * 128u;
    const float* gg = gpart + base_gp + (u64)j * 128u + (lane << 2u);
    const float* gu = gpart + base_gp + (u64)ks1 * max_col * 128u + (u64)j * 128u + (lane << 2u);
    for (u32 h = 0u; h < ks1; h++) {
      float4 gv = trellis::ld_cg_f4(gg + (u64)h * max_col * 128u);
      float4 uv = trellis::ld_cg_f4(gu + (u64)h * max_col * 128u);
      g[0] += gv.x; g[1] += gv.y; g[2] += gv.z; g[3] += gv.w;
      u[0] += uv.x; u[1] += uv.y; u[2] += uv.z; u[3] += uv.w;
    }
    trellis::wht128(g, trellis::INV_SQRT128); trellis::wht128(u, trellis::INV_SQRT128);
    u32 o = c * 128u + (lane << 2u);
    float4 gsv = *(const float4*)(gate_sv + o), usv = *(const float4*)(up_sv + o), dsu = *(const float4*)(down_su + o);
    float gv[4] = {g[0] * gsv.x, g[1] * gsv.y, g[2] * gsv.z, g[3] * gsv.w};
    float uv[4] = {u[0] * usv.x, u[1] * usv.y, u[2] * usv.z, u[3] * usv.w};
    float ds[4] = {dsu.x, dsu.y, dsu.z, dsu.w}, a[4];
#pragma unroll
    for (u32 i = 0u; i < 4u; i++) a[i] = (gv[i] / (1.0f + __expf(-gv[i]))) * uv[i] * ds[i];
    trellis::wht128(a, trellis::INV_SQRT128); trellis::store_bfrag(sm.btab, j, lane, a);
  }
  trellis::word_map wm = trellis::map_word<K>(lane);
  const u32* ha = sm.btab + (lane << 1u);
  u32 nl = lane >> 2u, col2 = (lane & 3u) << 1u;
  for (u32 idx = 0u; idx < ob_per; idx++) {
    if (idx > 0u) prim::block_sync();
    trellis::blk_store<K>(nw, tid, sm.wtiles);
    prim::block_sync();
    if (idx + 1u < ob_per) { gb += sO; trellis::blk_load<K>(gb, tid, aligned, nw); }
    float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
    trellis::blk_mma<K, BOOK, 1u>(sm.wtiles, ha, 0u, j, lane, wm, acc);
    u32 out_base = ((ob0 + idx) << 7u) + (j << 4u) + nl;
  if (col2 < n) { u16* dst = z + z_off0 + out_base; dst[0] = f32_to_f16(acc[0][0]); dst[8] = f32_to_f16(acc[0][2]); }
  if (col2 + 1u < n) { u16* dst = z + z_off1 + out_base; dst[0] = f32_to_f16(acc[0][1]); dst[8] = f32_to_f16(acc[0][3]); }
  }
}

DEFINE_COLD(fc2_cold,
    (const float* gpart, const float* gate_sv, const float* up_sv, const float* down_su, bytes dbytes, u16* z,
     Fc2Smem& sm, u32 m, u32 c, u32 ns, u32 ns2, u32 slot, u32 chunks, u32 ks1, u32 max_col, u32 hidden, u32 n,
     u64 z_off0, u64 z_off1),
    (gpart, gate_sv, up_sv, down_su, dbytes, z, sm, m, c, ns, ns2, slot, chunks, ks1, max_col, hidden, n, z_off0, z_off1),
    fc2_body<K, BOOK>)

// SwiGLU and down projection into fp16 partials.
extern "C" __global__ __launch_bounds__(256, FC2_MIN_BLOCKS) void glm53_fc2(
    const float* plan_in, const float* gpart, const float* gate_sv, const float* up_sv,
    const float* down_su, bytes down, const float* shared_gate_sv, const float* shared_up_sv,
    bytes shared_down, const float* shared_down_su, float* z_out, u32 rows, u32 hidden, u32 ffn,
    u32 picks, u32 k, u32 shared_k, u32 book, u32 shared_book, u32 first, u32 shared_ffn,
    u32 ks1, u32 ns2) {
  const Plan* plan = reinterpret_cast<const Plan*>(plan_in);
  u32 n_sh = (rows + 7u) >> 3u, slot = blockIdx.y, c = blockIdx.x / ns2, ns = blockIdx.x % ns2;
  bool is_shared = slot < n_sh;
  if (is_shared ? (c >= (shared_ffn >> 7u)) : (slot - n_sh >= plan->n_slots)) return;
  u32 lane = threadIdx.x & 31u, col2 = (lane & 3u) << 1u, chunks = ffn >> 7u, n, m;
  const float *gsv, *usv, *dsu; bytes dbytes; u64 z_off0 = 0ull, z_off1 = 0ull;
  if (is_shared) {
    u32 r0 = slot * 8u; n = min(rows - r0, 8u); m = 0u;
    gsv = shared_gate_sv; usv = shared_up_sv; dsu = shared_down_su; dbytes = shared_down; k = shared_k; book = shared_book;
    if (col2 < n) z_off0 = (((u64)(r0 + col2) * (picks + 1u) + picks) * chunks + c) * hidden;
    if (col2 + 1u < n) z_off1 = (((u64)(r0 + col2 + 1u) * (picks + 1u) + picks) * chunks + c) * hidden;
  } else {
    u32 rs = slot - n_sh, e = plan->slot_expert[rs], tp = plan->slot_tok[rs][col2 >> 1u];
    n = plan->slot_count[rs]; m = e - first;
    gsv = gate_sv + (u64)e * ffn; usv = up_sv + (u64)e * ffn; dsu = down_su + (u64)e * ffn; dbytes = down;
    if (c == 0u && ns == 0u && threadIdx.x == 0u) trellis::note_hit(dbytes, m);
    if (col2 < n) z_off0 = (((u64)(tp & 0xFFFFu) * (picks + 1u) + plan->slot_pick[rs][col2]) * chunks + c) * hidden;
    if (col2 + 1u < n) z_off1 = (((u64)(tp >> 16u) * (picks + 1u) + plan->slot_pick[rs][col2 + 1u]) * chunks + c) * hidden;
  }
  __shared__ Fc2Smem sm;
  DISPATCH_KB(fc2_body, fc2_cold, k, book, gpart, gsv, usv, dsu, dbytes, reinterpret_cast<u16*>(z_out),
      sm, m, c, ns, ns2, slot, chunks, ks1, min(rows, 8u), hidden, n, z_off0, z_off1);
}

// Pre-turns routed and shared activations once per slot block.
extern "C" __global__ __launch_bounds__(256, FC1_TURN_MIN_BLOCKS) void glm53_fc1_turn(
    const float* x, const float* plan_in, const float* signs_in, const float* gate_su,
    const float* up_su, const float* sh_gate_su, const float* sh_up_su, float* x_out,
    u32 rows, u32 hidden, u32 first) {
  (void)first;
#if GLM53_USE_MX8
  const Plan* plan = reinterpret_cast<const Plan*>(plan_in);
  u32 n_sh = (rows + 63u) >> 6u, slot = blockIdx.y, c0 = blockIdx.x, nblk = hidden >> 7u;
  bool is_sh = slot < n_sh;
  if (!is_sh && slot - n_sh >= plan->n_slots) return;
  __shared__ Pf1TurnSmem sm;
  u32 r0_sh = slot * 64u, rs = is_sh ? 0u : (slot - n_sh), n;
  const float *gsu, *usu, *signs;
  if (is_sh) { n = min(rows - r0_sh, 64u); gsu = sh_gate_su; usu = sh_up_su; signs = 0; }
  else { u32 e = plan->slot_expert[rs]; n = plan->slot_count[rs]; gsu = gate_su + (u64)e * hidden; usu = up_su + (u64)e * hidden; signs = signs_in; }
  u32* dst_blk = reinterpret_cast<u32*>(x_out) + ((u64)slot * nblk + c0) * (16u * MX8_TILE);
  pf_turn_slot_blk(x, signs, gsu, usu, dst_blk, sm, c0, hidden, n,
                   [&](u32 r) { return slot_tok_at(plan, is_sh, r0_sh, rs, r); });
#else
  (void)x; (void)plan_in; (void)signs_in; (void)gate_su; (void)up_su; (void)sh_gate_su; (void)sh_up_su; (void)x_out; (void)rows; (void)hidden;
#endif
}

// Prefill fused gate, up, and SwiGLU into bfrag tiles.
template <u32 K, u32 BOOK, u32 NT>
static __device__ __forceinline__ void fc1_pf_body(
    const float* x, const Plan* plan, const u32* x_slot, const float* signs, const float* gsu, bytes gate,
    const float* gsv, const float* usu, bytes up, const float* usv, const float* dsu,
    u32* act_dst, Pf1Smem& sm, bool is_sh, u32 r0_sh, u32 rs, u32 m, u32 c, u32 hidden, u32 n) {
  constexpr u32 TB = 32u * K;
  u32 tid = threadIdx.x, j = tid >> 5u, lane = tid & 31u, nblk = hidden >> 7u;
  Mat Wg = matrix(gate), Wu = matrix(up);
  u64 sGg = (Wg.input_side ? u64(Wg.out >> 1u) : 64ull) * u64(TB), sGu = (Wu.input_side ? u64(Wu.out >> 1u) : 64ull) * u64(TB);
  bytes gbp = trellis::tile_ptr<K>(Wg.base, Wg.inp, Wg.out, m, 0u, c, Wg.input_side);
  bytes ubp = trellis::tile_ptr<K>(Wu.base, Wu.inp, Wu.out, m, 0u, c, Wu.input_side);
  bool ga = (reinterpret_cast<u64>(gbp) & 15ull) == 0ull, ua = (reinterpret_cast<u64>(ubp) & 15ull) == 0ull;
  u32 nw[8];
  trellis::blk_load<K>(gbp, tid, ga, nw);
  trellis::word_map wm = trellis::map_word<K>(lane);
  float ag[NT][4], au[NT][4];
#pragma unroll
  for (u32 ti = 0u; ti < NT; ti++) { ag[ti][0] = ag[ti][1] = ag[ti][2] = ag[ti][3] = 0.f; au[ti][0] = au[ti][1] = au[ti][2] = au[ti][3] = 0.f; }
#if GLM53_USE_MX8
  (void)x; (void)plan; (void)signs; (void)gsu; (void)usu; (void)is_sh; (void)r0_sh; (void)rs;
  u32 nv4 = ((n + 7u) >> 3u) * MX8_V4;
  pf_cp_one<NT>(x_slot, sm.k.btab[0], tid, nv4);
  for (u32 c0 = 0u; c0 < nblk; c0++) {
    pf_cp_one<NT>(x_slot + (u64)c0 * (16u * MX8_TILE) + 8u * MX8_TILE, sm.k.btab[1], tid, nv4);
    trellis::blk_store<K>(nw, tid, sm.k.wtiles[0]);
    asm volatile("cp.async.wait_group 1;");
    prim::block_sync();
    trellis::blk_load<K>(ubp, tid, ua, nw);
    pf_blk_mma<K, BOOK, NT>(sm.k.wtiles[0], sm.k.btab[0], j, lane, wm, ag);
    trellis::blk_store<K>(nw, tid, sm.k.wtiles[1]);
    asm volatile("cp.async.wait_group 0;");
    prim::block_sync();
    if (c0 + 1u < nblk) {
      pf_cp_one<NT>(x_slot + (u64)(c0 + 1u) * (16u * MX8_TILE), sm.k.btab[0], tid, nv4);
      gbp += sGg; ubp += sGu; trellis::blk_load<K>(gbp, tid, ga, nw);
    }
    pf_blk_mma<K, BOOK, NT>(sm.k.wtiles[1], sm.k.btab[1], j, lane, wm, au);
    if (c0 + 1u < nblk) prim::block_sync();
  }
#else
  (void)x_slot;
  for (u32 c0 = 0u; c0 < nblk; c0++) {
    u32 at = (c0 << 7u) + (lane << 2u);
    float4 gsv4 = *(const float4*)(gsu + at), usv4 = *(const float4*)(usu + at);
    float4 sg = signs ? *(const float4*)(signs + at) : make_float4(1.f, 1.f, 1.f, 1.f);
    float scg[4] = {gsv4.x * sg.x, gsv4.y * sg.y, gsv4.z * sg.z, gsv4.w * sg.w}, scu[4] = {usv4.x * sg.x, usv4.y * sg.y, usv4.z * sg.z, usv4.w * sg.w};
#pragma unroll
    for (u32 ti = 0u; ti < NT; ti++) {
      u32 r = (ti << 3u) + j;
      float vg[4] = {}, vu[4] = {};
      if (r < n) {
        u32 tok = slot_tok_at(plan, is_sh, r0_sh, rs, r);
        float4 xv = *(const float4*)(x + (u64)tok * hidden + at);
        vg[0] = xv.x * scg[0]; vg[1] = xv.y * scg[1]; vg[2] = xv.z * scg[2]; vg[3] = xv.w * scg[3];
        vu[0] = xv.x * scu[0]; vu[1] = xv.y * scu[1]; vu[2] = xv.z * scu[2]; vu[3] = xv.w * scu[3];
        trellis::wht128(vg, trellis::INV_SQRT128); trellis::wht128(vu, trellis::INV_SQRT128);
      }
      pf_store_bfrag(sm.k.btab[0], ti, j, lane, r < n, vg); pf_store_bfrag(sm.k.btab[1], ti, j, lane, r < n, vu);
    }
    trellis::blk_store<K>(nw, tid, sm.k.wtiles[0]); prim::block_sync();
    trellis::blk_load<K>(ubp, tid, ua, nw);
    pf_blk_mma<K, BOOK, NT>(sm.k.wtiles[0], sm.k.btab[0], j, lane, wm, ag); prim::block_sync();
    trellis::blk_store<K>(nw, tid, sm.k.wtiles[0]); prim::block_sync();
    if (c0 + 1u < nblk) { gbp += sGg; ubp += sGu; trellis::blk_load<K>(gbp, tid, ga, nw); }
    pf_blk_mma<K, BOOK, NT>(sm.k.wtiles[0], sm.k.btab[1], j, lane, wm, au);
    if (c0 + 1u < nblk) prim::block_sync();
  }
#endif
  transpose_acc<NT>(ag, sm.out, ag, n);
  transpose_acc<NT>(au, sm.out, au, n);
  prim::block_sync();
  u32 o = c * 128u + (lane << 2u);
  float4 sg = *(const float4*)(gsv + o), su = *(const float4*)(usv + o), sd = *(const float4*)(dsu + o);
#pragma unroll
  for (u32 ti = 0u; ti < NT; ti++) {
    u32 r = (ti << 3u) + j;
    float a[4] = {};
    if (r < n) {
      trellis::wht128(ag[ti], trellis::INV_SQRT128); trellis::wht128(au[ti], trellis::INV_SQRT128);
      float g0 = ag[ti][0] * sg.x, g1 = ag[ti][1] * sg.y, g2 = ag[ti][2] * sg.z, g3 = ag[ti][3] * sg.w;
      a[0] = (g0 / (1.0f + __expf(-g0))) * (au[ti][0] * su.x) * sd.x;
      a[1] = (g1 / (1.0f + __expf(-g1))) * (au[ti][1] * su.y) * sd.y;
      a[2] = (g2 / (1.0f + __expf(-g2))) * (au[ti][2] * su.z) * sd.z;
      a[3] = (g3 / (1.0f + __expf(-g3))) * (au[ti][3] * su.w) * sd.w;
      trellis::wht128(a, trellis::INV_SQRT128);
    }
    pf_store_bfrag(sm.k.btab[0], ti, j, lane, true, a);
  }
  prim::block_sync();
  uint4* dst4 = reinterpret_cast<uint4*>(act_dst);
  const uint4* src4 = reinterpret_cast<const uint4*>(sm.k.btab[0]);
  for (u32 i = tid; i < NT * (MX8_TILE >> 2u); i += 256u) dst4[i] = src4[i];
}

#define FC1_PF_PARAMS (const float* x, const Plan* plan, const u32* xs, const float* signs, const float* gsu, bytes gate, const float* gsv, const float* usu, bytes up, const float* usv, const float* dsu, u32* act_dst, Pf1Smem& sm, bool is_sh, u32 r0_sh, u32 rs, u32 m, u32 c, u32 hidden, u32 n)
#define FC1_PF_ARGS x, plan, xs, signs, gsu, gate, gsv, usu, up, usv, dsu, act_dst, sm, is_sh, r0_sh, rs, m, c, hidden, n
template <u32 K, u32 BOOK>
static __device__ __noinline__ void fc1_pf_hot FC1_PF_PARAMS { DISPATCH_NT(fc1_pf_body, n, FC1_PF_ARGS); }
DEFINE_COLD(fc1_pf_cold, FC1_PF_PARAMS, (FC1_PF_ARGS), fc1_pf_body<K, BOOK, 8u>)
// Prefill fused gate, up and SwiGLU for 64-token slots.
extern "C" __global__ __launch_bounds__(256, FC1_PF_MIN_BLOCKS) void glm53_fc1_pf(
    const float* x, const float* plan_in, const float* x_turned, const float* signs_in, const float* gate_su, bytes gate, const float* gate_sv, const float* up_su, bytes up, const float* up_sv, const float* down_su, bytes sh_gate, const float* sh_gate_su, const float* sh_gate_sv, bytes sh_up, const float* sh_up_su, const float* sh_up_sv, const float* sh_down_su, float* act_out, u32 rows, u32 hidden, u32 ffn, u32 picks, u32 k, u32 shared_k, u32 book, u32 shared_book, u32 first, u32 shared_ffn) {
  (void)picks;
  const Plan* plan = reinterpret_cast<const Plan*>(plan_in);
  u32 n_sh = (rows + 63u) >> 6u, slot = blockIdx.y, c = blockIdx.x, chunks = ffn >> 7u;
  bool is_sh = slot < n_sh;
  if (is_sh ? (c >= (shared_ffn >> 7u)) : (slot - n_sh >= plan->n_slots)) return;
  __shared__ Pf1Smem sm;
  u32 r0_sh = slot * 64u, rs = is_sh ? 0u : (slot - n_sh), n, m;
  bytes gb, ub;
  const float *gsu, *gsv, *usu, *usv, *dsu, *signs;
  if (is_sh) { n = min(rows - r0_sh, 64u); m = 0u; gb = sh_gate; gsu = sh_gate_su; gsv = sh_gate_sv; ub = sh_up; usu = sh_up_su; usv = sh_up_sv; dsu = sh_down_su; signs = 0; k = shared_k; book = shared_book; }
  else { u32 e = plan->slot_expert[rs]; n = plan->slot_count[rs]; m = e - first; gb = gate; gsu = gate_su + (u64)e * hidden; gsv = gate_sv + (u64)e * ffn; ub = up; usu = up_su + (u64)e * hidden; usv = up_sv + (u64)e * ffn; dsu = down_su + (u64)e * ffn; signs = signs_in; if (c == 0u && threadIdx.x == 0u) { trellis::note_hit(gb, m); trellis::note_hit(ub, m); } }
  const u32* xs = reinterpret_cast<const u32*>(x_turned) + (u64)slot * (hidden >> 7u) * (16u * MX8_TILE);
  u32* act_dst = reinterpret_cast<u32*>(act_out) + ((u64)slot * chunks + c) * ACT_CHUNK;
  DISPATCH_KB_HOT(fc1_pf_hot, fc1_pf_cold, k, book, x, plan, xs, signs, gsu, gb, gsv, usu, ub, usv, dsu, act_dst, sm, is_sh, r0_sh, rs, m, c, hidden, n);
}

// Prefill down projection across all ffn chunks per ob.
template <u32 K, u32 BOOK, u32 NT>
static __device__ __forceinline__ void fc2_pf_body(
    const Plan* plan, const u32* act_slot, bytes dbytes, const float* sv_p, const float* sg_p,
    u16* z, Pf2Smem& sm, bool is_sh, u32 r0_sh, u32 rs, u32 m, u32 ob, u32 upto, u32 hidden,
    u32 picks, u32 n) {
  constexpr u32 TB = 32u * K;
  u32 tid = threadIdx.x, j = tid >> 5u, lane = tid & 31u;
  Mat D = matrix(dbytes);
  u64 sG = (D.input_side ? u64(D.out >> 1u) : 64ull) * u64(TB);
  bytes gb = trellis::tile_ptr<K>(D.base, D.inp, D.out, m, 0u, ob, D.input_side);
  bool aligned = (reinterpret_cast<u64>(gb) & 15ull) == 0ull;
  trellis::word_map wm = trellis::map_word<K>(lane);
  float acc[NT][4];
#pragma unroll
  for (u32 ti = 0u; ti < NT; ti++) acc[ti][0] = acc[ti][1] = acc[ti][2] = acc[ti][3] = 0.f;
  pf_fc2_loop<K, BOOK, NT>(act_slot, gb, sG, aligned, upto, tid, j, lane, wm, sm, acc);
  pf_store_fc2<NT>(acc, sm, sv_p, sg_p, z, ob, hidden, picks, n,
                   [&](u32 r) { return slot_tok_at(plan, is_sh, r0_sh, rs, r); },
                   [&](u32 r) { return is_sh ? picks : (u32)plan->slot_pick[rs][r]; },
                   [&](u32 tok, u32 pk) { return is_sh ? 1.0f : plan->tok_w[tok][pk]; });
}

#define FC2_PF_PARAMS (const Plan* plan, const u32* act_slot, bytes dbytes, const float* sv_p, const float* sg_p, u16* z, Pf2Smem& sm, bool is_sh, u32 r0_sh, u32 rs, u32 m, u32 ob, u32 upto, u32 hidden, u32 picks, u32 n)
#define FC2_PF_ARGS plan, act_slot, dbytes, sv_p, sg_p, z, sm, is_sh, r0_sh, rs, m, ob, upto, hidden, picks, n
template <u32 K, u32 BOOK>
static __device__ __noinline__ void fc2_pf_hot FC2_PF_PARAMS { DISPATCH_NT(fc2_pf_body, n, FC2_PF_ARGS); }
DEFINE_COLD(fc2_pf_cold, FC2_PF_PARAMS, (FC2_PF_ARGS), fc2_pf_body<K, BOOK, 8u>)

// Prefill down projection writing coalesced weighted fp16 partials.
extern "C" __global__ __launch_bounds__(256, FC2_PF_MIN_BLOCKS) void glm53_fc2_pf(
    const float* plan_in, const float* act_in, bytes down, bytes shared_down,
    const float* down_sv, const float* signs_out, const float* shared_down_sv, float* z_out,
    u32 rows, u32 hidden, u32 ffn, u32 picks, u32 k, u32 shared_k, u32 book, u32 shared_book,
    u32 first, u32 shared_ffn) {
  const Plan* plan = reinterpret_cast<const Plan*>(plan_in);
  u32 n_sh = (rows + 63u) >> 6u, slot = blockIdx.y, ob = blockIdx.x, chunks = ffn >> 7u;
  bool is_sh = slot < n_sh;
  if (!is_sh && slot - n_sh >= plan->n_slots) return;
  __shared__ Pf2Smem sm;
  u32 r0_sh = slot * 64u, rs = is_sh ? 0u : (slot - n_sh), n, m, upto;
  bytes dbytes;
  const float *sv_p, *sg_p;
  if (is_sh) { n = min(rows - r0_sh, 64u); m = 0u; upto = shared_ffn >> 7u; dbytes = shared_down; sv_p = shared_down_sv; sg_p = 0; k = shared_k; book = shared_book; }
  else { u32 e = plan->slot_expert[rs]; n = plan->slot_count[rs]; m = e - first; upto = chunks; dbytes = down; sv_p = down_sv + (u64)e * hidden; sg_p = signs_out; if (ob == 0u && threadIdx.x == 0u) trellis::note_hit(dbytes, m); }
  const u32* act_slot = reinterpret_cast<const u32*>(act_in) + (u64)slot * chunks * ACT_CHUNK;
  u16* z = reinterpret_cast<u16*>(z_out);
  DISPATCH_KB_HOT(fc2_pf_hot, fc2_pf_cold, k, book, FC2_PF_ARGS);
}

// Each row's fixed-order sum over picks, after output turn.
extern "C" __global__ FINISH_BOUNDS void glm53_finish(
    const float* plan_in, const float* down_sv, const float* signs_out,
    const float* shared_down_sv, const float* z_in, float* y,
    u32 rows, u32 hidden, u32 ffn, u32 picks, u32 shared_ffn, u32 z_chunks) {
  (void)ffn;
  const Plan* plan = reinterpret_cast<const Plan*>(plan_in);
  const u16* z = reinterpret_cast<const u16*>(z_in);
  u32 warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
  u32 base = blockIdx.x * 128u + (lane << 2u);
  if (z_chunks == 0u) {
    u32 row = blockIdx.y * 8u + warp;
    if (row < rows) pf_finish_row(z, y, row, base, hidden, picks, [&](u32 p) { return plan->tok_expert[row][p] != 0xFFFFFFFFu; });
    return;
  }
  __shared__ float wsum[PICKS + 1][128];
  u32 row = blockIdx.y;
  float acc[4] = {};
  if (warp <= picks) {
    bool is_shared = warp == picks;
    u32 e = is_shared ? 0u : plan->tok_expert[row][warp];
    if (is_shared || e != 0xFFFFFFFFu) {
      u32 upto = is_shared ? (shared_ffn >> 7u) : z_chunks;
      const u16* from = z + ((u64)row * (picks + 1u) + warp) * z_chunks * hidden + base;
      float s[4] = {};
      for (u32 c = 0u; c < upto; c++) {
        uint2 h = *(const uint2*)(from + (u64)c * hidden);
        s[0] += trellis::f16_to_f32(h.x & 0xFFFFu); s[1] += trellis::f16_to_f32(h.x >> 16u);
        s[2] += trellis::f16_to_f32(h.y & 0xFFFFu); s[3] += trellis::f16_to_f32(h.y >> 16u);
      }
      trellis::wht128(s, trellis::INV_SQRT128);
      if (is_shared) {
        float4 sv = *(const float4*)(shared_down_sv + base);
        acc[0] = s[0] * sv.x; acc[1] = s[1] * sv.y; acc[2] = s[2] * sv.z; acc[3] = s[3] * sv.w;
      } else {
        float gw = plan->tok_w[row][warp];
        float4 sv = *(const float4*)(down_sv + (u64)e * hidden + base);
        float4 sg = *(const float4*)(signs_out + base);
        acc[0] = gw * (s[0] * sv.x * sg.x); acc[1] = gw * (s[1] * sv.y * sg.y);
        acc[2] = gw * (s[2] * sv.z * sg.z); acc[3] = gw * (s[3] * sv.w * sg.w);
      }
    }
    *(float4*)(&wsum[warp][lane << 2u]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
  }
  prim::block_sync();
  if (warp == 0u) {
    float4 o = *(const float4*)(&wsum[0][lane << 2u]);
    for (u32 w = 1u; w <= picks; w++) {
      float4 q = *(const float4*)(&wsum[w][lane << 2u]);
      o.x += q.x; o.y += q.y; o.z += q.z; o.w += q.w;
    }
    *(float4*)(y + (u64)row * hidden + base) = o;
  }
}
