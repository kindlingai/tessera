// GLM-5.3 routed block: route+plan, fc1, fc2, finish.
#include <tessera/1/core>
#include <tessera/1/trellis>

using tessera::u16;
using tessera::u32;
using tessera::u64;
using tessera::u8;
namespace prim = tessera::prim;
namespace trellis = tessera::trellis;
using trellis::word;
typedef const u8* bytes;

#define PICKS 8
#define MAX_ROWS 64
#define MAX_SLOTS 512
#define ROUTE_SCALE 2.5f
#define PLAN_WORDS (1u + MAX_SLOTS * 6u + MAX_ROWS * PICKS * 2u)

// Compact routing plan for one layer launch.
struct Plan {
  u32 n_slots;
  u32 slot_expert[MAX_SLOTS];
  u32 slot_count[MAX_SLOTS];
  u32 slot_tok[MAX_SLOTS][2];
  u8 slot_pick[MAX_SLOTS][8];
  u32 tok_expert[MAX_ROWS][PICKS];
  float tok_w[MAX_ROWS][PICKS];
};
static_assert(sizeof(Plan) == PLAN_WORDS * 4u, "Plan size");

// Last-CTA completion counter, reset to zero by winner.
static __device__ u32 route_done;

static __device__ inline u16 f32_to_f16(float v) {
  u16 out;
  asm("{.reg .b16 t; cvt.rn.f16.f32 t, %1; mov.b16 %0, t;}" : "=h"(out) : "f"(v));
  return out;
}

// A code set's base, matrix shape and side.
struct Mat {
  bytes base;
  u32 inp, out;
  bool input_side;
};

static __device__ inline Mat matrix(bytes p) {
  Mat w = {p, word(p, 0), word(p, 4), (word(p, 8) & trellis::SIDE_MASK) == trellis::INPUT_SIDE};
  return w;
}

// One warp's picks and gates. Ties take lower expert.
static __device__ void route(
    const float* score, const float* bias, u32 experts, u32 picks, u16* ids, float* gates) {
  u32 lane = threadIdx.x & 31u;
  u64 taken = 0ull;
  float sum = 0.0f;
  float raw[8];
  float biased[8];
  bool cached = experts <= 256u;
  if (cached) {
    for (u32 i = 0u, e = lane; e < experts; i++, e += 32u) {
      float s = prim::load_fresh(score + e);
      raw[i] = s;
      biased[i] = s + bias[e];
    }
  }
  for (u32 p = 0u; p < picks; p++) {
    u32 best = experts;
    float most = 0.0f;
    float best_raw = 0.0f;
    for (u32 i = 0u, e = lane; e < experts; i++, e += 32u) {
      float s = cached ? raw[i] : prim::load_fresh(score + e);
      float v = cached ? biased[i] : (s + bias[e]);
      if (!((taken >> i) & 1ull) && (best == experts || v > most)) {
        best = e;
        most = v;
        best_raw = s;
      }
    }
    for (u32 off = 16u; off > 0u; off >>= 1u) {
      float v = __shfl_xor_sync(0xFFFFFFFFu, most, off);
      u32 e = __shfl_xor_sync(0xFFFFFFFFu, best, off);
      float r = __shfl_xor_sync(0xFFFFFFFFu, best_raw, off);
      bool better = best == experts || v > most || (v == most && e < best);
      if (e < experts && better) {
        best = e;
        most = v;
        best_raw = r;
      }
    }
    if ((best & 31u) == lane) taken |= 1ull << (best >> 5u);
    ids[p] = (u16)best;
    gates[p] = best_raw;
    sum += best_raw;
  }
  float inv = ROUTE_SCALE / sum;
  for (u32 p = 0u; p < picks; p++) gates[p] *= inv;
  __syncwarp();
}

// Four router weights from `at`, f32 or bf16.
static __device__ inline float4 weights4(const void* router, u64 at, u32 bf16) {
  if (!bf16) return *(const float4*)((const float*)router + at);
  uint2 h = *(const uint2*)((const unsigned short*)router + at);
  return make_float4(__uint_as_float(h.x << 16u), __uint_as_float(h.x & 0xFFFF0000u),
      __uint_as_float(h.y << 16u), __uint_as_float(h.y & 0xFFFF0000u));
}

struct PlanSmem {
  u16 ids[MAX_ROWS][PICKS];
  float gates[MAX_ROWS][PICKS];
  u32 emask[2][2048];
  u16 slot_of[2048];
  u32 wsum[8];
  u32 total;
};

union alignas(16) RouteSmem {
  float wsum[2][8][8];
  PlanSmem plan;
};

// Builds compact deduplicated expert slots across all rows.
static __device__ __noinline__ void route_plan(
    const float* scores, const float* bias, float* plan_out, PlanSmem& sm,
    u32 rows, u32 experts, u32 picks, u32 first, u32 count) {
  Plan* plan = reinterpret_cast<Plan*>(plan_out);
  u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
  for (u32 r = warp; r < rows; r += 8u) {
    route(scores + (u64)r * experts, bias, experts, picks, sm.ids[r], sm.gates[r]);
  }
  for (u32 i = tid; i < count; i += 256u) {
    sm.emask[0][i] = 0u;
    sm.emask[1][i] = 0u;
  }
  if (tid == 0u) sm.total = 0u;
  prim::block_sync();
  for (u32 i = tid; i < rows * picks; i += 256u) {
    u32 r = i / picks, p = i % picks;
    u32 e = sm.ids[r][p];
    bool mine = e >= first && e < first + count;
    plan->tok_expert[r][p] = mine ? e : 0xFFFFFFFFu;
    plan->tok_w[r][p] = sm.gates[r][p];
    if (mine) atomicOr(&sm.emask[r >> 5u][e - first], 1u << (r & 31u));
  }
  prim::block_sync();
  for (u32 base = 0u; base < count; base += 256u) {
    u32 le = base + tid;
    u64 m = le < count ? ((u64)sm.emask[0][le] | ((u64)sm.emask[1][le] << 32u)) : 0ull;
    u32 cnt = __popcll(m);
    u32 ns = (cnt + 7u) >> 3u;
    u32 incl = ns;
#pragma unroll
    for (u32 d = 1u; d < 32u; d <<= 1u) {
      u32 v = __shfl_up_sync(0xFFFFFFFFu, incl, d);
      if (lane >= d) incl += v;
    }
    if (lane == 31u) sm.wsum[warp] = incl;
    prim::block_sync();
    u32 chunk_base = sm.total;
    if (warp == 0u) {
      u32 wv = lane < 8u ? sm.wsum[lane] : 0u;
      u32 wi = wv;
#pragma unroll
      for (u32 d = 1u; d < 8u; d <<= 1u) {
        u32 v = __shfl_up_sync(0xFFFFFFFFu, wi, d);
        if (lane >= d) wi += v;
      }
      if (lane < 8u) sm.wsum[lane] = wi - wv;
      if (lane == 7u) sm.total = chunk_base + wi;
    }
    prim::block_sync();
    if (ns > 0u) {
      u32 s0 = chunk_base + sm.wsum[warp] + (incl - ns);
      sm.slot_of[le] = (u16)s0;
      u64 bits = m;
      for (u32 sub = 0u; sub < ns; sub++) {
        u32 s = s0 + sub;
        u32 take = min(cnt - sub * 8u, 8u);
        plan->slot_expert[s] = first + le;
        plan->slot_count[s] = take;
        u64 pack = 0ull;
        for (u32 k = 0u; k < take; k++) {
          u32 r = (u32)(__ffsll((long long)bits) - 1);
          bits &= bits - 1ull;
          pack |= (u64)r << (k << 3u);
        }
        plan->slot_tok[s][0] = (u32)pack;
        plan->slot_tok[s][1] = (u32)(pack >> 32u);
      }
    }
    prim::block_sync();
  }
  for (u32 i = tid; i < rows * picks; i += 256u) {
    u32 r = i / picks, p = i % picks;
    u32 e = sm.ids[r][p];
    if (e < first || e >= first + count) continue;
    u32 le = e - first;
    u64 em = (u64)sm.emask[0][le] | ((u64)sm.emask[1][le] << 32u);
    u32 rank = __popcll(em & ((1ull << r) - 1ull));
    plan->slot_pick[(u32)sm.slot_of[le] + (rank >> 3u)][rank & 7u] = (u8)p;
  }
  if (tid == 0u) plan->n_slots = sm.total;
}

// Scores via sigmoid(x router), then last CTA builds plan.
extern "C" __global__ __launch_bounds__(256, 2) void glm53_route(
    const float* x, const void* router, const float* bias, float* plan_out,
    u32 rows, u32 hidden, u32 experts, u32 bf16, u32 picks, u32 first, u32 count) {
  __shared__ RouteSmem sm;
  __shared__ u32 is_last;
  float* scores = plan_out + PLAN_WORDS;
  u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
  u32 e0 = blockIdx.x * 2u;
  if (e0 < experts) {
    bool has1 = e0 + 1u < experts;
    bool fast = bf16 && (hidden & 255u) == 0u && (((u64)router) & 15ull) == 0ull;
    const u8* wr0 = (const u8*)router + (u64)e0 * hidden * 2ull;
    const u8* wr1 = wr0 + (u64)hidden * 2ull;
    for (u32 r0 = 0u; r0 < rows; r0 += 8u) {
      u32 nr = min(rows - r0, 8u);
      float s0[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float s1[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      if (fast) {
        u32 i0 = warp * 256u;
        if (i0 < hidden) {
          u32 na[4], nb[4];
          u64 off0 = (u64)(i0 + (lane << 3u)) * 2ull;
          trellis::ld_nc_v4(wr0 + off0, na);
          if (has1) trellis::ld_nc_v4(wr1 + off0, nb);
          for (; i0 < hidden; i0 += 2048u) {
            u32 ha[4] = {na[0], na[1], na[2], na[3]};
            u32 hb[4] = {nb[0], nb[1], nb[2], nb[3]};
            if (i0 + 2048u < hidden) {
              u64 noff = (u64)(i0 + 2048u + (lane << 3u)) * 2ull;
              trellis::ld_nc_v4(wr0 + noff, na);
              if (has1) trellis::ld_nc_v4(wr1 + noff, nb);
            }
            float a0 = __uint_as_float(ha[0] << 16u), a1 = __uint_as_float(ha[0] & 0xFFFF0000u);
            float a2 = __uint_as_float(ha[1] << 16u), a3 = __uint_as_float(ha[1] & 0xFFFF0000u);
            float a4 = __uint_as_float(ha[2] << 16u), a5 = __uint_as_float(ha[2] & 0xFFFF0000u);
            float a6 = __uint_as_float(ha[3] << 16u), a7 = __uint_as_float(ha[3] & 0xFFFF0000u);
            float b0 = __uint_as_float(hb[0] << 16u), b1 = __uint_as_float(hb[0] & 0xFFFF0000u);
            float b2 = __uint_as_float(hb[1] << 16u), b3 = __uint_as_float(hb[1] & 0xFFFF0000u);
            float b4 = __uint_as_float(hb[2] << 16u), b5 = __uint_as_float(hb[2] & 0xFFFF0000u);
            float b6 = __uint_as_float(hb[3] << 16u), b7 = __uint_as_float(hb[3] & 0xFFFF0000u);
            u32 col = i0 + (lane << 3u);
#pragma unroll
            for (u32 r = 0u; r < 8u; r++) {
              if (r >= nr) break;
              const float* xp = x + (u64)(r0 + r) * hidden + col;
              float4 x0 = *(const float4*)xp;
              float4 x1 = *(const float4*)(xp + 4u);
              s0[r] = fmaf(a0, x0.x, fmaf(a1, x0.y, fmaf(a2, x0.z, fmaf(a3, x0.w, s0[r]))));
              s0[r] = fmaf(a4, x1.x, fmaf(a5, x1.y, fmaf(a6, x1.z, fmaf(a7, x1.w, s0[r]))));
              if (has1) {
                s1[r] = fmaf(b0, x0.x, fmaf(b1, x0.y, fmaf(b2, x0.z, fmaf(b3, x0.w, s1[r]))));
                s1[r] = fmaf(b4, x1.x, fmaf(b5, x1.y, fmaf(b6, x1.z, fmaf(b7, x1.w, s1[r]))));
              }
            }
          }
        }
      } else {
        for (u32 i = tid * 4u; i < hidden; i += 1024u) {
          float4 a0 = weights4(router, (u64)e0 * hidden + i, bf16);
          float4 a1 = has1 ? weights4(router, (u64)(e0 + 1u) * hidden + i, bf16)
                           : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
          for (u32 r = 0u; r < 8u; r++) {
            if (r >= nr) break;
            float4 b = *(const float4*)(x + (u64)(r0 + r) * hidden + i);
            s0[r] += a0.x * b.x + a0.y * b.y + a0.z * b.z + a0.w * b.w;
            if (has1) s1[r] += a1.x * b.x + a1.y * b.y + a1.z * b.z + a1.w * b.w;
          }
        }
      }
#pragma unroll
      for (u32 r = 0u; r < 8u; r++) {
        if (r >= nr) break;
        for (u32 off = 16u; off > 0u; off >>= 1u) {
          s0[r] += __shfl_xor_sync(0xFFFFFFFFu, s0[r], off);
          if (has1) s1[r] += __shfl_xor_sync(0xFFFFFFFFu, s1[r], off);
        }
        if (lane == 0u) {
          sm.wsum[0][r][warp] = s0[r];
          if (has1) sm.wsum[1][r][warp] = s1[r];
        }
      }
      prim::block_sync();
      if (warp < 2u && (warp == 0u || has1) && lane < nr) {
        float tot = sm.wsum[warp][lane][0] + sm.wsum[warp][lane][1] + sm.wsum[warp][lane][2] +
                    sm.wsum[warp][lane][3] + sm.wsum[warp][lane][4] + sm.wsum[warp][lane][5] +
                    sm.wsum[warp][lane][6] + sm.wsum[warp][lane][7];
        scores[(u64)(r0 + lane) * experts + e0 + warp] = 1.0f / (1.0f + __expf(-tot));
      }
      if (r0 + 8u < rows) prim::block_sync();
    }
  }
  __threadfence();
  prim::block_sync();
  if (tid == 0u) {
    u32 done = prim::fetch_add_gpu(&route_done, 1u) + 1u;
    is_last = done == gridDim.x ? 1u : 0u;
    if (is_last) prim::store_relaxed_gpu(&route_done, 0u);
  }
  prim::block_sync();
  if (!is_last) return;
  route_plan(scores, bias, plan_out, sm.plan, rows, experts, picks, first, count);
}

struct alignas(16) Fc1Smem {
  u32 btab[2048];
  u32 wtiles[2048];
};

struct alignas(16) Fc2Smem {
  u32 btab[512];
  u32 wtiles[2048];
};

// Warp-per-tile split-K fc1 in 16 KiB shared memory.
template <u32 K, u32 BOOK>
static __device__ __forceinline__ void fc1_body(
    const float* x, const float* su, const float* signs, bytes wbytes, float* gpart,
    Fc1Smem& sm, u32 m, u32 c, u32 h, u32 mat, u32 slot, u32 chunks, u32 ks1, u32 max_col,
    u32 hidden, u32 n, u64 toks_pack) {
  constexpr u32 TB = 32u * K;
  u32 tid = threadIdx.x, j = tid >> 5u, lane = tid & 31u;
  Mat W = matrix(wbytes);
  u32 nblk = (hidden >> 7u) / ks1;
  u32 blk0 = h * nblk;
  u64 sG = (W.input_side ? u64(W.out >> 1u) : 64ull) * u64(TB);
  bytes gb = trellis::tile_ptr<K>(W.base, W.inp, W.out, m, blk0 << 3u, c, W.input_side);
  bool aligned = (reinterpret_cast<u64>(gb) & 15ull) == 0ull;
  u32 nw[8];
  trellis::blk_load<K>(gb, tid, aligned, nw);
  trellis::word_map wm = trellis::map_word<K>(lane);
  float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
  for (u32 c0 = 0u; c0 < nblk; c0 += 4u) {
    u32 nb = min(4u, nblk - c0);
    if (c0 > 0u) prim::block_sync();
    u32 wpb = (nb <= 2u) ? (8u / nb) : 2u;
    u32 b = j / wpb;
    u32 r_base = j - b * wpb;
    if (b < nb && r_base < n) {
      u32 at = ((blk0 + c0 + b) << 7u) + (lane << 2u);
      float4 sv = *(const float4*)(su + at);
      float4 sg = signs ? *(const float4*)(signs + at) : make_float4(1.f, 1.f, 1.f, 1.f);
      float sc[4] = {sv.x * sg.x, sv.y * sg.y, sv.z * sg.z, sv.w * sg.w};
      for (u32 r = r_base; r < n; r += wpb) {
        u32 tok = (u32)((toks_pack >> (r << 3u)) & 0xFFu);
        float4 xv = *(const float4*)(x + (u64)tok * hidden + at);
        float v[4] = {xv.x * sc[0], xv.y * sc[1], xv.z * sc[2], xv.w * sc[3]};
        trellis::wht128(v, trellis::INV_SQRT128);
        trellis::store_bfrag(sm.btab + b * 512u, r, lane, v);
      }
    }
    trellis::blk_stream<K, BOOK, 1u>(
        gb, sG, c0, nb, nblk, tid, j, lane, aligned, wm, nw, sm.wtiles, sm.btab + (lane << 1u), 0u,
        acc);
  }
  float* gp = gpart + ((((u64)slot * chunks + c) * 2u + mat) * ks1 + h) * max_col * 128u;
  u32 col = (j << 4u) + (lane >> 2u);
  u32 r2 = (lane & 3u) << 1u;
  if (r2 < n) {
    gp[(u64)r2 * 128u + col] = acc[0][0];
    gp[(u64)r2 * 128u + col + 8u] = acc[0][2];
  }
  if (r2 + 1u < n) {
    gp[(u64)(r2 + 1u) * 128u + col] = acc[0][1];
    gp[(u64)(r2 + 1u) * 128u + col + 8u] = acc[0][3];
  }
}

template <u32 K, u32 BOOK>
static __device__ __noinline__ void fc1_cold(
    const float* x, const float* su, const float* signs, bytes wbytes, float* gpart,
    Fc1Smem& sm, u32 m, u32 c, u32 h, u32 mat, u32 slot, u32 chunks, u32 ks1, u32 max_col,
    u32 hidden, u32 n, u64 toks_pack) {
  fc1_body<K, BOOK>(x, su, signs, wbytes, gpart, sm, m, c, h, mat, slot, chunks, ks1, max_col, hidden, n, toks_pack);
}

// Split-K gate and up projections for routed and shared slots.
extern "C" __global__ __launch_bounds__(256, 4) void glm53_fc1(
    const float* x, const float* plan_in, const float* signs_in,
    const float* gate_su, bytes gate, const float* up_su, bytes up,
    bytes shared_gate, const float* shared_gate_su,
    bytes shared_up, const float* shared_up_su,
    float* gpart, u32 rows, u32 hidden, u32 ffn, u32 picks, u32 k, u32 shared_k,
    u32 book, u32 shared_book, u32 first, u32 shared_ffn, u32 ks1) {
  (void)picks;
  const Plan* plan = reinterpret_cast<const Plan*>(plan_in);
  u32 n_sh = (rows + 7u) >> 3u;
  u32 slot = blockIdx.y;
  bool is_shared = slot < n_sh;
  u32 chunks = ffn >> 7u;
  u32 c = blockIdx.x % chunks;
  u32 hm = blockIdx.x / chunks;
  u32 h = hm % ks1;
  u32 mat = hm / ks1;
  if (is_shared) {
    if (c >= (shared_ffn >> 7u)) return;
  } else if (slot - n_sh >= plan->n_slots) {
    return;
  }
  __shared__ Fc1Smem sm;
  u64 toks_pack = 0ull;
  u32 n, m;
  bytes wbytes;
  const float* su;
  const float* signs;
  if (is_shared) {
    u32 r0 = slot * 8u;
    n = min(rows - r0, 8u);
    for (u32 r = 0u; r < n; r++) toks_pack |= (u64)(r0 + r) << (r << 3u);
    m = 0u;
    wbytes = mat ? shared_up : shared_gate;
    su = mat ? shared_up_su : shared_gate_su;
    signs = 0;
    k = shared_k;
    book = shared_book;
  } else {
    u32 rs = slot - n_sh;
    n = plan->slot_count[rs];
    toks_pack = (u64)plan->slot_tok[rs][0] | ((u64)plan->slot_tok[rs][1] << 32u);
    u32 e = plan->slot_expert[rs];
    m = e - first;
    wbytes = mat ? up : gate;
    su = (mat ? up_su : gate_su) + (u64)e * hidden;
    signs = signs_in;
    if (c == 0u && h == 0u && threadIdx.x == 0u) trellis::note_hit(wbytes, m);
  }
  u32 max_col = min(rows, 8u);
  if (book && k == 3u) {
    fc1_body<3u, trellis::BOOK_MUL1>(x, su, signs, wbytes, gpart, sm, m, c, h, mat, slot, chunks, ks1, max_col, hidden, n, toks_pack);
  } else if (book && k == 4u) {
    fc1_cold<4u, trellis::BOOK_MUL1>(x, su, signs, wbytes, gpart, sm, m, c, h, mat, slot, chunks, ks1, max_col, hidden, n, toks_pack);
  } else if (book && k == 2u) {
    fc1_cold<2u, trellis::BOOK_MUL1>(x, su, signs, wbytes, gpart, sm, m, c, h, mat, slot, chunks, ks1, max_col, hidden, n, toks_pack);
  } else if (k == 3u) {
    fc1_cold<3u, trellis::BOOK_MCG>(x, su, signs, wbytes, gpart, sm, m, c, h, mat, slot, chunks, ks1, max_col, hidden, n, toks_pack);
  } else if (k == 4u) {
    fc1_cold<4u, trellis::BOOK_MCG>(x, su, signs, wbytes, gpart, sm, m, c, h, mat, slot, chunks, ks1, max_col, hidden, n, toks_pack);
  } else {
    fc1_cold<2u, trellis::BOOK_MCG>(x, su, signs, wbytes, gpart, sm, m, c, h, mat, slot, chunks, ks1, max_col, hidden, n, toks_pack);
  }
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
  u32 obs = hidden >> 7u;
  u32 ob_per = obs / ns2;
  u32 ob0 = ns * ob_per;
  u64 sO = (D.input_side ? 64ull : u64(D.inp >> 1u)) * u64(TB);
  bytes gb = trellis::tile_ptr<K>(D.base, D.inp, D.out, m, c << 3u, ob0, D.input_side);
  bool aligned = (reinterpret_cast<u64>(gb) & 15ull) == 0ull;
  u32 nw[8];
  trellis::blk_load<K>(gb, tid, aligned, nw);
  if (j < n) {
    float g[4] = {0.f, 0.f, 0.f, 0.f}, u[4] = {0.f, 0.f, 0.f, 0.f};
    u64 base_gp = ((u64)slot * chunks + c) * 2u * ks1 * max_col * 128u;
    const float* gg = gpart + base_gp + (u64)j * 128u + (lane << 2u);
    const float* gu = gpart + base_gp + (u64)ks1 * max_col * 128u + (u64)j * 128u + (lane << 2u);
    for (u32 h = 0u; h < ks1; h++) {
      float4 gv = trellis::ld_cg_f4(gg + (u64)h * max_col * 128u);
      float4 uv = trellis::ld_cg_f4(gu + (u64)h * max_col * 128u);
      g[0] += gv.x; g[1] += gv.y; g[2] += gv.z; g[3] += gv.w;
      u[0] += uv.x; u[1] += uv.y; u[2] += uv.z; u[3] += uv.w;
    }
    trellis::wht128(g, trellis::INV_SQRT128);
    trellis::wht128(u, trellis::INV_SQRT128);
    u32 o = c * 128u + (lane << 2u);
    float4 gsv = *(const float4*)(gate_sv + o);
    float4 usv = *(const float4*)(up_sv + o);
    float4 dsu = *(const float4*)(down_su + o);
    float gv[4] = {g[0] * gsv.x, g[1] * gsv.y, g[2] * gsv.z, g[3] * gsv.w};
    float uv[4] = {u[0] * usv.x, u[1] * usv.y, u[2] * usv.z, u[3] * usv.w};
    float ds[4] = {dsu.x, dsu.y, dsu.z, dsu.w};
    float a[4];
#pragma unroll
    for (u32 i = 0u; i < 4u; i++) a[i] = (gv[i] / (1.0f + __expf(-gv[i]))) * uv[i] * ds[i];
    trellis::wht128(a, trellis::INV_SQRT128);
    trellis::store_bfrag(sm.btab, j, lane, a);
  }
  trellis::word_map wm = trellis::map_word<K>(lane);
  const u32* ha = sm.btab + (lane << 1u);
  u32 nl = lane >> 2u, col2 = (lane & 3u) << 1u;
  for (u32 idx = 0u; idx < ob_per; idx++) {
    if (idx > 0u) prim::block_sync();
    trellis::blk_store<K>(nw, tid, sm.wtiles);
    prim::block_sync();
    if (idx + 1u < ob_per) {
      gb += sO;
      trellis::blk_load<K>(gb, tid, aligned, nw);
    }
    float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
    trellis::blk_mma<K, BOOK, 1u>(sm.wtiles, ha, 0u, j, lane, wm, acc);
    u32 out_base = ((ob0 + idx) << 7u) + (j << 4u) + nl;
    if (col2 < n) {
      u16* dst = z + z_off0 + out_base;
      dst[0] = f32_to_f16(acc[0][0]);
      dst[8] = f32_to_f16(acc[0][2]);
    }
    if (col2 + 1u < n) {
      u16* dst = z + z_off1 + out_base;
      dst[0] = f32_to_f16(acc[0][1]);
      dst[8] = f32_to_f16(acc[0][3]);
    }
  }
}

template <u32 K, u32 BOOK>
static __device__ __noinline__ void fc2_cold(
    const float* gpart, const float* gate_sv, const float* up_sv, const float* down_su,
    bytes dbytes, u16* z, Fc2Smem& sm, u32 m, u32 c, u32 ns, u32 ns2, u32 slot, u32 chunks, u32 ks1,
    u32 max_col, u32 hidden, u32 n, u64 z_off0, u64 z_off1) {
  fc2_body<K, BOOK>(gpart, gate_sv, up_sv, down_su, dbytes, z, sm, m, c, ns, ns2, slot, chunks, ks1, max_col, hidden, n, z_off0, z_off1);
}

// SwiGLU and down projection into fp16 partials.
extern "C" __global__ __launch_bounds__(256, 4) void glm53_fc2(
    const float* plan_in, const float* gpart,
    const float* gate_sv, const float* up_sv, const float* down_su, bytes down,
    const float* shared_gate_sv, const float* shared_up_sv,
    bytes shared_down, const float* shared_down_su,
    float* z_out, u32 rows, u32 hidden, u32 ffn, u32 picks, u32 k, u32 shared_k,
    u32 book, u32 shared_book, u32 first, u32 shared_ffn, u32 ks1, u32 ns2) {
  const Plan* plan = reinterpret_cast<const Plan*>(plan_in);
  u32 n_sh = (rows + 7u) >> 3u;
  u32 slot = blockIdx.y;
  bool is_shared = slot < n_sh;
  u32 c = blockIdx.x / ns2;
  u32 ns = blockIdx.x % ns2;
  if (is_shared) {
    if (c >= (shared_ffn >> 7u)) return;
  } else if (slot - n_sh >= plan->n_slots) {
    return;
  }
  u32 lane = threadIdx.x & 31u;
  u32 col2 = (lane & 3u) << 1u;
  u32 chunks = ffn >> 7u;
  u32 n, m;
  const float *gsv, *usv, *dsu;
  bytes dbytes;
  u64 z_off0 = 0ull, z_off1 = 0ull;
  if (is_shared) {
    u32 r0 = slot * 8u;
    n = min(rows - r0, 8u);
    m = 0u;
    gsv = shared_gate_sv;
    usv = shared_up_sv;
    dsu = shared_down_su;
    dbytes = shared_down;
    k = shared_k;
    book = shared_book;
    if (col2 < n) z_off0 = (((u64)(r0 + col2) * (picks + 1u) + picks) * chunks + c) * hidden;
    if (col2 + 1u < n) z_off1 = (((u64)(r0 + col2 + 1u) * (picks + 1u) + picks) * chunks + c) * hidden;
  } else {
    u32 rs = slot - n_sh;
    n = plan->slot_count[rs];
    u32 e = plan->slot_expert[rs];
    m = e - first;
    gsv = gate_sv + (u64)e * ffn;
    usv = up_sv + (u64)e * ffn;
    dsu = down_su + (u64)e * ffn;
    dbytes = down;
    if (c == 0u && ns == 0u && threadIdx.x == 0u) trellis::note_hit(dbytes, m);
    u32 tp = plan->slot_tok[rs][col2 >> 2u];
    u32 sh = (col2 & 3u) << 3u;
    if (col2 < n) {
      u32 r = (tp >> sh) & 0xFFu;
      u32 p = plan->slot_pick[rs][col2];
      z_off0 = (((u64)r * (picks + 1u) + p) * chunks + c) * hidden;
    }
    if (col2 + 1u < n) {
      u32 r = (tp >> (sh + 8u)) & 0xFFu;
      u32 p = plan->slot_pick[rs][col2 + 1u];
      z_off1 = (((u64)r * (picks + 1u) + p) * chunks + c) * hidden;
    }
  }
  __shared__ Fc2Smem sm;
  u32 max_col = min(rows, 8u);
  u16* z = reinterpret_cast<u16*>(z_out);
  if (book && k == 3u) {
    fc2_body<3u, trellis::BOOK_MUL1>(gpart, gsv, usv, dsu, dbytes, z, sm, m, c, ns, ns2, slot, chunks, ks1, max_col, hidden, n, z_off0, z_off1);
  } else if (book && k == 4u) {
    fc2_cold<4u, trellis::BOOK_MUL1>(gpart, gsv, usv, dsu, dbytes, z, sm, m, c, ns, ns2, slot, chunks, ks1, max_col, hidden, n, z_off0, z_off1);
  } else if (book && k == 2u) {
    fc2_cold<2u, trellis::BOOK_MUL1>(gpart, gsv, usv, dsu, dbytes, z, sm, m, c, ns, ns2, slot, chunks, ks1, max_col, hidden, n, z_off0, z_off1);
  } else if (k == 3u) {
    fc2_cold<3u, trellis::BOOK_MCG>(gpart, gsv, usv, dsu, dbytes, z, sm, m, c, ns, ns2, slot, chunks, ks1, max_col, hidden, n, z_off0, z_off1);
  } else if (k == 4u) {
    fc2_cold<4u, trellis::BOOK_MCG>(gpart, gsv, usv, dsu, dbytes, z, sm, m, c, ns, ns2, slot, chunks, ks1, max_col, hidden, n, z_off0, z_off1);
  } else {
    fc2_cold<2u, trellis::BOOK_MCG>(gpart, gsv, usv, dsu, dbytes, z, sm, m, c, ns, ns2, slot, chunks, ks1, max_col, hidden, n, z_off0, z_off1);
  }
}

// Each row's fixed-order sum over picks, after output turn.
extern "C" __global__ void glm53_finish(
    const float* plan_in, const float* down_sv, const float* signs_out,
    const float* shared_down_sv, const float* z_in,
    float* y, u32 rows, u32 hidden, u32 ffn, u32 picks, u32 shared_ffn) {
  (void)rows;
  const Plan* plan = reinterpret_cast<const Plan*>(plan_in);
  const u16* z = reinterpret_cast<const u16*>(z_in);
  __shared__ float wsum[PICKS + 1][128];
  u32 warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
  u32 base = blockIdx.x * 128u + (lane << 2u), row = blockIdx.y;
  u32 chunks = ffn >> 7u;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  if (warp <= picks) {
    bool is_shared = warp == picks;
    u32 e = is_shared ? 0u : plan->tok_expert[row][warp];
    if (is_shared || e != 0xFFFFFFFFu) {
      u32 upto = is_shared ? (shared_ffn >> 7u) : chunks;
      const u16* from = z + ((u64)row * (picks + 1u) + warp) * chunks * hidden + base;
      float s[4] = {0.f, 0.f, 0.f, 0.f};
      for (u32 c = 0u; c < upto; c++) {
        uint2 h = *(const uint2*)(from + (u64)c * hidden);
        s[0] += trellis::f16_to_f32(h.x & 0xFFFFu);
        s[1] += trellis::f16_to_f32(h.x >> 16u);
        s[2] += trellis::f16_to_f32(h.y & 0xFFFFu);
        s[3] += trellis::f16_to_f32(h.y >> 16u);
      }
      trellis::wht128(s, trellis::INV_SQRT128);
      if (is_shared) {
        float4 sv = *(const float4*)(shared_down_sv + base);
        acc[0] = s[0] * sv.x;
        acc[1] = s[1] * sv.y;
        acc[2] = s[2] * sv.z;
        acc[3] = s[3] * sv.w;
      } else {
        float gw = plan->tok_w[row][warp];
        float4 sv = *(const float4*)(down_sv + (u64)e * hidden + base);
        float4 sg = *(const float4*)(signs_out + base);
        acc[0] = gw * (s[0] * sv.x * sg.x);
        acc[1] = gw * (s[1] * sv.y * sg.y);
        acc[2] = gw * (s[2] * sv.z * sg.z);
        acc[3] = gw * (s[3] * sv.w * sg.w);
      }
    }
    *(float4*)(&wsum[warp][lane << 2u]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
  }
  prim::block_sync();
  if (warp == 0u) {
    float4 o = *(const float4*)(&wsum[0][lane << 2u]);
    for (u32 w = 1u; w <= picks; w++) {
      float4 q = *(const float4*)(&wsum[w][lane << 2u]);
      o.x += q.x;
      o.y += q.y;
      o.z += q.z;
      o.w += q.w;
    }
    *(float4*)(y + (u64)row * hidden + base) = o;
  }
}
