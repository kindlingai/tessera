// GLM-5.3 routed block: route, plan, fc1, fc2, finish.
#include <tessera/1/core>
#include <tessera/1/trellis>

using tessera::u16;
using tessera::u32;
using tessera::u64;
using tessera::u8;
namespace prim = tessera::prim;
namespace trellis = tessera::trellis;
typedef const u8* bytes;

#define PICKS 8
#define MAX_ROWS 16
#define MAX_SLOTS 128
#define ROUTE_SCALE 2.5f

// Compact routing plan for one layer launch.
struct Plan {
  u32 n_slots;
  u32 slot_expert[MAX_SLOTS];
  u32 slot_count[MAX_SLOTS];
  u32 slot_tok[MAX_SLOTS][8];
  u32 slot_pick[MAX_SLOTS][8];
  u32 tok_expert[MAX_ROWS][PICKS];
  float tok_w[MAX_ROWS][PICKS];
};

static __device__ inline u32 word(bytes p, u32 at) {
  return *(const u32*)(p + at);
}

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
  Mat w = {p, word(p, 0), word(p, 4), word(p, 8) == trellis::INPUT_SIDE};
  return w;
}

// Two 16-bit states to E4M3-rounded f16x2.
template <u32 BOOK>
static __device__ inline u32 decode_pair(u32 a, u32 b) {
  if constexpr (BOOK == trellis::BOOK_MUL1) {
    u32 s0, s1, v;
    asm("dp4a.u32.u32 %0, %1, %2, %3;"
        : "=r"(s0) : "r"(a * trellis::MUL1), "r"(0x01010101u), "r"(0x6400u));
    asm("dp4a.u32.u32 %0, %1, %2, %3;"
        : "=r"(s1) : "r"(b * trellis::MUL1), "r"(0x01010101u), "r"(0x6400u));
    u32 h = trellis::prmt(s0, s1, 0x5410u);
    asm("fma.rn.f16x2 %0, %1, %2, %3;"
        : "=r"(v) : "r"(h), "r"(0x1EEE1EEEu), "r"(0xC931C931u));
    return trellis::narrow_f16x2_e4m3(v);
  } else {
    u32 x = a * trellis::MCG, y = b * trellis::MCG, sum;
    asm("lop3.b32 %0, %1, %2, %3, 0x6A;"
        : "=r"(x) : "r"(x), "r"(0x8FFF8FFFu), "r"(0x3B603B60u));
    asm("lop3.b32 %0, %1, %2, %3, 0x6A;"
        : "=r"(y) : "r"(y), "r"(0x8FFF8FFFu), "r"(0x3B603B60u));
    asm("add.rn.f16x2 %0, %1, %2;"
        : "=r"(sum)
        : "r"(trellis::prmt(x, y, 0x5410u)), "r"(trellis::prmt(x, y, 0x7632u)));
    return trellis::narrow_f16x2_e4m3(sum);
  }
}

// Decodes one smem 16x16 tile into an MMA A fragment.
template <u32 K, u32 BOOK>
static __device__ inline void decode_tile(
    const u32* words, const trellis::lane_map& lm, u32 frag[4]) {
  u32 w0 = words[lm.q0], w1 = words[lm.q1], w2 = words[lm.q2];
  u32 hi = trellis::shf_l(w1, w0, lm.shift);
  u32 lo = trellis::shf_l(w2, w1, lm.shift);
  u32 st[8];
#pragma unroll
  for (u32 s = 0u; s < 8u; s++) st[s] = trellis::shf_l(lo, hi, K * s) >> 16u;
  frag[0] = decode_pair<BOOK>(st[0], st[1]);
  frag[1] = decode_pair<BOOK>(st[4], st[5]);
  frag[2] = decode_pair<BOOK>(st[2], st[3]);
  frag[3] = decode_pair<BOOK>(st[6], st[7]);
}

// Stores one rotated 128-vector into MMA B fragment table.
static __device__ inline void store_frag(u32* tab, u32 row, u32 lane, const float v[4]) {
  u32 ks = lane >> 2u;
  u32 q = lane & 3u;
  u32 reg = q >> 1u;
  u32 c = (q & 1u) << 1u;
  u32* p = tab + ((ks * 32u + row * 4u + c) * 2u + reg);
  p[0] = trellis::f32x2_to_f16x2(v[0], v[1]);
  p[2] = trellis::f32x2_to_f16x2(v[2], v[3]);
}

// One warp's picks and gates. Ties take lower expert.
static __device__ void route(
    const float* score, const float* bias, u32 experts, u32 picks, u32* ids, float* gates) {
  u32 lane = threadIdx.x & 31u;
  u64 taken = 0ull;
  float sum = 0.0f;
  for (u32 p = 0u; p < picks; p++) {
    u32 best = experts;
    float most = 0.0f;
    for (u32 i = 0u, e = lane; e < experts; i++, e += 32u) {
      float v = score[e] + bias[e];
      if (!((taken >> i) & 1ull) && (best == experts || v > most)) {
        best = e;
        most = v;
      }
    }
    for (u32 off = 16u; off > 0u; off >>= 1u) {
      float v = __shfl_xor_sync(0xFFFFFFFFu, most, off);
      u32 e = __shfl_xor_sync(0xFFFFFFFFu, best, off);
      bool better = best == experts || v > most || (v == most && e < best);
      if (e < experts && better) {
        best = e;
        most = v;
      }
    }
    if ((best & 31u) == lane) taken |= 1ull << (best >> 5u);
    ids[p] = best;
    sum += score[best];
  }
  for (u32 p = 0u; p < picks; p++) gates[p] = score[ids[p]] / sum * ROUTE_SCALE;
  __syncwarp();
}

// Four router weights from `at`, f32 or bf16.
static __device__ inline float4 weights4(const void* router, u64 at, u32 bf16) {
  if (!bf16) return *(const float4*)((const float*)router + at);
  uint2 h = *(const uint2*)((const unsigned short*)router + at);
  return make_float4(__uint_as_float(h.x << 16u), __uint_as_float(h.x & 0xFFFF0000u),
      __uint_as_float(h.y << 16u), __uint_as_float(h.y & 0xFFFF0000u));
}

// Scores via sigmoid(x router). One CTA per expert.
extern "C" __global__ __launch_bounds__(256) void glm53_route(const float* x, const void* router,
    float* scores, u32 rows, u32 hidden, u32 experts, u32 bf16) {
  __shared__ float wsum[8][8];
  u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
  u32 e = blockIdx.x;
  if (e >= experts) return;
  bool fast = bf16 && (hidden & 255u) == 0u && (((u64)router) & 15ull) == 0ull;
  const u8* wr = (const u8*)router + (u64)e * hidden * 2ull;
  for (u32 r0 = 0u; r0 < rows; r0 += 8u) {
    u32 nr = min(rows - r0, 8u);
    float s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (fast) {
      for (u32 i0 = warp * 256u; i0 < hidden; i0 += 2048u) {
        u32 col = i0 + (lane << 3u);
        u32 h[4];
        trellis::ld_nc_v4(wr + (u64)col * 2ull, h);
        float w0 = __uint_as_float(h[0] << 16u), w1 = __uint_as_float(h[0] & 0xFFFF0000u);
        float w2 = __uint_as_float(h[1] << 16u), w3 = __uint_as_float(h[1] & 0xFFFF0000u);
        float w4 = __uint_as_float(h[2] << 16u), w5 = __uint_as_float(h[2] & 0xFFFF0000u);
        float w6 = __uint_as_float(h[3] << 16u), w7 = __uint_as_float(h[3] & 0xFFFF0000u);
#pragma unroll
        for (u32 r = 0u; r < 8u; r++) {
          if (r >= nr) break;
          const float* xp = x + (u64)(r0 + r) * hidden + col;
          float4 x0 = *(const float4*)xp;
          float4 x1 = *(const float4*)(xp + 4u);
          s[r] = fmaf(w0, x0.x, fmaf(w1, x0.y, fmaf(w2, x0.z, fmaf(w3, x0.w, s[r]))));
          s[r] = fmaf(w4, x1.x, fmaf(w5, x1.y, fmaf(w6, x1.z, fmaf(w7, x1.w, s[r]))));
        }
      }
    } else {
      for (u32 i = tid * 4u; i < hidden; i += 1024u) {
        float4 a = weights4(router, (u64)e * hidden + i, bf16);
#pragma unroll
        for (u32 r = 0u; r < 8u; r++) {
          if (r >= nr) break;
          float4 b = *(const float4*)(x + (u64)(r0 + r) * hidden + i);
          s[r] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
      }
    }
#pragma unroll
    for (u32 r = 0u; r < 8u; r++) {
      if (r >= nr) break;
      for (u32 off = 16u; off > 0u; off >>= 1u) s[r] += __shfl_xor_sync(0xFFFFFFFFu, s[r], off);
      if (lane == 0u) wsum[r][warp] = s[r];
    }
    prim::block_sync();
    if (warp == 0u && lane < nr) {
      float tot = wsum[lane][0] + wsum[lane][1] + wsum[lane][2] + wsum[lane][3] +
                  wsum[lane][4] + wsum[lane][5] + wsum[lane][6] + wsum[lane][7];
      scores[(u64)(r0 + lane) * experts + e] = 1.0f / (1.0f + __expf(-tot));
    }
    if (r0 + 8u < rows) prim::block_sync();
  }
}

// Builds compact deduplicated expert slots across all rows.
extern "C" __global__ __launch_bounds__(256) void glm53_plan(
    const float* scores, const float* bias, float* plan_out,
    u32 rows, u32 experts, u32 picks, u32 first, u32 count) {
  Plan* plan = reinterpret_cast<Plan*>(plan_out);
  __shared__ u32 ids[MAX_ROWS][PICKS];
  __shared__ float gates[MAX_ROWS][PICKS];
  __shared__ u32 emask[2048];
  __shared__ u32 slot_of[2048];
  __shared__ u32 wsum[8];
  __shared__ u32 total;
  u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
  for (u32 r = warp; r < rows; r += 8u) {
    route(scores + (u64)r * experts, bias, experts, picks, ids[r], gates[r]);
  }
  for (u32 i = tid; i < count; i += 256u) emask[i] = 0u;
  if (tid == 0u) total = 0u;
  prim::block_sync();
  for (u32 i = tid; i < rows * picks; i += 256u) {
    u32 r = i / picks, p = i % picks;
    u32 e = ids[r][p];
    bool mine = e >= first && e < first + count;
    plan->tok_expert[r][p] = mine ? e : 0xFFFFFFFFu;
    plan->tok_w[r][p] = gates[r][p];
    if (mine) atomicOr(&emask[e - first], 1u << r);
  }
  prim::block_sync();
  for (u32 base = 0u; base < count; base += 256u) {
    u32 le = base + tid;
    u32 m = le < count ? emask[le] : 0u;
    u32 cnt = __popc(m);
    u32 ns = (cnt + 7u) >> 3u;
    u32 incl = ns;
#pragma unroll
    for (u32 d = 1u; d < 32u; d <<= 1u) {
      u32 v = __shfl_up_sync(0xFFFFFFFFu, incl, d);
      if (lane >= d) incl += v;
    }
    if (lane == 31u) wsum[warp] = incl;
    prim::block_sync();
    u32 chunk_base = total;
    if (warp == 0u) {
      u32 wv = lane < 8u ? wsum[lane] : 0u;
      u32 wi = wv;
#pragma unroll
      for (u32 d = 1u; d < 8u; d <<= 1u) {
        u32 v = __shfl_up_sync(0xFFFFFFFFu, wi, d);
        if (lane >= d) wi += v;
      }
      if (lane < 8u) wsum[lane] = wi - wv;
      if (lane == 7u) total = chunk_base + wi;
    }
    prim::block_sync();
    if (ns > 0u) {
      u32 s0 = chunk_base + wsum[warp] + (incl - ns);
      slot_of[le] = s0;
      u32 bits = m;
      for (u32 sub = 0u; sub < ns; sub++) {
        u32 s = s0 + sub;
        u32 take = min(cnt - sub * 8u, 8u);
        plan->slot_expert[s] = first + le;
        plan->slot_count[s] = take;
        for (u32 k = 0u; k < take; k++) {
          u32 r = __ffs(bits) - 1u;
          bits &= bits - 1u;
          plan->slot_tok[s][k] = r;
        }
      }
    }
    prim::block_sync();
  }
  for (u32 i = tid; i < rows * picks; i += 256u) {
    u32 r = i / picks, p = i % picks;
    u32 e = ids[r][p];
    if (e < first || e >= first + count) continue;
    u32 le = e - first;
    u32 rank = __popc(emask[le] & ((1u << r) - 1u));
    plan->slot_pick[slot_of[le] + (rank >> 3u)][rank & 7u] = p;
  }
  if (tid == 0u) plan->n_slots = total;
}

union alignas(16) Fc1Smem {
  struct {
    u32 btab[8][512];
    u32 wtiles[8][256];
  } k;
  float red[8][4][128];
};

struct alignas(16) Fc2Smem {
  u32 btab[512];
  u32 wtiles[8][256];
};

// Runs 8 k-steps of one 128-wide block with 1-ahead prefetch.
template <u32 K, u32 BOOK>
static __device__ __forceinline__ void step_block(
    bytes bptr, bytes next_blk, u32 lane, bool wide, u32 reg[8], u32* wtiles,
    const u32* ha, const trellis::lane_map& pos, float acc[8][4]) {
  for (u32 ks = 0u; ks < 8u; ks++) {
    __syncwarp();
    trellis::tile_store<K>(reg, lane, wide, wtiles);
    __syncwarp();
    if (ks + 1u < 8u) {
      trellis::tile_load<K>(bptr + (u64)(ks + 1u) * (256u * K), lane, wide, reg);
    } else if (next_blk != 0) {
      trellis::tile_load<K>(next_blk, lane, wide, reg);
    }
    u32 fb[2] = {ha[ks * 64u], ha[ks * 64u + 1u]};
#pragma unroll
    for (u32 mt = 0u; mt < 8u; mt++) {
      u32 fa[4];
      decode_tile<K, BOOK>(wtiles + mt * 8u * K, pos, fa);
      trellis::mma_m16n8k16(acc[mt], fa, fb);
    }
  }
}

// Split-K fc1 body with coalesced 128-bit tile loads.
template <u32 K, u32 BOOK>
static __device__ __forceinline__ void fc1_body(
    const float* x, const float* su, const float* signs, bytes wbytes, float* gpart,
    Fc1Smem& sm, u32 m, u32 c, u32 h, u32 mat, u32 slot, u32 chunks, u32 ks1, u32 max_col,
    u32 hidden, u32 n, u32 toks_pack) {
  u32 warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
  Mat W = matrix(wbytes);
  u32 nblk = (hidden >> 7u) / ks1;
  u32 blk0 = h * nblk;
  u32 blk1 = blk0 + nblk;
  bool wide = (reinterpret_cast<u64>(W.base + trellis::HEAD) & 15u) == 0u;
  trellis::lane_map pos = trellis::map_lane<K, BOOK>(lane);
  u32* wbtab = sm.k.btab[warp];
  u32* wtiles = sm.k.wtiles[warp];
  const u32* ha = wbtab + (lane << 1u);
  float acc[8][4];
#pragma unroll
  for (u32 mt = 0u; mt < 8u; mt++) {
    acc[mt][0] = 0.f; acc[mt][1] = 0.f; acc[mt][2] = 0.f; acc[mt][3] = 0.f;
  }
  if (blk0 + warp < blk1) {
    u32 reg[8];
    bytes wptr = trellis::tile_ptr<K>(W.base, W.inp, W.out, m, (blk0 + warp) << 3u, c, W.input_side);
    trellis::tile_load<K>(wptr, lane, wide, reg);
    for (u32 ib = blk0 + warp; ib < blk1; ib += 8u) {
      u32 at = (ib << 7u) + (lane << 2u);
      float4 sv = *(const float4*)(su + at);
      float4 sg = signs ? *(const float4*)(signs + at) : make_float4(1.f, 1.f, 1.f, 1.f);
      float sc[4] = {sv.x * sg.x, sv.y * sg.y, sv.z * sg.z, sv.w * sg.w};
      for (u32 r = 0u; r < n; r++) {
        u32 tok = (toks_pack >> (r << 2u)) & 15u;
        float4 xv = *(const float4*)(x + (u64)tok * hidden + at);
        float v[4] = {xv.x * sc[0], xv.y * sc[1], xv.z * sc[2], xv.w * sc[3]};
        trellis::wht128(v, trellis::INV_SQRT128);
        store_frag(wbtab, r, lane, v);
      }
      bytes bptr = trellis::tile_ptr<K>(W.base, W.inp, W.out, m, ib << 3u, c, W.input_side);
      bytes next = ib + 8u < blk1
          ? trellis::tile_ptr<K>(W.base, W.inp, W.out, m, (ib + 8u) << 3u, c, W.input_side)
          : 0;
      step_block<K, BOOK>(bptr, next, lane, wide, reg, wtiles, ha, pos, acc);
    }
  }
  float* gp = gpart + ((((u64)slot * chunks + c) * 2u + mat) * ks1 + h) * max_col * 128u;
  u32 nl = lane >> 2u, col2 = (lane & 3u) << 1u;
  for (u32 r0 = 0u; r0 < n; r0 += 4u) {
    u32 nr = min(n - r0, 4u);
    prim::block_sync();
    if (col2 >= r0 && col2 < r0 + nr) {
      float* dst = sm.red[warp][col2 - r0];
#pragma unroll
      for (u32 mt = 0u; mt < 8u; mt++) {
        dst[mt * 16u + nl] = acc[mt][0];
        dst[mt * 16u + nl + 8u] = acc[mt][2];
      }
    }
    if (col2 + 1u >= r0 && col2 + 1u < r0 + nr) {
      float* dst = sm.red[warp][col2 + 1u - r0];
#pragma unroll
      for (u32 mt = 0u; mt < 8u; mt++) {
        dst[mt * 16u + nl] = acc[mt][1];
        dst[mt * 16u + nl + 8u] = acc[mt][3];
      }
    }
    prim::block_sync();
    if (warp < nr) {
      float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
      for (u32 w = 0u; w < 8u; w++) {
        float4 v = *(const float4*)(&sm.red[w][warp][lane << 2u]);
        s0 += v.x; s1 += v.y; s2 += v.z; s3 += v.w;
      }
      *(float4*)(gp + (u64)(r0 + warp) * 128u + (lane << 2u)) = make_float4(s0, s1, s2, s3);
    }
  }
}

template <u32 K, u32 BOOK>
static __device__ __noinline__ void fc1_cold(
    const float* x, const float* su, const float* signs, bytes wbytes, float* gpart,
    Fc1Smem& sm, u32 m, u32 c, u32 h, u32 mat, u32 slot, u32 chunks, u32 ks1, u32 max_col,
    u32 hidden, u32 n, u32 toks_pack) {
  fc1_body<K, BOOK>(x, su, signs, wbytes, gpart, sm, m, c, h, mat, slot, chunks, ks1, max_col, hidden, n, toks_pack);
}

// Split-K gate and up projections for routed and shared slots.
extern "C" __global__ __launch_bounds__(256, 3) void glm53_fc1(
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
  u32 h = blockIdx.x / (2u * chunks);
  u32 mc = blockIdx.x % (2u * chunks);
  u32 mat = mc / chunks;
  u32 c = mc % chunks;
  if (is_shared) {
    if (c >= (shared_ffn >> 7u)) return;
  } else if (slot - n_sh >= plan->n_slots) {
    return;
  }
  __shared__ Fc1Smem sm;
  u32 toks_pack = 0u;
  u32 n, m;
  bytes wbytes;
  const float* su;
  const float* signs;
  if (is_shared) {
    u32 r0 = slot * 8u;
    n = min(rows - r0, 8u);
    for (u32 r = 0u; r < n; r++) toks_pack |= (r0 + r) << (r << 2u);
    m = 0u;
    wbytes = mat ? shared_up : shared_gate;
    su = mat ? shared_up_su : shared_gate_su;
    signs = 0;
    k = shared_k;
    book = shared_book;
  } else {
    u32 rs = slot - n_sh;
    n = plan->slot_count[rs];
    for (u32 r = 0u; r < n; r++) toks_pack |= plan->slot_tok[rs][r] << (r << 2u);
    u32 e = plan->slot_expert[rs];
    m = e - first;
    wbytes = mat ? up : gate;
    su = (mat ? up_su : gate_su) + (u64)e * hidden;
    signs = signs_in;
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

// Split-N fc2 body with coalesced 128-bit tile loads.
template <u32 K, u32 BOOK>
static __device__ __forceinline__ void fc2_body(
    const float* gpart, const float* gate_sv, const float* up_sv, const float* down_su,
    bytes dbytes, u16* z, Fc2Smem& sm, u32 m, u32 c, u32 ns, u32 ns2, u32 slot, u32 chunks, u32 ks1,
    u32 max_col, u32 hidden, u32 n, u64 z_off0, u64 z_off1) {
  u32 warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
  Mat D = matrix(dbytes);
  u32 obs = hidden >> 7u;
  u32 ob_per = obs / ns2;
  u32 ob0 = ns * ob_per;
  u32 ob1 = ob0 + ob_per;
  bool wide = (reinterpret_cast<u64>(D.base + trellis::HEAD) & 15u) == 0u;
  trellis::lane_map pos = trellis::map_lane<K, BOOK>(lane);
  u32* wtiles = sm.wtiles[warp];
  u32 reg[8];
  if (ob0 + warp < ob1) {
    trellis::tile_load<K>(
        trellis::tile_ptr<K>(D.base, D.inp, D.out, m, c << 3u, ob0 + warp, D.input_side),
        lane, wide, reg);
  }
  if (warp < n) {
    float g[4] = {0.f, 0.f, 0.f, 0.f}, u[4] = {0.f, 0.f, 0.f, 0.f};
    u64 base_gp = ((u64)slot * chunks + c) * 2u * ks1 * max_col * 128u;
    const float* gg = gpart + base_gp + (u64)warp * 128u + (lane << 2u);
    const float* gu = gpart + base_gp + (u64)ks1 * max_col * 128u + (u64)warp * 128u + (lane << 2u);
    for (u32 h = 0u; h < ks1; h++) {
      float4 gv = *(const float4*)(gg + (u64)h * max_col * 128u);
      float4 uv = *(const float4*)(gu + (u64)h * max_col * 128u);
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
    store_frag(sm.btab, warp, lane, a);
  }
  prim::block_sync();
  if (ob0 + warp >= ob1) return;
  const u32* ha = sm.btab + (lane << 1u);
  u32 nl = lane >> 2u, col2 = (lane & 3u) << 1u;
  for (u32 ob = ob0 + warp; ob < ob1; ob += 8u) {
    float acc[8][4];
#pragma unroll
    for (u32 mt = 0u; mt < 8u; mt++) {
      acc[mt][0] = 0.f; acc[mt][1] = 0.f; acc[mt][2] = 0.f; acc[mt][3] = 0.f;
    }
    bytes dptr = trellis::tile_ptr<K>(D.base, D.inp, D.out, m, c << 3u, ob, D.input_side);
    bytes next = ob + 8u < ob1
        ? trellis::tile_ptr<K>(D.base, D.inp, D.out, m, c << 3u, ob + 8u, D.input_side)
        : 0;
    step_block<K, BOOK>(dptr, next, lane, wide, reg, wtiles, ha, pos, acc);
    u32 out_base = (ob << 7u) + nl;
    if (col2 < n) {
      u16* dst = z + z_off0 + out_base;
#pragma unroll
      for (u32 mt = 0u; mt < 8u; mt++) {
        dst[mt * 16u] = f32_to_f16(acc[mt][0]);
        dst[mt * 16u + 8u] = f32_to_f16(acc[mt][2]);
      }
    }
    if (col2 + 1u < n) {
      u16* dst = z + z_off1 + out_base;
#pragma unroll
      for (u32 mt = 0u; mt < 8u; mt++) {
        dst[mt * 16u] = f32_to_f16(acc[mt][1]);
        dst[mt * 16u + 8u] = f32_to_f16(acc[mt][3]);
      }
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

// SwiGLU and down projection into fp16 `z[row][pick][chunk][hidden]`.
extern "C" __global__ __launch_bounds__(256, 3) void glm53_fc2(
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
    if (col2 < n) {
      u32 r = plan->slot_tok[rs][col2], p = plan->slot_pick[rs][col2];
      z_off0 = (((u64)r * (picks + 1u) + p) * chunks + c) * hidden;
    }
    if (col2 + 1u < n) {
      u32 r = plan->slot_tok[rs][col2 + 1u], p = plan->slot_pick[rs][col2 + 1u];
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
