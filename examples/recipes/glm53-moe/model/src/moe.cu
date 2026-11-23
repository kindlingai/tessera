// GLM-5.3's routed block: route, experts, then finish.
#include <tessera/1/core>
#include <tessera/1/trellis>

using tessera::u32;
using tessera::u64;
using tessera::u8;
namespace prim = tessera::prim;
namespace trellis = tessera::trellis;
typedef const u8* bytes;

// Most picks per row, and rows per expert block.
#define PICKS 8
#define ROWS 4
#define WARPS 8
// The gate scale of the release.
#define ROUTE_SCALE 2.5f

static __device__ inline u32 word(bytes p, u32 at) {
  return *(const u32*)(p + at);
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

// Loads eight tiles at input tile `i`, output block `o`.
template <u32 K>
static __device__ inline void load_tiles(
    const Mat& w, u32 m, u32 i, u32 o, u32 lane, bool wide, u32 reg[8]) {
  bytes p = trellis::tile_ptr<K>(w.base, w.inp, w.out, m, i, o, w.input_side);
  trellis::tile_load<K>(p, lane, wide, reg);
}

// One warp's picks and gates. Ties take the lower expert.
static __device__ void route(const float* score, const float* bias, u32 experts, u32 picks,
    u32* ids, float* gates) {
  u32 lane = threadIdx.x % 32;
  u64 taken = 0;
  float sum = 0.0f;
  for (u32 p = 0; p < picks; p++) {
    u32 best = experts;
    float most = 0.0f;
    for (u32 i = 0, e = lane; e < experts; i++, e += 32) {
      float v = score[e] + bias[e];
      if (!((taken >> i) & 1) && (best == experts || v > most)) {
        best = e;
        most = v;
      }
    }
    for (u32 off = 16; off > 0; off /= 2) {
      float v = __shfl_xor_sync(0xFFFFFFFFu, most, off);
      u32 e = __shfl_xor_sync(0xFFFFFFFFu, best, off);
      bool better = best == experts || v > most || (v == most && e < best);
      if (e < experts && better) {
        best = e;
        most = v;
      }
    }
    if (best % 32 == lane) taken |= 1ull << (best / 32);
    ids[p] = best;
    sum += score[best];
  }
  for (u32 p = 0; p < picks; p++) gates[p] = score[ids[p]] / sum * ROUTE_SCALE;
  __syncwarp();
}

// Four router weights from `at`, f32 or bf16.
static __device__ inline float4 weights4(const void* router, u64 at, u32 bf16) {
  if (!bf16) return *(const float4*)((const float*)router + at);
  uint2 h = *(const uint2*)((const unsigned short*)router + at);
  return make_float4(__uint_as_float(h.x << 16), __uint_as_float(h.x & 0xFFFF0000u),
      __uint_as_float(h.y << 16), __uint_as_float(h.y & 0xFFFF0000u));
}

// `scores = sigmoid(x router)`. One warp per expert.
extern "C" __global__ void glm53_route(const float* x, const void* router, float* scores,
    u32 rows, u32 hidden, u32 experts, u32 bf16) {
  u32 lane = threadIdx.x & 31u;
  u32 e = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (e >= experts) return;
  if (rows == 1 && bf16 && (hidden & 7u) == 0u && (((u64)router) & 15ull) == 0ull) {
    const u8* wr = (const u8*)router + (u64)e * hidden * 2ull;
    float s = 0.0f;
    for (u32 i = lane << 3; i < hidden; i += 256u) {
      u32 h[4];
      trellis::ld_nc_v4(wr + (u64)i * 2ull, h);
      float4 x0 = *(const float4*)(x + i);
      float4 x1 = *(const float4*)(x + i + 4u);
      s = fmaf(__uint_as_float(h[0] << 16), x0.x,
          fmaf(__uint_as_float(h[0] & 0xFFFF0000u), x0.y,
          fmaf(__uint_as_float(h[1] << 16), x0.z,
          fmaf(__uint_as_float(h[1] & 0xFFFF0000u), x0.w, s))));
      s = fmaf(__uint_as_float(h[2] << 16), x1.x,
          fmaf(__uint_as_float(h[2] & 0xFFFF0000u), x1.y,
          fmaf(__uint_as_float(h[3] << 16), x1.z,
          fmaf(__uint_as_float(h[3] & 0xFFFF0000u), x1.w, s))));
    }
    for (u32 off = 16; off > 0; off /= 2) s += __shfl_xor_sync(0xFFFFFFFFu, s, off);
    if (lane == 0) scores[e] = 1.0f / (1.0f + __expf(-s));
    return;
  }
  for (u32 r0 = 0; r0 < rows; r0 += 8) {
    float s[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 i = 4 * lane; i < hidden; i += 128) {
      float4 a = weights4(router, (u64)e * hidden + i, bf16);
#pragma unroll
      for (u32 r = 0; r < 8; r++) {
        if (r0 + r >= rows) break;
        float4 b = *(const float4*)(x + (u64)(r0 + r) * hidden + i);
        s[r] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
      }
    }
#pragma unroll
    for (u32 r = 0; r < 8; r++) {
      for (u32 off = 16; off > 0; off /= 2) s[r] += __shfl_xor_sync(0xFFFFFFFFu, s[r], off);
      if (lane == 0 && r0 + r < rows) scores[(u64)(r0 + r) * experts + e] = 1.0f / (1.0f + expf(-s[r]));
    }
  }
}

// Shared memory of one expert block.
template <u32 MAX_R>
struct __align__(16) Space {
  float buf[WARPS][MAX_R][128];
  u32 words[WARPS][256];
  float vec[MAX_R][128];
  u32 rows[MAX_R];
  u64 at[MAX_R];
};

// One expert block's work.
struct Job {
  bytes gate, up, down;
  u32 m, c, n, hidden;
  const float *gate_su, *up_su, *gate_sv, *up_sv, *down_su, *signs;
};

// Inner chunk `c` of one expert, then its down part.
template <u32 K, u32 BOOK, u32 MAX_R>
static __device__ void expert(
    const Job& j, const float* x, float* partial, Space<MAX_R>& sp) {
  u32 warp = threadIdx.x / 32, lane = threadIdx.x % 32, n = MAX_R == 1 ? 1u : j.n, c = j.c;
  trellis::lane_map l = trellis::map_lane<K, BOOK>(lane);
  Mat D = matrix(j.down);
  // Even warps take the gate, odd warps the up.
  u32 h = warp % 2;
  Mat W = matrix(h ? j.up : j.gate);
  const float* su = h ? j.up_su : j.gate_su;
  float* b = &sp.buf[warp][0][0];
  float acc[8][4];
#pragma unroll
  for (u32 mt = 0; mt < 8; mt++) acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.0f;
  u32* held = sp.words[warp];
  // 16 byte loads need 16 byte aligned tiles.
  bool wide = ((u64)W.base & 15) == 0 && ((u64)D.base & 15) == 0;
  // Each step loads the next step's eight tiles first.
  u32 ibs = W.inp / 128;
  u32 steps = ibs > warp / 2 ? ((ibs - 1 - warp / 2) / (WARPS / 2) + 1) * 8 : 0;
  u32 got[8];
  if (steps > 0) load_tiles<K>(W, j.m, 8 * (warp / 2), c, lane, wide, got);
  for (u32 t = 0; t < steps; t++) {
    u32 q = warp / 2 + (t / 8) * (WARPS / 2), ks = t % 8;
    if (ks == 0) {
      __syncwarp();
      u32 at = q * 128 + (lane << 2);
      float4 sv = *(const float4*)(su + at);
      float4 sg = j.signs ? *(const float4*)(j.signs + at) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
      float sc[4] = {sv.x * sg.x, sv.y * sg.y, sv.z * sg.z, sv.w * sg.w};
      for (u32 r = 0; r < n; r++) {
        float4 xv = *(const float4*)(x + (u64)sp.rows[r] * j.hidden + at);
        float v[4] = {xv.x * sc[0], xv.y * sc[1], xv.z * sc[2], xv.w * sc[3]};
        trellis::wht128(v, trellis::INV_SQRT128);
        *(float4*)(b + r * 128 + (lane << 2)) = make_float4(v[0], v[1], v[2], v[3]);
      }
      __syncwarp();
    }
    __syncwarp();
    trellis::tile_store<K>(got, lane, wide, held);
    __syncwarp();
    if (t + 1 < steps) {
      u32 qn = warp / 2 + ((t + 1) / 8) * (WARPS / 2);
      load_tiles<K>(W, j.m, 8 * qn + (t + 1) % 8, c, lane, wide, got);
    }
    u32 f[2];
    trellis::act_frag(b, n, ks, lane, f);
#pragma unroll
    for (u32 mt = 0; mt < 8; mt++) {
      u32 a[4];
      trellis::decode_frag<K, BOOK>(held + mt * 8 * K, l, a);
      trellis::mma_m16n8k16(acc[mt], a, f);
    }
  }
  __syncwarp();
  // Each warp's sums, then SwiGLU and input turn in `vec`.
  trellis::store_acc(acc, n, lane, b);
  prim::block_sync();
  if (warp < n) {
    float g[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float u[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (u32 w = 0; w < WARPS; w += 2) {
      float4 gp = *(const float4*)(&sp.buf[w][warp][lane << 2]);
      float4 up = *(const float4*)(&sp.buf[w + 1][warp][lane << 2]);
      g[0] += gp.x;
      g[1] += gp.y;
      g[2] += gp.z;
      g[3] += gp.w;
      u[0] += up.x;
      u[1] += up.y;
      u[2] += up.z;
      u[3] += up.w;
    }
    trellis::wht128(g, trellis::INV_SQRT128);
    trellis::wht128(u, trellis::INV_SQRT128);
    u32 o = c * 128 + (lane << 2);
    float4 gsv = *(const float4*)(j.gate_sv + o);
    float4 usv = *(const float4*)(j.up_sv + o);
    float4 dsu = *(const float4*)(j.down_su + o);
    float gv[4] = {g[0] * gsv.x, g[1] * gsv.y, g[2] * gsv.z, g[3] * gsv.w};
    float uv[4] = {u[0] * usv.x, u[1] * usv.y, u[2] * usv.z, u[3] * usv.w};
    float ds[4] = {dsu.x, dsu.y, dsu.z, dsu.w};
    float a[4];
#pragma unroll
    for (u32 i = 0; i < 4; i++) a[i] = (gv[i] / (1.0f + __expf(-gv[i]))) * uv[i] * ds[i];
    trellis::wht128(a, trellis::INV_SQRT128);
    *(float4*)(&sp.vec[warp][lane << 2]) = make_float4(a[0], a[1], a[2], a[3]);
  }
  prim::block_sync();
  u32 obs = D.out / 128;
  u32 blocks = obs > warp ? ((obs - 1 - warp) / WARPS + 1) * 8 : 0;
  float out[8][4];
  if (blocks > 0) load_tiles<K>(D, j.m, 8 * c, warp, lane, wide, got);
  u32 r = lane % 4, g = lane / 4;
  for (u32 t = 0; t < blocks; t++) {
    u32 ob = warp + (t / 8) * WARPS, ks = t % 8;
    if (ks == 0) {
#pragma unroll
      for (u32 mt = 0; mt < 8; mt++) out[mt][0] = out[mt][1] = out[mt][2] = out[mt][3] = 0.0f;
    }
    __syncwarp();
    trellis::tile_store<K>(got, lane, wide, held);
    __syncwarp();
    if (t + 1 < blocks) {
      load_tiles<K>(D, j.m, 8 * c + (t + 1) % 8, warp + ((t + 1) / 8) * WARPS, lane, wide, got);
    }
    u32 f[2];
    trellis::act_frag(&sp.vec[0][0], n, ks, lane, f);
#pragma unroll
    for (u32 mt = 0; mt < 8; mt++) {
      u32 a[4];
      trellis::decode_frag<K, BOOK>(held + mt * 8 * K, l, a);
      trellis::mma_m16n8k16(out[mt], a, f);
    }
    if (ks == 7 && r < n) {
      float* to = partial + sp.at[r] + ob * 128;
#pragma unroll
      for (u32 mt = 0; mt < 8; mt++) {
        to[mt * 16 + g] = out[mt][0] + out[mt][1];
        to[mt * 16 + g + 8] = out[mt][2] + out[mt][3];
      }
    }
  }
}

template <u32 BOOK, u32 MAX_R>
static __device__ void expert_k(
    u32 k, const Job& j, const float* x, float* partial, Space<MAX_R>& sp) {
  if (k == 2) expert<2, BOOK, MAX_R>(j, x, partial, sp);
  if (k == 3) expert<3, BOOK, MAX_R>(j, x, partial, sp);
  if (k == 4) expert<4, BOOK, MAX_R>(j, x, partial, sp);
}

template <u32 MAX_R>
static __device__ void run_experts(
    const float* x, const float* scores, const float* bias, const float* signs_in,
    const float* gate_su, bytes gate, const float* gate_sv,
    const float* up_su, bytes up, const float* up_sv,
    const float* down_su, bytes down,
    bytes shared_gate, const float* shared_gate_su, const float* shared_gate_sv,
    bytes shared_up, const float* shared_up_su, const float* shared_up_sv,
    bytes shared_down, const float* shared_down_su,
    float* partial, u32 rows, u32 hidden, u32 ffn, u32 experts, u32 picks, u32 k, u32 shared_k,
    u32 book, u32 shared_book, u32 first, u32 count, u32 shared_ffn, Space<MAX_R>& sp) {
  __shared__ u32 ids[MAX_R][PICKS];
  __shared__ float gates[MAX_R][PICKS];
  __shared__ u32 chosen;
  u32 warp = threadIdx.x / 32, c = blockIdx.x;
  u32 groups = MAX_R == 1 ? 1u : (rows + MAX_R - 1) / MAX_R;
  u32 per = gridDim.y / groups;
  u32 group = MAX_R == 1 ? 0u : blockIdx.y / per;
  u32 slot = MAX_R == 1 ? blockIdx.y : blockIdx.y % per;
  u32 base = group * MAX_R, n = MAX_R == 1 ? 1u : min(rows - base, MAX_R);
  u32 chunks = ffn / 128;
  Job j;
  j.c = c;
  j.hidden = hidden;
  j.n = 0;
  if (slot == 0) {
    if (c >= shared_ffn / 128) return;
    j.gate = shared_gate;
    j.up = shared_up;
    j.down = shared_down;
    j.m = 0;
    j.gate_su = shared_gate_su;
    j.up_su = shared_up_su;
    j.gate_sv = shared_gate_sv;
    j.up_sv = shared_up_sv;
    j.down_su = shared_down_su;
    j.signs = 0;
    for (u32 r = 0; r < n; r++) {
      if (threadIdx.x == 0) {
        sp.rows[j.n] = base + r;
        sp.at[j.n] = (((u64)(base + r) * (picks + 1) + picks) * chunks + c) * hidden;
      }
      j.n++;
    }
    k = shared_k;
    book = shared_book;
    prim::block_sync();
  } else if constexpr (MAX_R == 1) {
    if (warp == 0) route(scores, bias, experts, picks, ids[0], gates[0]);
    if (threadIdx.x == 0) {
      u32 left = slot - 1, found = experts;
      for (u32 p = 0; p < picks; p++) {
        u32 cand = ids[0][p];
        if (cand >= first && cand < first + count) {
          if (left == 0) {
            found = cand;
            sp.rows[0] = 0;
            sp.at[0] = ((u64)p * chunks + c) * hidden;
            break;
          }
          left--;
        }
      }
      chosen = found;
    }
    prim::block_sync();
    u32 e = chosen;
    if (e == experts) return;
    j.gate = gate;
    j.up = up;
    j.down = down;
    j.m = e - first;
    j.gate_su = gate_su + (u64)e * hidden;
    j.up_su = up_su + (u64)e * hidden;
    j.gate_sv = gate_sv + (u64)e * ffn;
    j.up_sv = up_sv + (u64)e * ffn;
    j.down_su = down_su + (u64)e * ffn;
    j.signs = signs_in;
    j.n = 1;
  } else {
    __shared__ u32 mask[64];
    if (warp < n) {
      route(scores + (u64)(base + warp) * experts, bias, experts, picks, ids[warp], gates[warp]);
    }
    for (u32 i = threadIdx.x; i < 64; i += blockDim.x) mask[i] = 0u;
    prim::block_sync();
    for (u32 i = threadIdx.x; i < n * picks; i += blockDim.x) {
      u32 e = ids[i / picks][i % picks];
      if (e >= first && e < first + count) atomicOr(&mask[e / 32], 1u << (e % 32));
    }
    prim::block_sync();
    if (threadIdx.x == 0) {
      // The `slot - 1`-th picked expert, or `experts` if none.
      u32 left = slot - 1, e = experts;
      for (u32 i = 0; i < 64 && e == experts; i++) {
        u32 bits = mask[i], have = __popc(bits);
        if (left < have) {
          for (u32 s = 0; s < left; s++) bits &= bits - 1;
          e = 32 * i + __ffs(bits) - 1;
        } else {
          left -= have;
        }
      }
      chosen = e;
    }
    prim::block_sync();
    u32 e = chosen;
    if (e == experts) return;
    j.gate = gate;
    j.up = up;
    j.down = down;
    j.m = e - first;
    j.gate_su = gate_su + (u64)e * hidden;
    j.up_su = up_su + (u64)e * hidden;
    j.gate_sv = gate_sv + (u64)e * ffn;
    j.up_sv = up_sv + (u64)e * ffn;
    j.down_su = down_su + (u64)e * ffn;
    j.signs = signs_in;
    for (u32 r = 0; r < n; r++) {
      for (u32 p = 0; p < picks; p++) {
        if (ids[r][p] != e) continue;
        if (threadIdx.x == 0) {
          sp.rows[j.n] = base + r;
          sp.at[j.n] = (((u64)(base + r) * (picks + 1) + p) * chunks + c) * hidden;
        }
        j.n++;
      }
    }
    prim::block_sync();
  }
  if (book) {
    expert_k<trellis::BOOK_MUL1, MAX_R>(k, j, x, partial, sp);
  } else {
    expert_k<trellis::BOOK_MCG, MAX_R>(k, j, x, partial, sp);
  }
}

// Single-row decode experts with 12.5 KiB shared memory.
extern "C" __global__ __launch_bounds__(256, 2) void glm53_experts1(
    const float* x, const float* scores, const float* bias, const float* signs_in,
    const float* gate_su, bytes gate, const float* gate_sv,
    const float* up_su, bytes up, const float* up_sv,
    const float* down_su, bytes down,
    bytes shared_gate, const float* shared_gate_su, const float* shared_gate_sv,
    bytes shared_up, const float* shared_up_su, const float* shared_up_sv,
    bytes shared_down, const float* shared_down_su,
    float* partial, u32 rows, u32 hidden, u32 ffn, u32 experts, u32 picks, u32 k, u32 shared_k,
    u32 book, u32 shared_book, u32 first, u32 count, u32 shared_ffn) {
  __shared__ Space<1> sp;
  run_experts<1>(x, scores, bias, signs_in, gate_su, gate, gate_sv, up_su, up, up_sv, down_su,
      down, shared_gate, shared_gate_su, shared_gate_sv, shared_up, shared_up_su, shared_up_sv,
      shared_down, shared_down_su, partial, rows, hidden, ffn, experts, picks, k, shared_k, book,
      shared_book, first, count, shared_ffn, sp);
}

// Writes `partial[row][pick][chunk][hidden]`. Pick `picks` is shared.
extern "C" __global__ __launch_bounds__(256, 3) void glm53_experts(
    const float* x, const float* scores, const float* bias, const float* signs_in,
    const float* gate_su, bytes gate, const float* gate_sv,
    const float* up_su, bytes up, const float* up_sv,
    const float* down_su, bytes down,
    bytes shared_gate, const float* shared_gate_su, const float* shared_gate_sv,
    bytes shared_up, const float* shared_up_su, const float* shared_up_sv,
    bytes shared_down, const float* shared_down_su,
    float* partial, u32 rows, u32 hidden, u32 ffn, u32 experts, u32 picks, u32 k, u32 shared_k,
    u32 book, u32 shared_book, u32 first, u32 count, u32 shared_ffn) {
  __shared__ Space<ROWS> sp;
  run_experts<ROWS>(x, scores, bias, signs_in, gate_su, gate, gate_sv, up_su, up, up_sv, down_su,
      down, shared_gate, shared_gate_su, shared_gate_sv, shared_up, shared_up_su, shared_up_sv,
      shared_down, shared_down_su, partial, rows, hidden, ffn, experts, picks, k, shared_k, book,
      shared_book, first, count, shared_ffn, sp);
}

// Each row's sum over picks, after the output turn.
extern "C" __global__ void glm53_finish(
    const float* scores, const float* bias, const float* down_sv, const float* signs_out,
    const float* shared_down_sv, const float* partial,
    float* y, u32 rows, u32 hidden, u32 ffn, u32 experts, u32 picks, u32 first, u32 count,
    u32 shared_ffn) {
  (void)rows;
  __shared__ u32 ids[PICKS];
  __shared__ float gates[PICKS];
  __shared__ float wsum[4][128];
  u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
  u32 base = blockIdx.x * 128 + (lane << 2), row = blockIdx.y;
  if (warp == 0) route(scores + (u64)row * experts, bias, experts, picks, ids, gates);
  prim::block_sync();
  u32 chunks = ffn / 128;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (u32 p = warp; p <= picks; p += 4) {
    bool is_shared = p == picks;
    if (!is_shared && (ids[p] < first || ids[p] >= first + count)) continue;
    u32 upto = is_shared ? shared_ffn / 128 : chunks;
    const float* from = partial + ((u64)row * (picks + 1) + p) * chunks * hidden + base;
    float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 c = 0; c < upto; c++) {
      float4 v = *(const float4*)(from + (u64)c * hidden);
      s[0] += v.x;
      s[1] += v.y;
      s[2] += v.z;
      s[3] += v.w;
    }
    trellis::wht128(s, trellis::INV_SQRT128);
    if (is_shared) {
      float4 sv = *(const float4*)(shared_down_sv + base);
      acc[0] += s[0] * sv.x;
      acc[1] += s[1] * sv.y;
      acc[2] += s[2] * sv.z;
      acc[3] += s[3] * sv.w;
    } else {
      u32 e = ids[p];
      float gw = gates[p];
      float4 sv = *(const float4*)(down_sv + (u64)e * hidden + base);
      float4 sg = *(const float4*)(signs_out + base);
      acc[0] += gw * (s[0] * sv.x * sg.x);
      acc[1] += gw * (s[1] * sv.y * sg.y);
      acc[2] += gw * (s[2] * sv.z * sg.z);
      acc[3] += gw * (s[3] * sv.w * sg.w);
    }
  }
  *(float4*)(&wsum[warp][lane << 2]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
  prim::block_sync();
  if (warp == 0) {
    float4 o = *(const float4*)(&wsum[0][lane << 2]);
    for (u32 w = 1; w < 4; w++) {
      float4 q = *(const float4*)(&wsum[w][lane << 2]);
      o.x += q.x;
      o.y += q.y;
      o.z += q.z;
      o.w += q.w;
    }
    *(float4*)(y + (u64)row * hidden + base) = o;
  }
}
