// GLM-5.3's routed block: route, experts, then finish.

typedef unsigned int u32;
typedef unsigned long long u64;
typedef const unsigned char* bytes;

// Most picks per row, and rows one expert block takes.
#define PICKS 8
#define ROWS 4
#define WARPS 8
// The gate scale of the release.
#define ROUTE_SCALE 2.5f
// Each rotation of 128 channels scales by 1 / 128 here.
#define TURNED (1.0f / 128.0f)
// Trellis header bytes, and the side word of input entries.
#define HEAD 16
#define INPUT_SIDE 2u
#define MCG 0xCBAC1FEDu
#define MUL1 0x83DCD12Du

static __device__ inline u32 word(bytes p, u32 at) {
  return *(const u32*)(p + at);
}

static __device__ inline float half(u32 h) {
  float r;
  unsigned short s = (unsigned short)h;
  asm("{.reg .b16 t; mov.b16 t, %1; cvt.f32.f16 %0, t;}" : "=f"(r) : "h"(s));
  return r;
}

// `lo` and `hi` as f16, rounded to nearest even.
static __device__ inline u32 halves(float lo, float hi) {
  u32 r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

// Two f16 values rounded to E4M3, saturated at 448.
static __device__ inline u32 narrowed(u32 h) {
  u32 r;
  asm("{.reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.rn.f16x2.e4m3x2 %0, t;}"
      : "=r"(r) : "r"(h));
  return r;
}

static __device__ inline u32 perm(u32 a, u32 b, u32 s) {
  u32 r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
  return r;
}

// The 32 bits from bit `s` of `hi` then `lo`.
static __device__ inline u32 funnel(u32 lo, u32 hi, u32 s) {
  u32 r;
  asm("shf.l.clamp.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(s));
  return r;
}

static __device__ inline u32 load(bytes p) {
  u32 r;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(p));
  return r;
}

// Two states' weights as f16. `BOOK`: 0 mcg, 1 mul1.
template <u32 BOOK>
static __device__ inline u32 pair(u32 a, u32 b) {
  if (BOOK == 1) {
    u32 x = a * MUL1, y = b * MUL1;
    u32 sx, sy;
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(sx) : "r"(x), "r"(0x01010101u), "r"(0u));
    asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(sy) : "r"(y), "r"(0x01010101u), "r"(0u));
    // Exact in f32: one rounding to f16.
    float scale = half(0x1EEEu), offset = half(0xC931u);
    return narrowed(halves((float)(1024u + sx) * scale + offset, (float)(1024u + sy) * scale + offset));
  }
  u32 x = ((a * MCG) & 0x8FFF8FFFu) ^ 0x3B603B60u;
  u32 y = ((b * MCG) & 0x8FFF8FFFu) ^ 0x3B603B60u;
  // The f16 sum of two halves is exact in f32.
  u32 s;
  asm("add.rn.f16x2 %0, %1, %2;" : "=r"(s) : "r"(perm(x, y, 0x5410)), "r"(perm(x, y, 0x7632)));
  return narrowed(s);
}

// A tile into this lane's mma A fragment.
template <u32 K, u32 BOOK>
static __device__ inline void decode(u32 w, u32 lane, u32* a) {
  const u32 words = 8 * K, span = (16 + K - 1) / K, bits = 256 * K;
  // A state spans `span` codes ending at its position.
  u32 first = ((8 * lane + 257 - span) % 256) * K + span * K - 16;
  u32 q = first / 32;
  u32 w0 = __shfl_sync(0xFFFFFFFFu, w, q);
  u32 w1 = __shfl_sync(0xFFFFFFFFu, w, (q + 1) % words);
  u32 w2 = __shfl_sync(0xFFFFFFFFu, w, (q + 2) % words);
  u32 st[8];
#pragma unroll
  for (u32 s = 0; s < 8; s++) {
    u32 at = ((8 * lane + s + 257 - span) % 256) * K + span * K - 16;
    u32 rel = (at + bits - 32 * q) % bits;
    u32 hi = rel >= 32 ? w1 : w0, lo = rel >= 32 ? w2 : w1;
    st[s] = funnel(lo, hi, rel % 32) >> 16;
  }
  a[0] = pair<BOOK>(st[0], st[1]);
  a[1] = pair<BOOK>(st[4], st[5]);
  a[2] = pair<BOOK>(st[2], st[3]);
  a[3] = pair<BOOK>(st[6], st[7]);
}

// `c += a b` for one 16 by 8 output tile.
static __device__ inline void mma(float* c, const u32* a, const u32* b) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A code set's tiles, matrix shape and side.
struct Mat {
  bytes p;
  u32 inp, out, side;
};

static __device__ inline Mat matrix(bytes p) {
  Mat w = {p + HEAD, word(p, 0), word(p, 4), word(p, 8)};
  return w;
}

// Tile of inputs `16 i`, outputs `16 o`, matrix `m`.
template <u32 K>
static __device__ inline bytes tile(const Mat& w, u32 m, u32 i, u32 o) {
  u32 t = w.side == INPUT_SIDE ? (i / 8) * (w.out / 2) + 64 * (o / 8) + 8 * (i % 8) + o % 8
                               : (o / 8) * (w.inp / 2) + 8 * i + o % 8;
  u64 first = (u64)m * (w.inp / 16) * (w.out / 16);
  return w.p + (first + t) * (32 * K);
}

static __device__ inline u32 lane_load(bytes p, u32 lane, u32 words) {
  return lane < words ? load(p + 4 * lane) : 0u;
}

// Walsh-Hadamard transform of `n` runs of 128, one warp.
static __device__ void turn_warp(float* v, u32 n) {
  u32 lane = threadIdx.x % 32;
  for (u32 len = 1; len < 128; len *= 2) {
    for (u32 j = lane; j < 64 * n; j += 32) {
      u32 q = j % 64;
      u32 i = (j / 64) * 128 + (q / len) * 2 * len + q % len;
      float a = v[i], b = v[i + len];
      v[i] = a + b;
      v[i + len] = a - b;
    }
    __syncwarp();
  }
}

// The same, by the whole block.
static __device__ void turn_block(float* v, u32 n) {
  for (u32 len = 1; len < 128; len *= 2) {
    for (u32 j = threadIdx.x; j < 64 * n; j += blockDim.x) {
      u32 q = j % 64;
      u32 i = (j / 64) * 128 + (q / len) * 2 * len + q % len;
      float a = v[i], b = v[i + len];
      v[i] = a + b;
      v[i + len] = a - b;
    }
    __syncthreads();
  }
}

// B fragment of step `ks`: rows as f16, then remainders.
static __device__ inline void fragment(const float* v, u32 n, u32 ks, u32 lane, u32* b) {
  u32 col = lane / 4, r = col / 2;
  if (r >= n) {
    b[0] = b[1] = 0u;
    return;
  }
  const float* at = v + r * 128 + 16 * ks + 2 * (lane % 4);
#pragma unroll
  for (u32 h = 0; h < 2; h++) {
    float x0 = at[8 * h], x1 = at[8 * h + 1];
    u32 hi = halves(x0, x1);
    b[h] = col % 2 ? halves(x0 - half(hi & 0xFFFFu), x1 - half(hi >> 16)) : hi;
  }
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

// `scores = sigmoid(x router)`. One warp per expert.
extern "C" __global__ void glm53_route(const float* x, const float* router, float* scores,
    u32 rows, u32 hidden, u32 experts) {
  u32 lane = threadIdx.x % 32;
  u32 e = blockIdx.x * WARPS + threadIdx.x / 32;
  if (e >= experts) return;
  const float* w = router + (u64)e * hidden;
  for (u32 r0 = 0; r0 < rows; r0 += 8) {
    float s[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 i = 4 * lane; i < hidden; i += 128) {
      float4 a = *(const float4*)(w + i);
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
struct Space {
  float buf[WARPS][2][ROWS][128];
  float vec[3][ROWS][128];
};

// What one expert block computes.
struct Job {
  bytes gate, up, down;
  u32 m, c, n, hidden;
  const float *gate_su, *up_su, *gate_sv, *up_sv, *down_su, *signs;
  u32 rows[ROWS];
  u64 at[ROWS];
};

// Inner chunk `c` of one expert, then its down part.
template <u32 K, u32 BOOK>
static __device__ void expert(const Job& j, const float* x, float* partial, Space& sp) {
  float(*buf)[2][ROWS][128] = sp.buf;
  float(*vec)[ROWS][128] = sp.vec;
  const u32 words = 8 * K;
  u32 warp = threadIdx.x / 32, lane = threadIdx.x % 32, n = j.n, c = j.c;
  Mat G = matrix(j.gate), U = matrix(j.up), D = matrix(j.down);
  float acc[2][8][4];
#pragma unroll
  for (u32 h = 0; h < 2; h++)
#pragma unroll
    for (u32 mt = 0; mt < 8; mt++) acc[h][mt][0] = acc[h][mt][1] = acc[h][mt][2] = acc[h][mt][3] = 0.0f;
  for (u32 q = warp; q < G.inp / 128; q += WARPS) {
    float* bg = &buf[warp][0][0][0];
    float* bu = &buf[warp][1][0][0];
    for (u32 r = 0; r < n; r++) {
      const float* row = x + (u64)j.rows[r] * j.hidden + q * 128;
      for (u32 i = lane; i < 128; i += 32) {
        u32 at = q * 128 + i;
        float v = row[i] * (j.signs ? j.signs[at] : 1.0f) * TURNED;
        bg[r * 128 + i] = v * j.gate_su[at];
        bu[r * 128 + i] = v * j.up_su[at];
      }
    }
    __syncwarp();
    turn_warp(bg, n);
    turn_warp(bu, n);
    for (u32 ks = 0; ks < 8; ks++) {
      u32 fg[2], fu[2], wg[8], wu[8];
      fragment(bg, n, ks, lane, fg);
      fragment(bu, n, ks, lane, fu);
#pragma unroll
      for (u32 mt = 0; mt < 8; mt++) {
        wg[mt] = lane_load(tile<K>(G, j.m, q * 8 + ks, c * 8 + mt), lane, words);
        wu[mt] = lane_load(tile<K>(U, j.m, q * 8 + ks, c * 8 + mt), lane, words);
      }
#pragma unroll
      for (u32 mt = 0; mt < 8; mt++) {
        u32 a[4];
        decode<K, BOOK>(wg[mt], lane, a);
        mma(acc[0][mt], a, fg);
        decode<K, BOOK>(wu[mt], lane, a);
        mma(acc[1][mt], a, fu);
      }
    }
    __syncwarp();
  }
  __syncthreads();
  // Each warp's sums, then their total, in `vec`.
  u32 r = lane % 4, g = lane / 4;
  if (r < n) {
#pragma unroll
    for (u32 h = 0; h < 2; h++)
#pragma unroll
      for (u32 mt = 0; mt < 8; mt++) {
        buf[warp][h][r][mt * 16 + g] = acc[h][mt][0] + acc[h][mt][1];
        buf[warp][h][r][mt * 16 + g + 8] = acc[h][mt][2] + acc[h][mt][3];
      }
  }
  __syncthreads();
  for (u32 i = threadIdx.x; i < 2 * ROWS * 128; i += blockDim.x) {
    float s = 0.0f;
    for (u32 w = 0; w < WARPS; w++) s += (&buf[w][0][0][0])[i];
    (&vec[0][0][0])[i] = s;
  }
  __syncthreads();
  turn_block(&vec[0][0][0], 2 * ROWS);
  for (u32 i = threadIdx.x; i < n * 128; i += blockDim.x) {
    u32 o = c * 128 + i % 128;
    float gate = (&vec[0][0][0])[i] * j.gate_sv[o];
    float up = (&vec[1][0][0])[i] * j.up_sv[o];
    (&vec[2][0][0])[i] = gate / (1.0f + expf(-gate)) * up * j.down_su[o] * TURNED;
  }
  __syncthreads();
  turn_block(&vec[2][0][0], ROWS);
  u32 fd[8][2];
#pragma unroll
  for (u32 ks = 0; ks < 8; ks++) fragment(&vec[2][0][0], n, ks, lane, fd[ks]);
  for (u32 ob = warp; ob < D.out / 128; ob += WARPS) {
    float out[8][4];
#pragma unroll
    for (u32 mt = 0; mt < 8; mt++) out[mt][0] = out[mt][1] = out[mt][2] = out[mt][3] = 0.0f;
    for (u32 ks = 0; ks < 8; ks++) {
      u32 w[8];
#pragma unroll
      for (u32 mt = 0; mt < 8; mt++) w[mt] = lane_load(tile<K>(D, j.m, c * 8 + ks, ob * 8 + mt), lane, words);
#pragma unroll
      for (u32 mt = 0; mt < 8; mt++) {
        u32 a[4];
        decode<K, BOOK>(w[mt], lane, a);
        mma(out[mt], a, fd[ks]);
      }
    }
    if (r < n) {
      float* to = partial + j.at[r] + ob * 128;
#pragma unroll
      for (u32 mt = 0; mt < 8; mt++) {
        to[mt * 16 + g] = out[mt][0] + out[mt][1];
        to[mt * 16 + g + 8] = out[mt][2] + out[mt][3];
      }
    }
  }
}

template <u32 BOOK>
static __device__ void expert_k(u32 k, const Job& j, const float* x, float* partial, Space& sp) {
  if (k == 2) expert<2, BOOK>(j, x, partial, sp);
  if (k == 3) expert<3, BOOK>(j, x, partial, sp);
  if (k == 4) expert<4, BOOK>(j, x, partial, sp);
}

// Writes `partial[row][pick][chunk][hidden]`. Pick `picks` is shared.
extern "C" __global__ void glm53_experts(
    const float* x, const float* scores, const float* bias, const float* signs_in,
    const float* gate_su, bytes gate, const float* gate_sv,
    const float* up_su, bytes up, const float* up_sv,
    const float* down_su, bytes down,
    bytes shared_gate, const float* shared_gate_su, const float* shared_gate_sv,
    bytes shared_up, const float* shared_up_su, const float* shared_up_sv,
    bytes shared_down, const float* shared_down_su,
    float* partial, u32 rows, u32 hidden, u32 ffn, u32 experts, u32 picks, u32 k, u32 shared_k,
    u32 book, u32 shared_book, u32 first, u32 count, u32 shared_ffn) {
  __shared__ u32 ids[ROWS][PICKS];
  __shared__ float gates[ROWS][PICKS];
  __shared__ u32 mask[64];
  __shared__ u32 chosen;
  __shared__ Space sp;
  u32 warp = threadIdx.x / 32, c = blockIdx.x;
  u32 groups = (rows + ROWS - 1) / ROWS, per = gridDim.y / groups;
  u32 group = blockIdx.y / per, slot = blockIdx.y % per;
  u32 base = group * ROWS, n = min(rows - base, ROWS);
  if (warp < n) route(scores + (u64)(base + warp) * experts, bias, experts, picks, ids[warp], gates[warp]);
  for (u32 i = threadIdx.x; i < 64; i += blockDim.x) mask[i] = 0u;
  __syncthreads();
  for (u32 i = threadIdx.x; i < n * picks; i += blockDim.x) {
    u32 e = ids[i / picks][i % picks];
    if (e >= first && e < first + count) atomicOr(&mask[e / 32], 1u << (e % 32));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // The `slot - 1`-th picked expert, or `experts` if none.
    u32 left = slot - 1, e = experts;
    for (u32 i = 0; slot > 0 && i < 64 && e == experts; i++) {
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
  __syncthreads();
  u32 e = chosen;
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
      j.rows[j.n] = base + r;
      j.at[j.n++] = (((u64)(base + r) * (picks + 1) + picks) * chunks + c) * hidden;
    }
    if (shared_book) {
      expert_k<1>(shared_k, j, x, partial, sp);
    } else {
      expert_k<0>(shared_k, j, x, partial, sp);
    }
    return;
  }
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
      j.rows[j.n] = base + r;
      j.at[j.n++] = (((u64)(base + r) * (picks + 1) + p) * chunks + c) * hidden;
    }
  }
  if (book) {
    expert_k<1>(k, j, x, partial, sp);
  } else {
    expert_k<0>(k, j, x, partial, sp);
  }
}

// Each row's sum over picks, after the output turn.
extern "C" __global__ void glm53_finish(
    const float* scores, const float* bias, const float* down_sv, const float* signs_out,
    const float* shared_down_sv, const float* partial,
    float* y, u32 rows, u32 hidden, u32 ffn, u32 experts, u32 picks, u32 first, u32 count,
    u32 shared_ffn) {
  __shared__ u32 ids[PICKS];
  __shared__ float gates[PICKS];
  __shared__ float z[PICKS + 1][128];
  u32 o = threadIdx.x, at = blockIdx.x * 128 + o, row = blockIdx.y;
  if (threadIdx.x < 32) route(scores + (u64)row * experts, bias, experts, picks, ids, gates);
  __syncthreads();
  u32 chunks = ffn / 128;
  for (u32 p = 0; p <= picks; p++) {
    bool mine = p == picks || (ids[p] >= first && ids[p] < first + count);
    u32 upto = p == picks ? shared_ffn / 128 : chunks;
    const float* from = partial + ((u64)row * (picks + 1) + p) * chunks * hidden + at;
    float s = 0.0f;
    for (u32 c = 0; mine && c < upto; c++) s += from[(u64)c * hidden];
    z[p][o] = s;
  }
  __syncthreads();
  turn_block(&z[0][0], picks + 1);
  float out = z[picks][o] * shared_down_sv[at];
  for (u32 p = 0; p < picks; p++) {
    u32 e = ids[p];
    if (e < first || e >= first + count) continue;
    out += gates[p] * (z[p][o] * down_sv[(u64)e * hidden + at] * signs_out[at]);
  }
  y[(u64)row * hidden + at] = out;
}
