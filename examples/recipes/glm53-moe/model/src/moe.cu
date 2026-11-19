// GLM-5.3's routed block. One thread block per row.

typedef unsigned int u32;
typedef unsigned long long u64;
typedef const unsigned char* bytes;

// The widest stream, inner width and pick count.
#define WIDE 6144
#define NARROW 2048
#define PICKS 8
// The gate scale of the release.
#define ROUTE_SCALE 2.5f
// Two rotations of 128 channels scale by 1 / 128.
#define TURNED (1.0f / 128.0f)
// Trellis header bytes, and the side word of input entries.
#define HEAD 16
#define INPUT_SIDE 2u
#define MCG 0xCBAC1FEDu

static __device__ inline u32 word(bytes p, u32 at) {
  return *(const u32*)(p + at);
}

static __device__ inline float half(u32 h) {
  float r;
  unsigned short s = (unsigned short)h;
  asm("{.reg .b16 t; mov.b16 t, %1; cvt.f32.f16 %0, t;}" : "=f"(r) : "h"(s));
  return r;
}

// `x` rounded to nearest even, `e` and `m` bits.
static __device__ inline float rounded(u32 e, u32 m, float top, float x) {
  float a = x < 0.0f ? -x : x;
  float sign = x < 0.0f ? -1.0f : 1.0f;
  float steps = __uint_as_float((125u + (1u << (e - 1)) + m) << 23);
  float small = (a * steps + 8388608.0f) - 8388608.0f;
  u32 bits = __float_as_uint(a);
  u32 nudge = ((1u << (22 - m)) - 1) + ((bits >> (23 - m)) & 1);
  float r = __uint_as_float((bits + nudge) & ~((1u << (23 - m)) - 1));
  float big = r > top ? top : r;
  float least = __uint_as_float((129u - (1u << (e - 1))) << 23);
  return sign * (a < least ? small / steps : big);
}

// 32 stream bits of a tile from bit `s`, wrapping.
static __device__ inline u32 window(bytes tile, u32 k, u32 s) {
  u32 q = s >> 5, shift = s & 31;
  u32 high = word(tile, 4 * (q % (8 * k)));
  if (shift == 0) return high;
  return (high << shift) | (word(tile, 4 * ((q + 1) % (8 * k))) >> (32 - shift));
}

// `T[at][row]` of the `k`-bit codes at `p`.
static __device__ inline float weight(bytes p, u32 k, u32 row, u32 at) {
  u32 inp = word(p, 0), out = word(p, 4);
  u32 col = at % out;
  u64 first = (u64)(at / out) * (inp / 16) * (out / 16);
  u32 t = word(p, 8) == INPUT_SIDE
      ? (row / 128) * (out / 2) + 64 * (col / 128) + 8 * ((row % 128) / 16) + (col % 128) / 16
      : (col / 128) * (inp / 2) + 8 * (row / 16) + (col % 128) / 16;
  bytes tile = p + (HEAD + (first + t) * (32 * k));
  u32 r = row % 16, c = col % 16;
  u32 j = 8 * (4 * (c % 8) + (r % 8) / 2) + 4 * (c >= 8) + 2 * (r >= 8) + (r % 2);
  // A state spans `span` codes ending at `j`.
  u32 span = (16 + k - 1) / k;
  u32 state = window(tile, k, ((j + 257 - span) % 256) * k + (span * k - 16)) >> 16;
  u32 x = (state & 0xFFFFu) * MCG;
  u32 y = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
  float v = rounded(5, 10, 65504.0f, half(y & 0xFFFFu) + half(y >> 16));
  return rounded(4, 3, 448.0f, v);
}

// The Walsh-Hadamard transform of each 128 values of `v`.
static __device__ void turn(float* v, u32 n) {
  for (u32 len = 1; len < 128; len *= 2) {
    for (u32 j = threadIdx.x; j < n / 2; j += blockDim.x) {
      u32 q = j % 64;
      u32 i = (j / 64) * 128 + (q / len) * 2 * len + q % len;
      float a = v[i], b = v[i + len];
      v[i] = a + b;
      v[i + len] = a - b;
    }
    __syncthreads();
  }
}

// `to = turn(x * signs * scale)`. `signs` may be null.
static __device__ void enter(const float* x, const float* signs, const float* scale, float* to, u32 n) {
  for (u32 i = threadIdx.x; i < n; i += blockDim.x) {
    to[i] = x[i] * scale[i] * (signs ? signs[i] : 1.0f);
  }
  __syncthreads();
  turn(to, n);
}

// `to = turn(T from)`, `T` matrix `e` at `w`.
static __device__ void product(bytes w, u32 k, u32 e, const float* from, float* to) {
  u32 inp = word(w, 0), out = word(w, 4);
  for (u32 o = threadIdx.x; o < out; o += blockDim.x) {
    float s = 0.0f;
    for (u32 i = 0; i < inp; i++) s += weight(w, k, i, e * out + o) * from[i];
    to[o] = s;
  }
  __syncthreads();
  turn(to, out);
}

// One expert's three matrices: codes, input scales, output scales.
struct Expert {
  bytes gate, up, down;
  const float *gate_su, *up_su, *down_su;
  const float *gate_sv, *up_sv;
  u32 k, e;
};

// `a = down(silu(gate(x)) * up(x))`, before the last scale.
static __device__ void apply(const Expert& w, const float* x, const float* signs,
    u32 hidden, u32 ffn, float* a, float* g, float* u) {
  enter(x, signs, w.gate_su, a, hidden);
  product(w.gate, w.k, w.e, a, g);
  enter(x, signs, w.up_su, a, hidden);
  product(w.up, w.k, w.e, a, u);
  for (u32 o = threadIdx.x; o < ffn; o += blockDim.x) {
    float gate = g[o] * w.gate_sv[o] * TURNED;
    float inner = gate / (1.0f + expf(-gate)) * (u[o] * w.up_sv[o] * TURNED);
    g[o] = inner * w.down_su[o];
  }
  __syncthreads();
  turn(g, ffn);
  product(w.down, w.k, w.e, g, a);
}

// The best `picks` of `score + bias`. Ties take the lower.
static __device__ void route(const float* score, const float* bias, u32 experts, u32 picks,
    u32* ids, float* gates) {
  float sum = 0.0f;
  for (u32 p = 0; p < picks; p++) {
    u32 best = experts;
    float most = 0.0f;
    for (u32 e = 0; e < experts; e++) {
      float v = score[e] + bias[e];
      bool taken = false;
      for (u32 q = 0; q < p; q++) taken = taken || ids[q] == e;
      if (!taken && (best == experts || v > most)) {
        best = e;
        most = v;
      }
    }
    ids[p] = best;
    sum += score[best];
  }
  for (u32 p = 0; p < picks; p++) gates[p] = score[ids[p]] / sum * ROUTE_SCALE;
}

extern "C" __global__ void glm53_moe(
    const float* x, const float* router, const float* bias, const float* signs_in,
    const float* gate_su, bytes gate, const float* gate_sv,
    const float* up_su, bytes up, const float* up_sv,
    const float* down_su, bytes down, const float* down_sv,
    const float* signs_out,
    bytes shared_gate, const float* shared_gate_su, const float* shared_gate_sv,
    bytes shared_up, const float* shared_up_su, const float* shared_up_sv,
    bytes shared_down, const float* shared_down_su, const float* shared_down_sv,
    float* y, u32 rows, u32 hidden, u32 ffn, u32 experts, u32 picks, u32 k, u32 shared_k) {
  __shared__ float a[WIDE];
  __shared__ float g[NARROW];
  __shared__ float u[NARROW];
  __shared__ u32 ids[PICKS];
  __shared__ float gates[PICKS];
  u32 row = blockIdx.y;
  if (row >= rows) return;
  const float* h = x + (u64)row * hidden;
  float* out = y + (u64)row * hidden;
  for (u32 e = threadIdx.x; e < experts; e += blockDim.x) {
    float s = 0.0f;
    for (u32 i = 0; i < hidden; i++) s += h[i] * router[(u64)e * hidden + i];
    g[e] = 1.0f / (1.0f + expf(-s));
  }
  __syncthreads();
  if (threadIdx.x == 0) route(g, bias, experts, picks, ids, gates);
  __syncthreads();
  // The shared expert first: it reads the plain row.
  Expert all = {shared_gate, shared_up, shared_down, shared_gate_su, shared_up_su,
      shared_down_su, shared_gate_sv, shared_up_sv, shared_k, 0};
  apply(all, h, 0, hidden, ffn, a, g, u);
  for (u32 o = threadIdx.x; o < hidden; o += blockDim.x) {
    out[o] = a[o] * shared_down_sv[o] * TURNED;
  }
  __syncthreads();
  for (u32 p = 0; p < picks; p++) {
    u32 e = ids[p];
    Expert one = {gate, up, down, gate_su + (u64)e * hidden, up_su + (u64)e * hidden,
        down_su + (u64)e * ffn, gate_sv + (u64)e * ffn, up_sv + (u64)e * ffn, k, e};
    apply(one, h, signs_in, hidden, ffn, a, g, u);
    const float* sv = down_sv + (u64)e * hidden;
    for (u32 o = threadIdx.x; o < hidden; o += blockDim.x) {
      out[o] += gates[p] * (a[o] * sv[o] * TURNED * signs_out[o]);
    }
    __syncthreads();
  }
}
