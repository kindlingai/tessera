// Single-launch in-place Metal RoPE with register-held trig.

#include <metal_stdlib>
using namespace metal;

static inline float rp_bf16_to_f32(ushort u) {
    return as_type<float>(uint(u) << 16u);
}

static inline ushort rp_f32_to_bf16(float f) {
    uint u = as_type<uint>(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) {
        return ushort((u >> 16u) | 0x0040u);
    }
    uint bias = 0x7FFFu + ((u >> 16u) & 1u);
    return ushort((u + bias) >> 16u);
}

template <uint FMT, bool ADJACENT>
static inline void rp_rotate_pair(
    device uchar* x,
    ulong base,
    uint pair,
    uint pairs,
    float c,
    float sn) {
    if constexpr (FMT == 0u) {
        device float* xf = reinterpret_cast<device float*>(x) + base;
        if constexpr (ADJACENT) {
            uint a = 2u * pair;
            float2 v = float2(*reinterpret_cast<device packed_float2*>(xf + a));
            float2 o = float2(v.x * c - v.y * sn, v.x * sn + v.y * c);
            *reinterpret_cast<device packed_float2*>(xf + a) = packed_float2(o);
        } else {
            float x0 = xf[pair];
            float x1 = xf[pairs + pair];
            xf[pair] = x0 * c - x1 * sn;
            xf[pairs + pair] = x0 * sn + x1 * c;
        }
    } else if constexpr (FMT == 1u) {
        device ushort* xu = reinterpret_cast<device ushort*>(x) + base;
        uint a = ADJACENT ? (2u * pair) : pair;
        uint b = ADJACENT ? (a + 1u) : (pairs + pair);
        float x0 = rp_bf16_to_f32(xu[a]);
        float x1 = rp_bf16_to_f32(xu[b]);
        xu[a] = rp_f32_to_bf16(x0 * c - x1 * sn);
        xu[b] = rp_f32_to_bf16(x0 * sn + x1 * c);
    } else {
        device half* xh = reinterpret_cast<device half*>(x) + base;
        uint a = ADJACENT ? (2u * pair) : pair;
        uint b = ADJACENT ? (a + 1u) : (pairs + pair);
        float x0 = float(xh[a]);
        float x1 = float(xh[b]);
        xh[a] = half(x0 * c - x1 * sn);
        xh[b] = half(x0 * sn + x1 * c);
    }
}

template <uint FMT, bool ADJACENT>
static inline void rope_body(
    device uchar* x,
    const device uint* pos_buf,
    uint positions,
    uint heads,
    uint width,
    uint turned,
    uint heads_per_tg,
    float theta,
    uint2 tgid,
    uint tid) {
    uint p = tgid.y;
    if (p >= positions || turned == 0u) {
        return;
    }
    uint h0 = tgid.x * heads_per_tg;
    if (h0 >= heads) {
        return;
    }
    uint h_end = min(h0 + heads_per_tg, heads);
    uint pairs = width >> 1u;
    float pos_f = float(pos_buf[p]);

    if (turned <= 128u) {
        uint shift = (turned <= 32u) ? 5u : ((turned <= 64u) ? 6u : 7u);
        uint span = 1u << shift;
        uint pair = tid & (span - 1u);
        if (pair >= turned) {
            return;
        }
        uint h_off = tid >> shift;
        uint h_step = 128u >> shift;
        float angle = pos_f * precise::pow(theta, -float(2u * pair) / float(width));
        float c = 0.0f;
        float sn = precise::sincos(angle, c);
        ulong base = ulong(p * heads + h0 + h_off) * ulong(width);
        ulong h_stride = ulong(h_step) * ulong(width);
        for (uint h = h0 + h_off; h < h_end; h += h_step) {
            rp_rotate_pair<FMT, ADJACENT>(x, base, pair, pairs, c, sn);
            base += h_stride;
        }
        return;
    }

    uint nh = h_end - h0;
    uint total = nh * turned;
    for (uint idx = tid; idx < total; idx += 128u) {
        uint lh = idx / turned;
        uint pair = idx - lh * turned;
        float angle = pos_f * precise::pow(theta, -float(2u * pair) / float(width));
        float c = 0.0f;
        float sn = precise::sincos(angle, c);
        ulong base = ulong(p * heads + h0 + lh) * ulong(width);
        rp_rotate_pair<FMT, ADJACENT>(x, base, pair, pairs, c, sn);
    }
}

#define DEFINE_ROPE_MTL(NAME, FMT, ADJ)                                              \
    kernel void NAME(                                                                \
        device uchar* x [[buffer(0)]], const device uint* pos [[buffer(1)]],         \
        constant uint& positions [[buffer(2)]], constant uint& heads [[buffer(3)]],  \
        constant uint& width [[buffer(4)]], constant uint& turned [[buffer(5)]],     \
        constant uint& heads_per_tg [[buffer(6)]], constant float& theta [[buffer(7)]], \
        uint2 tgid [[threadgroup_position_in_grid]],                                 \
        uint tid [[thread_index_in_threadgroup]]) {                                  \
        rope_body<FMT, ADJ>(                                                         \
            x, pos, positions, heads, width, turned, heads_per_tg, theta, tgid, tid); \
    }

DEFINE_ROPE_MTL(rope_mtl_f32_half, 0u, false)
DEFINE_ROPE_MTL(rope_mtl_f32_adj, 0u, true)
DEFINE_ROPE_MTL(rope_mtl_bf16_half, 1u, false)
DEFINE_ROPE_MTL(rope_mtl_bf16_adj, 1u, true)
DEFINE_ROPE_MTL(rope_mtl_f16_half, 2u, false)
DEFINE_ROPE_MTL(rope_mtl_f16_adj, 2u, true)
