#include <metal_stdlib>

using namespace metal;

constant uint BM_THREADS = 128u;
constant uint BM_NC = 4u;

enum BmKind : uint {
    BM_MXFP8 = 0u,
    BM_Q8_0 = 1u,
    BM_INT4_G32 = 2u,
    BM_Q4_0 = 3u,
};

// Packed uint2 E4M3 to half4 decode.
static inline half4 bm_e4m3_4(uchar4 raw) {
    uint2 u = uint2(uint(raw.x) | (uint(raw.y) << 16u), uint(raw.z) | (uint(raw.w) << 16u));
    uint2 s = (u & 0x00800080u) << 8u;
    uint2 m = u & 0x007F007Fu;
    half4 norm = as_type<half4>(s | ((m << 7u) + 0x20002000u));
    half4 sub = as_type<half4>(s | 0x40004000u | m) - as_type<half4>(s | 0x40004000u);
    return select(norm, sub, abs(norm) < half4(0.015625h));
}

template <uint KIND>
static inline void bm_decode8(
    device const uchar *b,
    uint lane_sub,
    thread float &scale,
    thread half4 &wa,
    thread half4 &wb) {
    if constexpr (KIND == BM_MXFP8) {
        scale = as_type<float>(uint(b[0]) << 23u);
        uint sub = lane_sub * 8u;
        uchar4 c0 = uchar4(*(device const packed_uchar4 *)(b + 1u + sub));
        uchar4 c1 = uchar4(*(device const packed_uchar4 *)(b + 5u + sub));
        wa = bm_e4m3_4(c0);
        wb = bm_e4m3_4(c1);
    } else if constexpr (KIND == BM_Q8_0) {
        scale = float(*(device const half *)b);
        uint sub = lane_sub * 8u;
        char4 c0 = char4(*(device const packed_char4 *)(b + 2u + sub));
        char4 c1 = char4(*(device const packed_char4 *)(b + 6u + sub));
        wa = half4(c0);
        wb = half4(c1);
    } else if constexpr (KIND == BM_INT4_G32) {
        scale = float(*(device const half *)b);
        uint sub = lane_sub * 4u;
        uchar4 raw = uchar4(*(device const packed_uchar4 *)(b + 2u + sub));
        uchar4 lo = raw & uchar4(0xFu);
        uchar4 hi = raw >> 4u;
        wa = half4(lo.x, hi.x, lo.y, hi.y) - half4(8.0h);
        wb = half4(lo.z, hi.z, lo.w, hi.w) - half4(8.0h);
    } else {
        // Low nibbles hold elements 0..15, high hold 16..31.
        scale = float(*(device const half *)b);
        uint hi = lane_sub >> 1u;
        uint boff = (lane_sub & 1u) * 8u;
        uchar4 r0 = uchar4(*(device const packed_uchar4 *)(b + 2u + boff));
        uchar4 r1 = uchar4(*(device const packed_uchar4 *)(b + 6u + boff));
        uchar4 q0 = hi ? (r0 >> 4u) : (r0 & uchar4(0xFu));
        uchar4 q1 = hi ? (r1 >> 4u) : (r1 & uchar4(0xFu));
        wa = half4(q0) - half4(8.0h);
        wb = half4(q1) - half4(8.0h);
    }
}

template <uint KIND>
static inline constexpr uint bm_block_bytes() {
    if constexpr (KIND == BM_MXFP8) return 33u;
    else if constexpr (KIND == BM_Q8_0) return 34u;
    else return 18u;
}

// Four output cols per warp, four rows per tile.
template <uint KIND, uint MT>
static inline void bm_dec_body(
    device const float *x,
    device const uchar *w,
    device float *out,
    uint m,
    uint n,
    uint k,
    uint2 tg,
    uint tid) {
    constexpr uint BB = bm_block_bytes<KIND>();
    uint sg = tid >> 5u;
    uint lane = tid & 31u;
    uint col0 = (tg.x * 4u + sg) * BM_NC;
    uint r0 = tg.y * MT;
    if (col0 >= n || r0 >= m) return;

    uint nblk = k >> 5u;
    ulong row_bytes = ulong(nblk) * ulong(BB);
    uint b0 = lane >> 2u;
    uint lane_sub = lane & 3u;
    uint elem_sub = lane_sub * 8u;

    device const uchar *ptrs[BM_NC];
    ptrs[0] = w + ulong(col0) * row_bytes;
    #pragma unroll
    for (uint c = 1u; c < BM_NC; c++) {
        ptrs[c] = (col0 + c < n) ? (ptrs[0] + ulong(c) * row_bytes) : ptrs[0];
    }

    ulong xrow[MT];
    #pragma unroll
    for (uint r = 0u; r < MT; r++) {
        xrow[r] = ulong(min(r0 + r, m - 1u)) * ulong(k);
    }

    float acc[BM_NC][MT];
    #pragma unroll
    for (uint c = 0u; c < BM_NC; c++) {
        #pragma unroll
        for (uint r = 0u; r < MT; r++) {
            acc[c][r] = 0.0f;
        }
    }

    for (uint blk = b0; blk < nblk; blk += 8u) {
        ulong boff = ulong(blk) * ulong(BB);
        uint xoff = (blk << 5u) + elem_sub;

        if constexpr (MT <= 2u) {
            float4 xa[MT], xb[MT];
            #pragma unroll
            for (uint r = 0u; r < MT; r++) {
                device const float *xr = x + xrow[r] + ulong(xoff);
                xa[r] = *(device const float4 *)(xr);
                xb[r] = *(device const float4 *)(xr + 4u);
            }
            #pragma unroll
            for (uint c = 0u; c < BM_NC; c++) {
                if (c == 0u || col0 + c < n) {
                    float sc;
                    half4 wa, wb;
                    bm_decode8<KIND>(ptrs[c] + boff, lane_sub, sc, wa, wb);
                    float4 fa = float4(wa);
                    float4 fb = float4(wb);
                    #pragma unroll
                    for (uint r = 0u; r < MT; r++) {
                        acc[c][r] = fma(sc, dot(xa[r], fa) + dot(xb[r], fb), acc[c][r]);
                    }
                }
            }
        } else {
            // Stream x from L1 cache to keep registers unspilled.
            #pragma unroll
            for (uint c = 0u; c < BM_NC; c++) {
                if (c == 0u || col0 + c < n) {
                    float sc;
                    half4 wa, wb;
                    bm_decode8<KIND>(ptrs[c] + boff, lane_sub, sc, wa, wb);
                    float4 fa = float4(wa);
                    float4 fb = float4(wb);
                    #pragma unroll
                    for (uint r = 0u; r < MT; r++) {
                        device const float *xr = x + xrow[r] + ulong(xoff);
                        float4 xa = *(device const float4 *)(xr);
                        float4 xb = *(device const float4 *)(xr + 4u);
                        acc[c][r] = fma(sc, dot(xa, fa) + dot(xb, fb), acc[c][r]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (uint r = 0u; r < MT; r++) {
        float sums[BM_NC];
        #pragma unroll
        for (uint c = 0u; c < BM_NC; c++) {
            sums[c] = simd_sum(acc[c][r]);
        }
        if (lane == 0u && r0 + r < m) {
            ulong ob = ulong(r0 + r) * ulong(n) + ulong(col0);
            #pragma unroll
            for (uint c = 0u; c < BM_NC; c++) {
                if (c == 0u || col0 + c < n) out[ob + c] = sums[c];
            }
        }
    }
}

#define BM_DEF(NAME, KIND, MT) \
kernel void NAME( \
    device const float *x [[buffer(0)]], \
    device const uchar *w [[buffer(1)]], \
    device float *out [[buffer(2)]], \
    constant uint &m [[buffer(3)]], \
    constant uint &n [[buffer(4)]], \
    constant uint &k [[buffer(5)]], \
    uint2 tg [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]) { \
    bm_dec_body<KIND, MT>(x, w, out, m, n, k, tg, tid); \
}

#define BM_FORM(PRE, KIND) \
    BM_DEF(PRE##_m1, KIND, 1u) \
    BM_DEF(PRE##_m2, KIND, 2u) \
    BM_DEF(PRE##_m3, KIND, 3u) \
    BM_DEF(PRE##_m4, KIND, 4u)

BM_FORM(bm_mxfp8, BM_MXFP8)
BM_FORM(bm_q8_0, BM_Q8_0)
BM_FORM(bm_int4_g32, BM_INT4_G32)
BM_FORM(bm_q4_0, BM_Q4_0)
