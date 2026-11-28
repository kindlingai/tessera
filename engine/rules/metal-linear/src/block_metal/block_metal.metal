#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

constant uint BM_THREADS = 128u;

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
        ushort2 raw2 = *(device const packed_ushort2 *)(b + 2u + lane_sub * 4u);
        ushort2 lo0 = ushort2(raw2.x & 0xFu, (raw2.x >> 4u) & 0xFu) | ushort2(0x6400u);
        ushort2 hi0 = ushort2((raw2.x >> 8u) & 0xFu, raw2.x >> 12u) | ushort2(0x6400u);
        ushort2 lo1 = ushort2(raw2.y & 0xFu, (raw2.y >> 4u) & 0xFu) | ushort2(0x6400u);
        ushort2 hi1 = ushort2((raw2.y >> 8u) & 0xFu, raw2.y >> 12u) | ushort2(0x6400u);
        wa = half4(as_type<half2>(lo0), as_type<half2>(hi0)) - half4(1032.0h);
        wb = half4(as_type<half2>(lo1), as_type<half2>(hi1)) - half4(1032.0h);
    } else {
        // Low nibbles hold elements 0..15, high hold 16..31.
        scale = float(*(device const half *)b);
        uint hi = lane_sub >> 1u;
        uint boff = (lane_sub & 1u) * 8u;
        ushort2 r0 = *(device const packed_ushort2 *)(b + 2u + boff);
        ushort2 r1 = *(device const packed_ushort2 *)(b + 6u + boff);
        ushort2 q0 = hi ? (r0 >> 4u) : r0;
        ushort2 q1 = hi ? (r1 >> 4u) : r1;
        ushort2 lo0 = ushort2(q0.x & 0xFu, (q0.x >> 8u) & 0xFu) | ushort2(0x6400u);
        ushort2 hi0 = ushort2(q0.y & 0xFu, (q0.y >> 8u) & 0xFu) | ushort2(0x6400u);
        ushort2 lo1 = ushort2(q1.x & 0xFu, (q1.x >> 8u) & 0xFu) | ushort2(0x6400u);
        ushort2 hi1 = ushort2(q1.y & 0xFu, (q1.y >> 8u) & 0xFu) | ushort2(0x6400u);
        wa = half4(as_type<half2>(lo0), as_type<half2>(hi0)) - half4(1032.0h);
        wb = half4(as_type<half2>(lo1), as_type<half2>(hi1)) - half4(1032.0h);
    }
}

template <uint KIND>
static inline constexpr uint bm_block_bytes() {
    if constexpr (KIND == BM_MXFP8) return 33u;
    else if constexpr (KIND == BM_Q8_0) return 34u;
    else return 18u;
}

// NC output columns per warp, MT rows per tile.
template <uint KIND, uint MT, uint NC>
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
    uint col0 = (tg.x * 4u + sg) * NC;
    uint r0 = tg.y * MT;
    if (col0 >= n || r0 >= m) return;

    uint nblk = k >> 5u;
    ulong row_bytes = ulong(nblk) * ulong(BB);
    uint b0 = lane >> 2u;
    uint lane_sub = lane & 3u;
    uint elem_sub = lane_sub * 8u;
    device const uchar *p0 = w + ulong(col0) * row_bytes;

    ulong xrow[MT];
    #pragma unroll
    for (uint r = 0u; r < MT; r++) {
        xrow[r] = ulong(min(r0 + r, m - 1u)) * ulong(k);
    }

    float acc[NC][MT];
    #pragma unroll
    for (uint c = 0u; c < NC; c++) {
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
            for (uint c = 0u; c < NC; c++) {
                if (c == 0u || col0 + c < n) {
                    float sc;
                    half4 wa, wb;
                    bm_decode8<KIND>(p0 + ulong(c) * row_bytes + boff, lane_sub, sc, wa, wb);
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
            for (uint c = 0u; c < NC; c++) {
                if (c == 0u || col0 + c < n) {
                    float sc;
                    half4 wa, wb;
                    bm_decode8<KIND>(p0 + ulong(c) * row_bytes + boff, lane_sub, sc, wa, wb);
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
        float sums[NC];
        #pragma unroll
        for (uint c = 0u; c < NC; c++) {
            sums[c] = simd_sum(acc[c][r]);
        }
        if (lane == 0u && r0 + r < m) {
            ulong ob = ulong(r0 + r) * ulong(n) + ulong(col0);
            #pragma unroll
            for (uint c = 0u; c < NC; c++) {
                if (c == 0u || col0 + c < n) out[ob + c] = sums[c];
            }
        }
    }
}

// FP32 simdgroup MMA decode for 5..=8 rows.
template <uint KIND, uint MT>
static inline void bm_mma_body(
    device const float *x,
    device const uchar *w,
    device float *out,
    uint n,
    uint k,
    uint tg_x,
    uint tid,
    threadgroup float xs[8][36],
    threadgroup float ws[32][36]) {
    constexpr uint BB = bm_block_bytes<KIND>();
    uint col0_tg = tg_x * 32u;
    if (col0_tg >= n) return;

    uint sg = tid >> 5u;
    uint lane = tid & 31u;
    uint wc = tid >> 2u;
    uint wsub = tid & 3u;
    uint xr = tid >> 4u;
    uint xc = (tid & 15u) << 1u;
    uint nblk = k >> 5u;
    ulong row_bytes = ulong(nblk) * ulong(BB);
    uint wcol = col0_tg + wc;
    bool w_ok = (wcol < n);
    device const uchar *wp = w + ulong(w_ok ? wcol : (n - 1u)) * row_bytes;

    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint blk = 0u; blk < nblk; blk++) {
        if (blk > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        uint k_base = blk << 5u;
        float2 xv = (xr < MT)
            ? *(device const float2 *)(x + ulong(xr) * ulong(k) + ulong(k_base + xc))
            : float2(0.0f);
        *(threadgroup float2 *)(&xs[xr][xc]) = xv;

        if (w_ok) {
            float sc;
            half4 wa, wb;
            bm_decode8<KIND>(wp + ulong(blk) * ulong(BB), wsub, sc, wa, wb);
            *(threadgroup float4 *)(&ws[wc][wsub * 8u]) = float4(wa) * sc;
            *(threadgroup float4 *)(&ws[wc][wsub * 8u + 4u]) = float4(wb) * sc;
        } else {
            *(threadgroup float4 *)(&ws[wc][wsub * 8u]) = float4(0.0f);
            *(threadgroup float4 *)(&ws[wc][wsub * 8u + 4u]) = float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0u; kk < 32u; kk += 8u) {
            simdgroup_float8x8 ta, tb;
            simdgroup_load(ta, &xs[0][kk], 36u);
            simdgroup_load(tb, &ws[sg * 8u][kk], 36u, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, ta, tb, acc);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(acc, &ws[sg * 8u][0], 36u);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    uint c_lane = lane & 7u;
    uint r_base = lane >> 3u;
    uint c_out = col0_tg + sg * 8u + c_lane;
    if (c_out < n) {
        if (r_base < MT) out[ulong(r_base) * ulong(n) + ulong(c_out)] = ws[sg * 8u + r_base][c_lane];
        if (r_base + 4u < MT) out[ulong(r_base + 4u) * ulong(n) + ulong(c_out)] = ws[sg * 8u + r_base + 4u][c_lane];
    }
}

#define BM_DEF(NAME, KIND, MT, NC) \
kernel void NAME( \
    device const float *x [[buffer(0)]], \
    device const uchar *w [[buffer(1)]], \
    device float *out [[buffer(2)]], \
    constant uint &m [[buffer(3)]], \
    constant uint &n [[buffer(4)]], \
    constant uint &k [[buffer(5)]], \
    uint2 tg [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]) { \
    bm_dec_body<KIND, MT, NC>(x, w, out, m, n, k, tg, tid); \
}

#define BM_DEF_MMA(NAME, KIND, MT) \
kernel void NAME( \
    device const float *x [[buffer(0)]], \
    device const uchar *w [[buffer(1)]], \
    device float *out [[buffer(2)]], \
    constant uint &m [[buffer(3)]], \
    constant uint &n [[buffer(4)]], \
    constant uint &k [[buffer(5)]], \
    uint2 tg [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]) { \
    (void)m; \
    threadgroup float xs[8][36]; \
    threadgroup float ws[32][36]; \
    bm_mma_body<KIND, MT>(x, w, out, n, k, tg.x, tid, xs, ws); \
}

#define BM_FORM(PRE, KIND, NC1) \
    BM_DEF(PRE##_m1, KIND, 1u, NC1) \
    BM_DEF(PRE##_m2, KIND, 2u, 4u) \
    BM_DEF(PRE##_m3, KIND, 3u, 4u) \
    BM_DEF(PRE##_m4, KIND, 4u, 4u) \
    BM_DEF_MMA(PRE##_m5, KIND, 5u) \
    BM_DEF_MMA(PRE##_m6, KIND, 6u) \
    BM_DEF_MMA(PRE##_m7, KIND, 7u) \
    BM_DEF_MMA(PRE##_m8, KIND, 8u)

BM_FORM(bm_mxfp8, BM_MXFP8, 4u)
BM_FORM(bm_q8_0, BM_Q8_0, 4u)
BM_DEF(bm_int4_g32_m1_c2, BM_INT4_G32, 1u, 2u)
BM_FORM(bm_int4_g32, BM_INT4_G32, 8u)
BM_DEF(bm_q4_0_m1_c2, BM_Q4_0, 1u, 2u)
BM_FORM(bm_q4_0, BM_Q4_0, 8u)
