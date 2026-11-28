#include <metal_stdlib>

using namespace metal;

constant uint PM_THREADS = 128u;
constant uint PM_NC = 4u;

enum PmKind : uint {
    PM_F32 = 0u,
    PM_BF16 = 1u,
    PM_F16 = 2u,
    PM_FP8 = 3u,
    PM_MXFP8 = 4u,
    PM_Q8_0 = 5u,
    PM_INT4_G32 = 6u,
    PM_Q4_0 = 7u,
};

// Branchless E4M3FN byte quad to float4.
static inline float4 pm_e4m3(uchar4 raw) {
    uint2 u = uint2(uint(raw.x) | (uint(raw.y) << 16u), uint(raw.z) | (uint(raw.w) << 16u));
    uint2 s = (u & 0x00800080u) << 8u;
    uint2 m = u & 0x007F007Fu;
    half4 norm = as_type<half4>(s | ((m << 7u) + 0x20002000u));
    half4 sub = as_type<half4>(s | 0x40004000u | m) - as_type<half4>(s | 0x40004000u);
    return float4(select(norm, sub, abs(norm) < half4(0.015625h)));
}

// Four dense elements at `at` with optional alignment.
template <uint KIND, bool ALIGNED>
static inline float4 pm_load4(device const uchar *w, ulong at) {
    if constexpr (KIND == PM_F32) {
        device const float *p = (device const float *)w + at;
        return ALIGNED ? *(device const float4 *)p : float4(*(device const packed_float4 *)p);
    } else if constexpr (KIND == PM_BF16) {
        device const ushort *p = (device const ushort *)w + at;
        ushort4 u = ALIGNED ? *(device const ushort4 *)p : ushort4(*(device const packed_ushort4 *)p);
        return as_type<float4>(uint4(u) << 16u);
    } else if constexpr (KIND == PM_F16) {
        device const half *p = (device const half *)w + at;
        half4 h = ALIGNED ? *(device const half4 *)p : half4(*(device const packed_half4 *)p);
        return float4(h);
    } else {
        device const uchar *p = w + at;
        uchar4 c = ALIGNED ? *(device const uchar4 *)p : uchar4(*(device const packed_uchar4 *)p);
        return pm_e4m3(c);
    }
}

// Single dense element for tail indices.
template <uint KIND>
static inline float pm_load1(device const uchar *w, ulong at) {
    if constexpr (KIND == PM_F32) {
        return ((device const float *)w)[at];
    } else if constexpr (KIND == PM_BF16) {
        return as_type<float>(uint(((device const ushort *)w)[at]) << 16u);
    } else if constexpr (KIND == PM_F16) {
        return float(((device const half *)w)[at]);
    } else {
        return pm_e4m3(uchar4(w[at], 0, 0, 0)).x;
    }
}

template <uint KIND>
static inline constexpr uint pm_blk_bytes() {
    if constexpr (KIND == PM_MXFP8) return 33u;
    else if constexpr (KIND == PM_Q8_0) return 34u;
    else return 18u;
}

// Eight unscaled weights and block scale for sub-lane `sub`.
template <uint KIND>
static inline void pm_blk8(
    device const uchar *b,
    uint sub,
    thread float &sc,
    thread float4 &w0,
    thread float4 &w1) {
    if constexpr (KIND == PM_MXFP8) {
        sc = as_type<float>(uint(b[0]) << 23u);
        device const uchar *p = b + 1u + sub * 8u;
        w0 = pm_e4m3(uchar4(*(device const packed_uchar4 *)(p)));
        w1 = pm_e4m3(uchar4(*(device const packed_uchar4 *)(p + 4u)));
    } else if constexpr (KIND == PM_Q8_0) {
        sc = float(*(device const half *)b);
        device const uchar *p = b + 2u + sub * 8u;
        w0 = float4(char4(*(device const packed_char4 *)(p)));
        w1 = float4(char4(*(device const packed_char4 *)(p + 4u)));
    } else if constexpr (KIND == PM_INT4_G32) {
        device const ushort *u = (device const ushort *)b;
        sc = float(as_type<half>(u[0]));
        uint p0 = u[1u + sub * 2u];
        uint p1 = u[2u + sub * 2u];
        w0 = float4(p0 & 15u, (p0 >> 4u) & 15u, (p0 >> 8u) & 15u, p0 >> 12u) - 8.0f;
        w1 = float4(p1 & 15u, (p1 >> 4u) & 15u, (p1 >> 8u) & 15u, p1 >> 12u) - 8.0f;
    } else {
        device const ushort *u = (device const ushort *)b;
        sc = float(as_type<half>(u[0]));
        uint sh = (sub >> 1u) * 4u;
        uint bo = 1u + (sub & 1u) * 4u;
        uint d0 = u[bo + 0u], d1 = u[bo + 1u], d2 = u[bo + 2u], d3 = u[bo + 3u];
        w0 = float4((d0 >> sh) & 15u, (d0 >> (sh + 8u)) & 15u, (d1 >> sh) & 15u, (d1 >> (sh + 8u)) & 15u) - 8.0f;
        w1 = float4((d2 >> sh) & 15u, (d2 >> (sh + 8u)) & 15u, (d3 >> sh) & 15u, (d3 >> (sh + 8u)) & 15u) - 8.0f;
    }
}

// Writes four reduced columns for one picked output row.
static inline void pm_finish(
    device float *out,
    uint row,
    uint col0,
    uint n,
    uint lane,
    thread float acc[PM_NC]) {
    float sums[PM_NC];
    #pragma unroll
    for (uint c = 0u; c < PM_NC; c++) {
        sums[c] = simd_sum(acc[c]);
    }
    if (lane == 0u) {
        ulong ob = ulong(row) * n + col0;
        #pragma unroll
        for (uint c = 0u; c < PM_NC; c++) {
            if (col0 + c < n) out[ob + c] = sums[c];
        }
    }
}

// Validates bounds and zeroes out-of-range expert rows.
static inline bool pm_begin(
    device const uint *ids,
    device float *out,
    uint rows,
    uint n,
    uint count,
    uint stride,
    uint offset,
    uint first,
    uint2 tg,
    uint tid,
    thread uint &row,
    thread uint &col0,
    thread uint &lane,
    thread ulong &base_row) {
    row = tg.y;
    uint sg = tid >> 5u;
    lane = tid & 31u;
    col0 = (tg.x * 4u + sg) * PM_NC;
    if (row >= rows || col0 >= n) return false;
    uint raw_id = ids[row];
    if (raw_id < first || (raw_id - first) >= count) {
        if (lane == 0u) {
            ulong ob = ulong(row) * n + col0;
            #pragma unroll
            for (uint c = 0u; c < PM_NC; c++) {
                if (col0 + c < n) out[ob + c] = 0.0f;
            }
        }
        return false;
    }
    base_row = ulong(raw_id - first) * stride + offset + col0;
    return true;
}

// Coalesced picked matmul for dense weight formats.
template <uint KIND>
static inline void pm_dense(
    device const float *x,
    device const uchar *w,
    device const uint *ids,
    device float *out,
    uint rows,
    uint n,
    uint k,
    uint count,
    uint stride,
    uint offset,
    uint repeat,
    uint first,
    uint2 tg,
    uint tid) {
    uint row, col0, lane;
    ulong base_row;
    if (!pm_begin(ids, out, rows, n, count, stride, offset, first, tg, tid, row, col0, lane, base_row)) {
        return;
    }
    float acc[PM_NC] = {0.0f, 0.0f, 0.0f, 0.0f};
    ulong w_off[PM_NC];
    #pragma unroll
    for (uint c = 0u; c < PM_NC; c++) {
        w_off[c] = ((col0 + c < n) ? (base_row + c) : base_row) * k;
    }

    device const float *xp = x + ulong(row / repeat) * k;
    uint k4 = k & ~3u;
    if ((k & 3u) == 0u) {
        for (uint i = lane * 4u; i < k4; i += 128u) {
            float4 xv = *(device const float4 *)(xp + i);
            #pragma unroll
            for (uint c = 0u; c < PM_NC; c++) {
                if (c == 0u || col0 + c < n) {
                    acc[c] += dot(xv, pm_load4<KIND, true>(w, w_off[c] + i));
                }
            }
        }
    } else {
        for (uint i = lane * 4u; i < k4; i += 128u) {
            float4 xv = float4(*(device const packed_float4 *)(xp + i));
            #pragma unroll
            for (uint c = 0u; c < PM_NC; c++) {
                if (c == 0u || col0 + c < n) {
                    acc[c] += dot(xv, pm_load4<KIND, false>(w, w_off[c] + i));
                }
            }
        }
        uint tail = k4 + lane;
        if (tail < k) {
            float xv = xp[tail];
            #pragma unroll
            for (uint c = 0u; c < PM_NC; c++) {
                if (c == 0u || col0 + c < n) {
                    acc[c] = fma(xv, pm_load1<KIND>(w, w_off[c] + tail), acc[c]);
                }
            }
        }
    }

    pm_finish(out, row, col0, n, lane, acc);
}

// Coalesced picked matmul for 32-element block weight formats.
template <uint KIND>
static inline void pm_block(
    device const float *x,
    device const uchar *w,
    device const uint *ids,
    device float *out,
    uint rows,
    uint n,
    uint k,
    uint count,
    uint stride,
    uint offset,
    uint repeat,
    uint first,
    uint2 tg,
    uint tid) {
    constexpr uint BB = pm_blk_bytes<KIND>();
    uint row, col0, lane;
    ulong base_row;
    if (!pm_begin(ids, out, rows, n, count, stride, offset, first, tg, tid, row, col0, lane, base_row)) {
        return;
    }
    float acc[PM_NC] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint nblk = k >> 5u;
    ulong row_bytes = ulong(nblk) * BB;
    device const uchar *wp[PM_NC];
    #pragma unroll
    for (uint c = 0u; c < PM_NC; c++) {
        wp[c] = w + ((col0 + c < n) ? (base_row + c) : base_row) * row_bytes;
    }

    device const float *xp = x + ulong(row / repeat) * k;
    uint b0 = lane >> 2u;
    uint sub = lane & 3u;
    uint esub = sub * 8u;

    for (uint blk = b0; blk < nblk; blk += 8u) {
        ulong boff = ulong(blk) * BB;
        device const float *xr = xp + (blk << 5u) + esub;
        float4 xa = *(device const float4 *)(xr);
        float4 xb = *(device const float4 *)(xr + 4u);
        #pragma unroll
        for (uint c = 0u; c < PM_NC; c++) {
            if (c == 0u || col0 + c < n) {
                float sc;
                float4 w0, w1;
                pm_blk8<KIND>(wp[c] + boff, sub, sc, w0, w1);
                acc[c] = fma(sc, dot(xa, w0) + dot(xb, w1), acc[c]);
            }
        }
    }

    pm_finish(out, row, col0, n, lane, acc);
}

#define PM_ARGS                                      \
    device const float *x [[buffer(0)]],             \
    device const uchar *w [[buffer(1)]],             \
    device const uint *ids [[buffer(2)]],            \
    device float *out [[buffer(3)]],                 \
    constant uint &rows [[buffer(4)]],               \
    constant uint &n [[buffer(5)]],                  \
    constant uint &k [[buffer(6)]],                  \
    constant uint &count [[buffer(7)]],              \
    constant uint &stride [[buffer(8)]],             \
    constant uint &offset [[buffer(9)]],             \
    constant uint &repeat [[buffer(10)]],            \
    constant uint &first [[buffer(11)]],             \
    uint2 tg [[threadgroup_position_in_grid]],       \
    uint tid [[thread_index_in_threadgroup]]

#define PM_DENSE(NAME, KIND) \
kernel void NAME(PM_ARGS) { \
    pm_dense<KIND>(x, w, ids, out, rows, n, k, count, stride, offset, repeat, first, tg, tid); \
}

#define PM_BLOCK(NAME, KIND) \
kernel void NAME(PM_ARGS) { \
    pm_block<KIND>(x, w, ids, out, rows, n, k, count, stride, offset, repeat, first, tg, tid); \
}

PM_DENSE(pm_f32, PM_F32)
PM_DENSE(pm_bf16, PM_BF16)
PM_DENSE(pm_f16, PM_F16)
PM_DENSE(pm_fp8, PM_FP8)
PM_BLOCK(pm_mxfp8, PM_MXFP8)
PM_BLOCK(pm_q8_0, PM_Q8_0)
PM_BLOCK(pm_int4_g32, PM_INT4_G32)
PM_BLOCK(pm_q4_0, PM_Q4_0)
