#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

constant uint MM_BK = 32u;
constant uint MM_BN = 64u;
constant uint MM_STRIDE = 36u;

enum MmKind : uint {
    MM_F16 = 0u,
    MM_FP8 = 1u,
    MM_MXFP8 = 2u,
    MM_Q8_0 = 3u,
    MM_INT4_G32 = 4u,
    MM_Q4_0 = 5u,
};

// Branchless E4M3FN byte quad to float4.
static inline float4 mm_e4m3_4(uchar4 raw) {
    uint2 u = uint2(uint(raw.x) | (uint(raw.y) << 16u), uint(raw.z) | (uint(raw.w) << 16u));
    uint2 s = (u & 0x00800080u) << 8u;
    uint2 m = u & 0x007F007Fu;
    half4 norm = as_type<half4>(s | ((m << 7u) + 0x20002000u));
    half4 sub = as_type<half4>(s | 0x40004000u | m) - as_type<half4>(s | 0x40004000u);
    return float4(select(norm, sub, abs(norm) < half4(0.015625h)));
}

template <uint KIND>
static inline constexpr uint mm_block_bytes() {
    if constexpr (KIND == MM_MXFP8) return 33u;
    else if constexpr (KIND == MM_Q8_0) return 34u;
    else return 18u;
}

// Loads 8 weights from row `r` at `k0`.
template <uint KIND>
static inline void mm_load_w8(
    device const uchar *w,
    uint r,
    uint k0,
    uint sub,
    uint k,
    thread float4 &v0,
    thread float4 &v1) {
    if constexpr (KIND == MM_F16) {
        device const half *p = (device const half *)w + ulong(r) * k + k0 + sub * 8u;
        v0 = float4(half4(*(device const packed_half4 *)(p)));
        v1 = float4(half4(*(device const packed_half4 *)(p + 4u)));
    } else if constexpr (KIND == MM_FP8) {
        device const uchar *p = w + ulong(r) * k + k0 + sub * 8u;
        v0 = mm_e4m3_4(uchar4(*(device const packed_uchar4 *)(p)));
        v1 = mm_e4m3_4(uchar4(*(device const packed_uchar4 *)(p + 4u)));
    } else if constexpr (KIND == MM_MXFP8) {
        constexpr uint BB = mm_block_bytes<MM_MXFP8>();
        device const uchar *b = w + (ulong(r) * (k >> 5u) + (k0 >> 5u)) * BB;
        float sc = as_type<float>(uint(b[0]) << 23u);
        device const uchar *p = b + 1u + sub * 8u;
        v0 = mm_e4m3_4(uchar4(*(device const packed_uchar4 *)(p))) * sc;
        v1 = mm_e4m3_4(uchar4(*(device const packed_uchar4 *)(p + 4u))) * sc;
    } else if constexpr (KIND == MM_Q8_0) {
        constexpr uint BB = mm_block_bytes<MM_Q8_0>();
        device const uchar *b = w + (ulong(r) * (k >> 5u) + (k0 >> 5u)) * BB;
        float sc = float(*(device const half *)b);
        device const uchar *p = b + 2u + sub * 8u;
        v0 = float4(char4(*(device const packed_char4 *)(p))) * sc;
        v1 = float4(char4(*(device const packed_char4 *)(p + 4u))) * sc;
    } else if constexpr (KIND == MM_INT4_G32) {
        constexpr uint BB = mm_block_bytes<MM_INT4_G32>();
        device const ushort *b = (device const ushort *)(w + (ulong(r) * (k >> 5u) + (k0 >> 5u)) * BB);
        float sc = float(as_type<half>(b[0]));
        uint p0 = b[1u + sub * 2u];
        uint p1 = b[2u + sub * 2u];
        v0 = (float4(p0 & 15u, (p0 >> 4u) & 15u, (p0 >> 8u) & 15u, p0 >> 12u) - 8.0f) * sc;
        v1 = (float4(p1 & 15u, (p1 >> 4u) & 15u, (p1 >> 8u) & 15u, p1 >> 12u) - 8.0f) * sc;
    } else {
        constexpr uint BB = mm_block_bytes<MM_Q4_0>();
        device const ushort *b = (device const ushort *)(w + (ulong(r) * (k >> 5u) + (k0 >> 5u)) * BB);
        float sc = float(as_type<half>(b[0]));
        uint sh = (sub >> 1u) * 4u;
        uint bo = 1u + (sub & 1u) * 4u;
        uint w0 = b[bo + 0u], w1 = b[bo + 1u], w2 = b[bo + 2u], w3 = b[bo + 3u];
        v0 = (float4((w0 >> sh) & 15u, (w0 >> (sh + 8u)) & 15u, (w1 >> sh) & 15u, (w1 >> (sh + 8u)) & 15u) - 8.0f) * sc;
        v1 = (float4((w2 >> sh) & 15u, (w2 >> (sh + 8u)) & 15u, (w3 >> sh) & 15u, (w3 >> (sh + 8u)) & 15u) - 8.0f) * sc;
    }
}

// Loads 8 tail elements safely when k0 + 32 > k.
template <uint KIND>
static inline void mm_load_w8_tail(
    device const uchar *w,
    uint r,
    uint k0,
    uint sub,
    uint k,
    thread float4 &v0,
    thread float4 &v1) {
    uint c0 = k0 + sub * 8u;
    if (c0 + 8u <= k) {
        mm_load_w8<KIND>(w, r, k0, sub, k, v0, v1);
        return;
    }
    float tmp[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if constexpr (KIND == MM_F16 || KIND == MM_FP8) {
        for (uint i = 0u; i < 8u && c0 + i < k; i++) {
            ulong idx = ulong(r) * k + c0 + i;
            if constexpr (KIND == MM_F16) {
                tmp[i] = float(((device const half *)w)[idx]);
            } else {
                tmp[i] = mm_e4m3_4(uchar4(w[idx], 0, 0, 0)).x;
            }
        }
    }
    v0 = float4(tmp[0], tmp[1], tmp[2], tmp[3]);
    v1 = float4(tmp[4], tmp[5], tmp[6], tmp[7]);
}

// Direct simdgroup_float8x8 tiled GEMM (max 8 accs/warp to avoid spills).
template <uint KIND, uint BM, uint WM_WARPS, uint WN_WARPS>
static inline void mm_body(
    device const float *x,
    device const uchar *w,
    device float *out,
    uint m,
    uint n,
    uint k,
    uint tg_id,
    uint tid,
    threadgroup float sa[BM][MM_STRIDE],
    threadgroup float sb[MM_BN][MM_STRIDE]) {
    constexpr uint WARPS = WM_WARPS * WN_WARPS;
    constexpr uint THREADS = WARPS * 32u;
    constexpr uint TM = (BM / WM_WARPS) / 8u;
    constexpr uint TN = (MM_BN / WN_WARPS) / 8u;
    uint tiles_m = (m + BM - 1u) / BM;
    uint tiles_n = (n + MM_BN - 1u) / MM_BN;
    // Banded 8-row scheduling keeps weights hot in L2.
    uint band = min(8u, tiles_m);
    uint group = tg_id / (band * tiles_n);
    uint rem = tg_id - group * (band * tiles_n);
    uint first_m = group * band;
    uint rows_in = min(band, tiles_m - first_m);
    uint tile_n = rem / rows_in;
    uint tile_m = first_m + (rem - tile_n * rows_in);
    uint row0 = tile_m * BM;
    uint col0 = tile_n * MM_BN;
    if (row0 >= m || col0 >= n) return;

    uint sg = tid >> 5u;
    uint lane = tid & 31u;
    uint wm = (sg / WN_WARPS) * (TM * 8u);
    uint wn = (sg % WN_WARPS) * (TN * 8u);

    simdgroup_float8x8 acc[TM][TN];
    #pragma unroll
    for (uint i = 0u; i < TM; i++) {
        #pragma unroll
        for (uint j = 0u; j < TN; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    uint sub = tid & 3u;
    uint row_step = THREADS >> 2u;

    for (uint k0 = 0u; k0 < k; k0 += MM_BK) {
        bool full_k = (k0 + MM_BK <= k);
        for (uint r = tid >> 2u; r < BM; r += row_step) {
            uint gr = row0 + r;
            uint c = sub * 8u;
            float4 a0 = float4(0.0f), a1 = float4(0.0f);
            if (gr < m) {
                if (full_k || k0 + c + 8u <= k) {
                    device const float *xp = x + ulong(gr) * k + k0 + c;
                    a0 = float4(*(device const packed_float4 *)(xp));
                    a1 = float4(*(device const packed_float4 *)(xp + 4u));
                } else {
                    float tmp[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    for (uint i = 0u; i < 8u && k0 + c + i < k; i++) {
                        tmp[i] = x[ulong(gr) * k + k0 + c + i];
                    }
                    a0 = float4(tmp[0], tmp[1], tmp[2], tmp[3]);
                    a1 = float4(tmp[4], tmp[5], tmp[6], tmp[7]);
                }
            }
            *(threadgroup float4 *)(&sa[r][c]) = a0;
            *(threadgroup float4 *)(&sa[r][c + 4u]) = a1;
        }

        for (uint r = tid >> 2u; r < MM_BN; r += row_step) {
            uint gc = col0 + r;
            uint c = sub * 8u;
            float4 b0 = float4(0.0f), b1 = float4(0.0f);
            if (gc < n) {
                if (full_k) {
                    mm_load_w8<KIND>(w, gc, k0, sub, k, b0, b1);
                } else {
                    mm_load_w8_tail<KIND>(w, gc, k0, sub, k, b0, b1);
                }
            }
            *(threadgroup float4 *)(&sb[r][c]) = b0;
            *(threadgroup float4 *)(&sb[r][c + 4u]) = b1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0u; kk < MM_BK; kk += 8u) {
            simdgroup_float8x8 ta[TM];
            simdgroup_float8x8 tb[TN];
            #pragma unroll
            for (uint i = 0u; i < TM; i++) {
                simdgroup_load(ta[i], &sa[wm + i * 8u][kk], MM_STRIDE);
            }
            #pragma unroll
            for (uint j = 0u; j < TN; j++) {
                simdgroup_load(tb[j], &sb[wn + j * 8u][kk], MM_STRIDE, ulong2(0, 0), true);
            }
            #pragma unroll
            for (uint i = 0u; i < TM; i++) {
                #pragma unroll
                for (uint j = 0u; j < TN; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], ta[i], tb[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint lr = ((lane >> 2u) & 4u) + ((lane >> 1u) & 3u);
    uint lc = (((lane >> 2u) & 2u) << 1u) + ((lane & 1u) << 1u);
    #pragma unroll
    for (uint i = 0u; i < TM; i++) {
        uint gr = row0 + wm + i * 8u + lr;
        if (gr >= m) continue;
        #pragma unroll
        for (uint j = 0u; j < TN; j++) {
            uint gc = col0 + wn + j * 8u + lc;
            float v0 = acc[i][j].thread_elements()[0];
            float v1 = acc[i][j].thread_elements()[1];
            if (gc + 1u < n) {
                out[ulong(gr) * n + gc] = v0;
                out[ulong(gr) * n + gc + 1u] = v1;
            } else if (gc < n) {
                out[ulong(gr) * n + gc] = v0;
            }
        }
    }
}

#define MM_ARGS                                             \
    device const float *x [[buffer(0)]],                    \
    device const uchar *w [[buffer(1)]],                    \
    device float *out [[buffer(2)]],                        \
    constant uint &m [[buffer(3)]],                         \
    constant uint &n [[buffer(4)]],                         \
    constant uint &k [[buffer(5)]],                         \
    uint tg_id [[threadgroup_position_in_grid]],            \
    uint tid [[thread_index_in_threadgroup]]

#define MM_PAIR(SUFFIX, KIND)                               \
kernel void mm32_##SUFFIX(MM_ARGS) {                        \
    threadgroup float sa[32u][MM_STRIDE];                   \
    threadgroup float sb[MM_BN][MM_STRIDE];                 \
    mm_body<KIND, 32u, 2u, 2u>(x, w, out, m, n, k, tg_id, tid, sa, sb); \
}                                                           \
kernel void mm64_##SUFFIX(MM_ARGS) {                        \
    threadgroup float sa[64u][MM_STRIDE];                   \
    threadgroup float sb[MM_BN][MM_STRIDE];                 \
    mm_body<KIND, 64u, 2u, 4u>(x, w, out, m, n, k, tg_id, tid, sa, sb); \
}

MM_PAIR(f16, MM_F16)
MM_PAIR(fp8, MM_FP8)
MM_PAIR(mxfp8, MM_MXFP8)
MM_PAIR(q8_0, MM_Q8_0)
MM_PAIR(int4_g32, MM_INT4_G32)
MM_PAIR(q4_0, MM_Q4_0)
