#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

constant uint BM_THREADS = 128u;
constant uint BM_STEP = 128u;
constant uint BM_TM = 32u;
constant uint BM_TN = 32u;
constant uint BM_TK = 16u;
constant uint BM_AS = 20u;
constant uint BM_BS = 36u;

// Accumulates 128 floats across NC columns and MT rows.
template <uint MT, uint NC>
static inline void bm_step(
    device const float *x,
    thread device const float *const (&wptrs)[NC],
    thread const ulong (&xrow)[MT],
    uint off,
    thread float (&acc)[NC][MT]) {
    if constexpr (MT <= 4u) {
        float4 xv[MT];
        #pragma unroll
        for (uint r = 0u; r < MT; r++) {
            xv[r] = *(device const float4 *)(x + xrow[r] + ulong(off));
        }
        #pragma unroll
        for (uint c = 0u; c < NC; c++) {
            float4 wv = *(device const float4 *)(wptrs[c] + ulong(off));
            #pragma unroll
            for (uint r = 0u; r < MT; r++) {
                acc[c][r] += dot(xv[r], wv);
            }
        }
    } else {
        float4 wv[NC];
        #pragma unroll
        for (uint c = 0u; c < NC; c++) {
            wv[c] = *(device const float4 *)(wptrs[c] + ulong(off));
        }
        #pragma unroll
        for (uint r = 0u; r < MT; r++) {
            float4 xv = *(device const float4 *)(x + xrow[r] + ulong(off));
            #pragma unroll
            for (uint c = 0u; c < NC; c++) {
                acc[c][r] += dot(xv, wv[c]);
            }
        }
    }
}

// Coalesced batched decode over NC columns and MT rows.
template <uint MT, uint NC>
static inline void bm_dec_body(
    device const float *x,
    device const float *w,
    device float *out,
    uint m,
    uint n,
    uint k,
    uint x_div,
    uint batches,
    uint2 tg,
    uint tid) {
    uint rtiles = (m + MT - 1u) / MT;
    uint b = tg.y / rtiles;
    uint ty = tg.y - b * rtiles;
    if (b >= batches) return;
    uint sg = tid >> 5u;
    uint lane = tid & 31u;
    uint col0 = (tg.x * 4u + sg) * NC;
    uint r0 = ty * MT;
    if (col0 >= n || r0 >= m) return;

    device const float *wbase = w + ulong(b) * ulong(n) * ulong(k);
    device const float *wptrs[NC];
    wptrs[0] = wbase + ulong(col0) * ulong(k);
    #pragma unroll
    for (uint c = 1u; c < NC; c++) {
        wptrs[c] = (col0 + c < n) ? (wptrs[0] + ulong(c) * ulong(k)) : wptrs[0];
    }

    ulong xrow[MT];
    #pragma unroll
    for (uint r = 0u; r < MT; r++) {
        uint row = b * m + min(r0 + r, m - 1u);
        xrow[r] = ulong(row / x_div) * ulong(k);
    }

    float acc[NC][MT];
    #pragma unroll
    for (uint c = 0u; c < NC; c++) {
        #pragma unroll
        for (uint r = 0u; r < MT; r++) {
            acc[c][r] = 0.0f;
        }
    }

    uint off = lane * 4u;
    for (; off + BM_STEP * 3u < k; off += BM_STEP * 4u) {
        bm_step<MT, NC>(x, wptrs, xrow, off, acc);
        bm_step<MT, NC>(x, wptrs, xrow, off + BM_STEP, acc);
        bm_step<MT, NC>(x, wptrs, xrow, off + BM_STEP * 2u, acc);
        bm_step<MT, NC>(x, wptrs, xrow, off + BM_STEP * 3u, acc);
    }
    for (; off < k; off += BM_STEP) {
        bm_step<MT, NC>(x, wptrs, xrow, off, acc);
    }

    device float *obase = out + ulong(b) * ulong(m) * ulong(n);
    #pragma unroll
    for (uint c = 0u; c < NC; c++) {
        #pragma unroll
        for (uint r = 0u; r < MT; r++) {
            float sum = simd_sum(acc[c][r]);
            if (lane == 0u && col0 + c < n && r0 + r < m) {
                obase[ulong(r0 + r) * ulong(n) + ulong(col0 + c)] = sum;
            }
        }
    }
}

#define BM_DEC_ENTRY(NAME, MT, NC)                                        \
    kernel void NAME(                                                     \
        device const float *x [[buffer(0)]],                              \
        device const float *w [[buffer(1)]],                              \
        device float *out [[buffer(2)]],                                  \
        constant uint &m [[buffer(3)]],                                   \
        constant uint &n [[buffer(4)]],                                   \
        constant uint &k [[buffer(5)]],                                   \
        constant uint &x_div [[buffer(6)]],                               \
        constant uint &batches [[buffer(7)]],                             \
        uint2 tg [[threadgroup_position_in_grid]],                        \
        uint tid [[thread_index_in_threadgroup]]) {                       \
        bm_dec_body<MT, NC>(x, w, out, m, n, k, x_div, batches, tg, tid); \
    }

BM_DEC_ENTRY(bm_dec_m1, 1u, 2u)
BM_DEC_ENTRY(bm_dec_m2, 2u, 4u)
BM_DEC_ENTRY(bm_dec_m3, 3u, 4u)
BM_DEC_ENTRY(bm_dec_m4, 4u, 4u)
BM_DEC_ENTRY(bm_dec_m5, 5u, 2u)
BM_DEC_ENTRY(bm_dec_m6, 6u, 2u)
BM_DEC_ENTRY(bm_dec_m7, 7u, 2u)
BM_DEC_ENTRY(bm_dec_m8, 8u, 2u)

// Batched 32x32 simdgroup_float8x8 tiled matmul for prefill.
kernel void bm_mma_f32(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &m [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &k [[buffer(5)]],
    constant uint &x_div [[buffer(6)]],
    constant uint &batches [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint rtiles = (m + BM_TM - 1u) / BM_TM;
    uint b = tg.y / rtiles;
    uint ty = tg.y - b * rtiles;
    if (b >= batches) return;
    uint r0 = ty * BM_TM;
    uint c0 = tg.x * BM_TN;
    if (r0 >= m || c0 >= n) return;

    threadgroup float as[BM_TM][BM_AS];
    threadgroup float bs[BM_TK][BM_BS];

    uint sg = tid >> 5u;
    uint sr = (sg >> 1u) * 16u;
    uint sc = (sg & 1u) * 16u;

    simdgroup_float8x8 acc[2][2];
    #pragma unroll
    for (uint i = 0u; i < 2u; i++) {
        #pragma unroll
        for (uint j = 0u; j < 2u; j++) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    device const float *wbase = w + ulong(b) * ulong(n) * ulong(k);
    uint ar = tid >> 2u;
    uint ac = (tid & 3u) * 4u;
    uint br = tid >> 2u;
    uint bc = (tid & 3u) * 4u;

    for (uint k0 = 0u; k0 < k; k0 += BM_TK) {
        // Loads A tile from x.
        if (r0 + ar < m) {
            uint row = b * m + r0 + ar;
            device const float *xr = x + ulong(row / x_div) * ulong(k) + ulong(k0 + ac);
            if (k0 + ac + 4u <= k) {
                float4 v = ((k & 3u) == 0u)
                    ? *(device const float4 *)xr
                    : float4(*(device const packed_float4 *)xr);
                as[ar][ac] = v.x;
                as[ar][ac + 1u] = v.y;
                as[ar][ac + 2u] = v.z;
                as[ar][ac + 3u] = v.w;
            } else {
                for (uint t = 0u; t < 4u; t++) {
                    as[ar][ac + t] = (k0 + ac + t < k) ? xr[t] : 0.0f;
                }
            }
        } else {
            as[ar][ac] = 0.0f;
            as[ar][ac + 1u] = 0.0f;
            as[ar][ac + 2u] = 0.0f;
            as[ar][ac + 3u] = 0.0f;
        }

        // Loads transposed B tile from w.
        if (c0 + br < n) {
            device const float *wr = wbase + ulong(c0 + br) * ulong(k) + ulong(k0 + bc);
            if (k0 + bc + 4u <= k) {
                float4 v = ((k & 3u) == 0u)
                    ? *(device const float4 *)wr
                    : float4(*(device const packed_float4 *)wr);
                bs[bc][br] = v.x;
                bs[bc + 1u][br] = v.y;
                bs[bc + 2u][br] = v.z;
                bs[bc + 3u][br] = v.w;
            } else {
                for (uint t = 0u; t < 4u; t++) {
                    bs[bc + t][br] = (k0 + bc + t < k) ? wr[t] : 0.0f;
                }
            }
        } else {
            bs[bc][br] = 0.0f;
            bs[bc + 1u][br] = 0.0f;
            bs[bc + 2u][br] = 0.0f;
            bs[bc + 3u][br] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0u; kk < BM_TK; kk += 8u) {
            simdgroup_float8x8 ta0, ta1, tb0, tb1;
            simdgroup_load(ta0, &as[sr][kk], BM_AS);
            simdgroup_load(ta1, &as[sr + 8u][kk], BM_AS);
            simdgroup_load(tb0, &bs[kk][sc], BM_BS);
            simdgroup_load(tb1, &bs[kk][sc + 8u], BM_BS);
            simdgroup_multiply_accumulate(acc[0][0], ta0, tb0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], ta0, tb1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], ta1, tb0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], ta1, tb1, acc[1][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *obase = out + ulong(b) * ulong(m) * ulong(n);
    if (r0 + BM_TM <= m && c0 + BM_TN <= n) {
        simdgroup_store(acc[0][0], obase + ulong(r0 + sr) * ulong(n) + ulong(c0 + sc), n);
        simdgroup_store(acc[0][1], obase + ulong(r0 + sr) * ulong(n) + ulong(c0 + sc + 8u), n);
        simdgroup_store(acc[1][0], obase + ulong(r0 + sr + 8u) * ulong(n) + ulong(c0 + sc), n);
        simdgroup_store(acc[1][1], obase + ulong(r0 + sr + 8u) * ulong(n) + ulong(c0 + sc + 8u), n);
    } else {
        threadgroup float cs[BM_TM][BM_TN];
        simdgroup_store(acc[0][0], &cs[sr][sc], BM_TN);
        simdgroup_store(acc[0][1], &cs[sr][sc + 8u], BM_TN);
        simdgroup_store(acc[1][0], &cs[sr + 8u][sc], BM_TN);
        simdgroup_store(acc[1][1], &cs[sr + 8u][sc + 8u], BM_TN);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < BM_TM * BM_TN; i += BM_THREADS) {
            uint r = i / BM_TN;
            uint c = i % BM_TN;
            if (r0 + r < m && c0 + c < n) {
                obase[ulong(r0 + r) * ulong(n) + ulong(c0 + c)] = cs[r][c];
            }
        }
    }
}
