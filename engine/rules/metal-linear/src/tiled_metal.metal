#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

constant uint TL_BM = 64u;
constant uint TL_BN = 64u;
constant uint TL_BK = 32u;
constant uint TL_STRIDE = 40u;
constant uint TL_BAND = 8u;

// Rounds four floats to bf16 bits, ties to even.
static inline ushort4 tl_bf16(float4 v) {
    uint4 b = as_type<uint4>(v);
    return ushort4((b + uint4(0x7FFFu) + ((b >> uint4(16u)) & uint4(1u))) >> uint4(16u));
}

static inline void tl_put4(threadgroup ushort *at, ushort4 v) {
    at[0] = v.x;
    at[1] = v.y;
    at[2] = v.z;
    at[3] = v.w;
}

// Stores one 8x8 sum, through scratch at an edge.
static inline void tl_store(
    simdgroup_float8x8 acc,
    device float *out,
    threadgroup float *scratch,
    uint r0,
    uint c0,
    uint m,
    uint n,
    uint lane) {
    if (r0 + 8u <= m && c0 + 8u <= n) {
        simdgroup_store(acc, out + ulong(r0) * n + c0, ulong(n));
        return;
    }
    if (r0 >= m || c0 >= n) return;
    simdgroup_store(acc, scratch, 8ul);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 64u; e += 32u) {
        uint r = r0 + (e >> 3u);
        uint c = c0 + (e & 7u);
        if (r < m && c < n) out[ulong(r) * n + c] = scratch[e];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// f32 rows times bf16 weights on bf16 matrix units.
kernel void tiled_metal_bf16(
    device const float *x [[buffer(0)]],
    device const ushort *w [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant uint &m [[buffer(3)]],
    constant uint &n [[buffer(4)]],
    constant uint &k [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat a_s[TL_BM][TL_STRIDE];
    threadgroup bfloat b_s[TL_BN][TL_STRIDE];
    threadgroup ushort *a_bits = (threadgroup ushort *)&a_s[0][0];
    threadgroup ushort *b_bits = (threadgroup ushort *)&b_s[0][0];

    uint simd = tid >> 5u;
    uint lane = tid & 31u;
    uint group = tg.x;
    // Down a band, then across: groups share a weight block.
    uint tiles_m = (m + TL_BM - 1u) / TL_BM;
    uint band = (n >= 2u * m) ? tiles_m : TL_BAND;
    uint per_band = band * ((n + TL_BN - 1u) / TL_BN);
    uint first = (group / per_band) * band;
    uint height = min(band, tiles_m - first);
    uint row0 = (first + (group % per_band) % height) * TL_BM;
    uint col0 = ((group % per_band) / height) * TL_BN;
    uint wm0 = (simd >> 2u) * 32u;
    uint wn0 = (simd & 3u) * 16u;

    simdgroup_float8x8 acc[4][2];
    for (uint i = 0u; i < 4u; i++) {
        acc[i][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc[i][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (uint k0 = 0u; k0 < k; k0 += TL_BK) {
        if (k0 > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < TL_BM * (TL_BK / 4u); i += 256u) {
            uint r = i >> 3u;
            uint c = (i & 7u) << 2u;
            ushort4 a = ushort4(0u);
            ushort4 b = ushort4(0u);
            if (k0 + c < k) {
                if (row0 + r < m) {
                    a = tl_bf16(*(device const float4 *)(x + ulong(row0 + r) * k + k0 + c));
                }
                if (col0 + r < n) {
                    b = ushort4(*(device const packed_ushort4 *)(w + ulong(col0 + r) * k + k0 + c));
                }
            }
            tl_put4(a_bits + r * TL_STRIDE + c, a);
            tl_put4(b_bits + r * TL_STRIDE + c, b);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k8 = 0u; k8 < TL_BK; k8 += 8u) {
            // Weight rows hold `k`, so they load transposed.
            simdgroup_bfloat8x8 tb0, tb1;
            simdgroup_load(tb0, &b_s[wn0][k8], TL_STRIDE, ulong2(0ul), true);
            simdgroup_load(tb1, &b_s[wn0 + 8u][k8], TL_STRIDE, ulong2(0ul), true);
            for (uint i = 0u; i < 4u; i++) {
                simdgroup_bfloat8x8 ta;
                simdgroup_load(ta, &a_s[wm0 + i * 8u][k8], TL_STRIDE);
                simdgroup_multiply_accumulate(acc[i][0], ta, tb0, acc[i][0]);
                simdgroup_multiply_accumulate(acc[i][1], ta, tb1, acc[i][1]);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *scratch = (threadgroup float *)&a_s[0][0] + simd * 64u;
    for (uint i = 0u; i < 4u; i++) {
        uint r0 = row0 + wm0 + i * 8u;
        tl_store(acc[i][0], out, scratch, r0, col0 + wn0, m, n, lane);
        tl_store(acc[i][1], out, scratch, r0, col0 + wn0 + 8u, m, n, lane);
    }
}
