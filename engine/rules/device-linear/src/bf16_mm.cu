#include <tessera/1/core>
#include <tessera/1/trellis>
#include "common.cuh"

#if defined(TESSERA_CUDA)

#ifndef MM_BF64_BLK
#define MM_BF64_BLK 4
#endif
#ifndef MM_BF64_SK_BLK
#define MM_BF64_SK_BLK 4
#endif
#ifndef MM_BF128_BLK
#define MM_BF128_BLK 2
#endif

template <bool SPLIT>
static __device__ __forceinline__ void bf_store_r64(const float acc[4][4],
                                                    const trellis::o_t* other, const u8* bias_p,
                                                    trellis::y_t* out, u32 m, u32 n, u32 r0,
                                                    u32 n0, u32 m_base, u32 n_base, u32 p, u32 g,
                                                    u32 t, u32 mode) {
    u32 col0 = n0 + n_base + g, col1 = col0 + 8u;
#pragma unroll
    for (u32 ti = 0u; ti < 4u; ti++) {
        u32 row0 = r0 + m_base + (ti << 3u) + (t << 1u);
        if (row0 < m) {
            u64 off0 = u64(SPLIT ? (p * m + row0) : row0) * n;
            if (col0 < n) {
                trellis::store_y(
                    out, off0 + col0,
                    SPLIT ? acc[ti][0] : dl_epilogue(acc[ti][0], other, off0 + col0, col0, mode, bias_p));
            }
            if (col1 < n) {
                trellis::store_y(
                    out, off0 + col1,
                    SPLIT ? acc[ti][2] : dl_epilogue(acc[ti][2], other, off0 + col1, col1, mode, bias_p));
            }
        }
        if (row0 + 1u < m) {
            u64 off1 = u64(SPLIT ? (p * m + row0 + 1u) : (row0 + 1u)) * n;
            if (col0 < n) {
                trellis::store_y(
                    out, off1 + col0,
                    SPLIT ? acc[ti][1] : dl_epilogue(acc[ti][1], other, off1 + col0, col0, mode, bias_p));
            }
            if (col1 < n) {
                trellis::store_y(
                    out, off1 + col1,
                    SPLIT ? acc[ti][3] : dl_epilogue(acc[ti][3], other, off1 + col1, col1, mode, bias_p));
            }
        }
    }
}

static __device__ __forceinline__ u32 bf_bf16x2_f16x2(u32 bw) {
    return trellis::f32x2_to_f16x2(__uint_as_float(bw << 16u), __uint_as_float(bw & 0xFFFF0000u));
}

struct alignas(16) bf64_smem {
    u32 sw[2][64][20];
    u32 sxh[64][20];
#if !defined(X_BF16)
    u32 sxl[64][20];
#endif
};

template <bool SPLIT>
static __device__ __forceinline__ void fg_bf16_r64(const trellis::x_t* x, const u16* w,
                                                   const trellis::o_t* other, const u8* bias_p,
                                                   trellis::y_t* out, u32 m, u32 n, u32 k,
                                                   u32 mode, u32 ks, bf64_smem& sm) {
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u, g = lane >> 2u, t = lane & 3u;
    u32 wm = warp >> 2u, wn = warp & 3u, nb = (n + 63u) >> 6u, mb = (m + 63u) >> 6u;
    u32 ob_in = SPLIT ? (blockIdx.x % nb) : blockIdx.x, p = SPLIT ? (blockIdx.x / nb) : 0u;
    if (ob_in >= nb || p >= ks || blockIdx.y >= mb) return;
    u32 bx = ob_in, by = blockIdx.y;
    if constexpr (!SPLIT) dl_swizzle(nb, mb, ob_in, blockIdx.y, bx, by);
    u32 n0 = bx << 6u, r0 = by << 6u, n_base = wn << 4u, m_base = wm << 5u;
    u32 steps = (k + 31u) >> 5u, st_per = SPLIT ? ((steps + ks - 1u) / ks) : steps;
    u32 s0 = SPLIT ? (p * st_per) : 0u, s1 = SPLIT ? min(steps, s0 + st_per) : steps;
    u32 wr = tid >> 2u, wc = tid & 3u, xr0 = tid >> 3u, xc = (tid & 7u) << 2u, xw = (tid & 7u) << 1u;
    float acc[4][4];
#pragma unroll
    for (u32 ti = 0u; ti < 4u; ti++) acc[ti][0] = acc[ti][1] = acc[ti][2] = acc[ti][3] = 0.0f;
    if (s0 < s1) {
        u32 k0 = s0 << 5u;
        dl_cp16(&sm.sw[0][wr][wc << 2u], w + u64(n0 + wr) * k + k0 + (wc << 3u),
                (n0 + wr < n) && (k0 + (wc << 3u) < k));
        dl_cp_commit();
    }
    for (u32 s = s0; s < s1; s++) {
        u32 st = (s - s0) & 1u, k0 = s << 5u;
        bool has_nxt = (s + 1u < s1);
        if (has_nxt) {
            u32 kn = (s + 1u) << 5u;
            dl_cp16(&sm.sw[st ^ 1u][wr][wc << 2u], w + u64(n0 + wr) * k + kn + (wc << 3u),
                    (n0 + wr < n) && (kn + (wc << 3u) < k));
            dl_cp_commit();
        }
#pragma unroll
        for (u32 dr = 0u; dr < 64u; dr += 32u) {
            u32 r = xr0 + dr;
            bool ok = (r0 + r < m) && (k0 + xc < k);
            const trellis::x_t* xp = x + u64(r0 + r) * k + k0 + xc;
#if defined(X_BF16)
            dl_stage_bf16_x4(xp, ok, &sm.sxh[r][xw]);
#else
            dl_stage_bf16_x4(xp, ok, &sm.sxh[r][xw], &sm.sxl[r][xw]);
#endif
        }
        if (has_nxt) dl_cp_wait1(); else dl_cp_wait0();
        prim::block_sync();
#pragma unroll
        for (u32 ks_i = 0u; ks_i < 2u; ks_i++) {
            u32 off = (ks_i << 3u) + t;
            u32 a[4] = {sm.sw[st][n_base + g][off], sm.sw[st][n_base + g + 8u][off],
                        sm.sw[st][n_base + g][off + 4u], sm.sw[st][n_base + g + 8u][off + 4u]};
#pragma unroll
            for (u32 ti = 0u; ti < 4u; ti++) {
                u32 xr = m_base + (ti << 3u) + g;
                u32 bh[2] = {sm.sxh[xr][off], sm.sxh[xr][off + 4u]};
                dl_mma_bf16(acc[ti], a, bh);
#if !defined(X_BF16)
                u32 bl[2] = {sm.sxl[xr][off], sm.sxl[xr][off + 4u]};
                dl_mma_bf16(acc[ti], a, bl);
#endif
            }
        }
        if (has_nxt) prim::block_sync();
    }
    bf_store_r64<SPLIT>(acc, other, bias_p, out, m, n, r0, n0, m_base, n_base, p, g, t, mode);
}

struct alignas(16) bf128_smem {
    u32 sw[2][128][20];
    u32 sx[2][128][20];
};

// Staged BF16->FP16 Tensor-Core matmul, 128x128 per block.
extern "C" __global__ __launch_bounds__(256, MM_BF128_BLK) void tessera_mm_bf128(
    const trellis::x_t* x, const u16* w, const trellis::o_t* other, const u8* bias_p,
    trellis::y_t* out, u32 m, u32 n, u32 k, u32 mode) {
    __shared__ bf128_smem sm;
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u, g = lane >> 2u, t = lane & 3u;
    u32 wm = warp >> 1u, wn = warp & 1u, nb = (n + 127u) >> 7u, mb = (m + 127u) >> 7u;
    if (blockIdx.x >= nb || blockIdx.y >= mb) return;
    u32 bx, by;
    dl_swizzle(nb, mb, blockIdx.x, blockIdx.y, bx, by);
    u32 n0 = bx << 7u, r0 = by << 7u, n_base = wn << 6u, m_base = wm << 5u;
    u32 steps = (k + 31u) >> 5u, wr0 = tid >> 2u, wc = tid & 3u;
    u32 xr0 = tid >> 3u, xc = (tid & 7u) << 2u, xw = (tid & 7u) << 1u;
    float acc[4][4][4];
#pragma unroll
    for (u32 ni = 0u; ni < 4u; ni++)
#pragma unroll
        for (u32 mi = 0u; mi < 4u; mi++)
            acc[ni][mi][0] = acc[ni][mi][1] = acc[ni][mi][2] = acc[ni][mi][3] = 0.0f;
    if (steps > 0u) {
        dl_cp16(&sm.sw[0][wr0][wc << 2u], w + u64(n0 + wr0) * k + (wc << 3u), (n0 + wr0 < n) && ((wc << 3u) < k));
        dl_cp16(&sm.sw[0][wr0 + 64u][wc << 2u], w + u64(n0 + wr0 + 64u) * k + (wc << 3u), (n0 + wr0 + 64u < n) && ((wc << 3u) < k));
        dl_cp_commit();
    }
    for (u32 s = 0u; s < steps; s++) {
        u32 st = s & 1u, k0 = s << 5u;
        bool has_nxt = (s + 1u < steps);
        if (has_nxt) {
            u32 kn = (s + 1u) << 5u;
            dl_cp16(&sm.sw[st ^ 1u][wr0][wc << 2u], w + u64(n0 + wr0) * k + kn + (wc << 3u), (n0 + wr0 < n) && (kn + (wc << 3u) < k));
            dl_cp16(&sm.sw[st ^ 1u][wr0 + 64u][wc << 2u], w + u64(n0 + wr0 + 64u) * k + kn + (wc << 3u), (n0 + wr0 + 64u < n) && (kn + (wc << 3u) < k));
            dl_cp_commit();
        }
#pragma unroll
        for (u32 dr = 0u; dr < 128u; dr += 32u) {
            u32 r = xr0 + dr;
            if (r0 + r < m && k0 + xc < k) {
                float4 xv = trellis::load_x4(x + u64(r0 + r) * k + k0 + xc);
                *reinterpret_cast<uint2*>(&sm.sx[st][r][xw]) = make_uint2(
                    trellis::f32x2_to_f16x2(xv.x, xv.y), trellis::f32x2_to_f16x2(xv.z, xv.w));
            } else {
                *reinterpret_cast<uint2*>(&sm.sx[st][r][xw]) = make_uint2(0u, 0u);
            }
        }
        if (has_nxt) dl_cp_wait1(); else dl_cp_wait0();
        prim::block_sync();
#pragma unroll
        for (u32 ks_i = 0u; ks_i < 2u; ks_i++) {
            u32 off = (ks_i << 3u) + t;
            u32 b[4][2];
#pragma unroll
            for (u32 mi = 0u; mi < 4u; mi++) {
                u32 xr = m_base + (mi << 3u) + g;
                b[mi][0] = sm.sx[st][xr][off]; b[mi][1] = sm.sx[st][xr][off + 4u];
            }
#pragma unroll
            for (u32 ni = 0u; ni < 4u; ni++) {
                u32 nr = n_base + (ni << 4u) + g;
                u32 a[4] = {bf_bf16x2_f16x2(sm.sw[st][nr][off]), bf_bf16x2_f16x2(sm.sw[st][nr + 8u][off]),
                            bf_bf16x2_f16x2(sm.sw[st][nr][off + 4u]), bf_bf16x2_f16x2(sm.sw[st][nr + 8u][off + 4u])};
#pragma unroll
                for (u32 mi = 0u; mi < 4u; mi++) trellis::mma_m16n8k16(acc[ni][mi], a, b[mi]);
            }
        }
        if (has_nxt) prim::block_sync();
    }
#pragma unroll
    for (u32 ni = 0u; ni < 4u; ni++) {
        bf_store_r64<false>(acc[ni], other, bias_p, out, m, n, r0, n0, m_base, n_base + (ni << 4u),
                            0u, g, t, mode);
    }
}

// Staged BF16 Tensor-Core matmul, 64x64 per block.
extern "C" __global__ __launch_bounds__(256, MM_BF64_BLK) void tessera_mm_bf64(
    const trellis::x_t* x, const u16* w, const trellis::o_t* other, const u8* bias_p,
    trellis::y_t* out, u32 m, u32 n, u32 k, u32 mode) {
    __shared__ bf64_smem sm;
    fg_bf16_r64<false>(x, w, other, bias_p, out, m, n, k, mode, 1u, sm);
}

// Split-K staged BF16 Tensor-Core matmul, 64x64 per block.
extern "C" __global__ __launch_bounds__(256, MM_BF64_SK_BLK) void tessera_mm_bf64_sk(
    const trellis::x_t* x, const u16* w, const trellis::o_t* other, const u8* bias_p,
    trellis::y_t* out, u32 m, u32 n, u32 k, u32 mode, u32 ks) {
    __shared__ bf64_smem sm;
    fg_bf16_r64<true>(x, w, other, bias_p, out, m, n, k, mode, ks, sm);
}

#endif
