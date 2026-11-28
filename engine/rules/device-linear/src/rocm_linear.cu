// RDNA 3/3.5 HIP GEMV and single-buffered WMMA GEMM kernels.

#include <tessera/1/core>

using namespace tessera;

#if defined(TESSERA_HIP)

namespace {

typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef short bf16x16 __attribute__((ext_vector_type(16)));

TESSERA_FN float wave_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
    return v;
}

TESSERA_FN float f16_to_f32(u16 h) {
    return static_cast<float>(__builtin_bit_cast(_Float16, h));
}

// Branchless RNE f32 to bf16 for finite activations.
TESSERA_FN short to_bf16_fast(float x) {
    u32 b = __float_as_uint(x);
    return static_cast<short>((b + 0x7FFFu + ((b >> 16u) & 1u)) >> 16u);
}

TESSERA_FN float apply_gemv_act(float v, u32 mode, float a, float b) {
    if (mode == 1u) {
        float t = a * v;
        return t / (1.0f + expf(-t));
    }
    if (mode == 2u) return b / (1.0f + expf(-(a * v)));
    return v;
}

TESSERA_FN float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

TESSERA_FN float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475244f));
}

TESSERA_FN float load_bias_1(const u8* p, u32 col, bool bf16) {
    return bf16 ? prim::from_bf16(reinterpret_cast<const u16*>(p)[col])
                : reinterpret_cast<const float*>(p)[col];
}

TESSERA_FN float apply_mm_act(float v, u32 mode, const float* other, const u8* bias_p, u64 idx,
                              u32 col) {
    u32 base = mode & 15u;
    bool bf16 = (mode & 16u) != 0u;
    if (base == 1u) return v + other[idx];
    if (base == 2u) return v + load_bias_1(reinterpret_cast<const u8*>(other), col, bf16);
    if (base == 3u) return (v / (1.0f + expf(-v))) * other[idx];
    if (base == 4u) return v * other[idx];
    if (base == 5u) return v / (1.0f + expf(-v));
    if (base == 6u) {
        return gelu_tanh(v + load_bias_1(reinterpret_cast<const u8*>(other), col, bf16));
    }
    if (base == 7u) {
        return gelu_exact(v + load_bias_1(reinterpret_cast<const u8*>(other), col, bf16));
    }
    if (base == 8u) return v + load_bias_1(bias_p, col, bf16) + other[idx];
    return v;
}

TESSERA_FN float dot8_bf16(uint4 wv, const float* xp) {
    float4 x0 = *reinterpret_cast<const float4*>(xp);
    float4 x1 = *reinterpret_cast<const float4*>(xp + 4u);
    float s = prim::from_bf16(static_cast<u16>(wv.x)) * x0.x;
    s += prim::from_bf16(static_cast<u16>(wv.x >> 16u)) * x0.y;
    s += prim::from_bf16(static_cast<u16>(wv.y)) * x0.z;
    s += prim::from_bf16(static_cast<u16>(wv.y >> 16u)) * x0.w;
    s += prim::from_bf16(static_cast<u16>(wv.z)) * x1.x;
    s += prim::from_bf16(static_cast<u16>(wv.z >> 16u)) * x1.y;
    s += prim::from_bf16(static_cast<u16>(wv.w)) * x1.z;
    s += prim::from_bf16(static_cast<u16>(wv.w >> 16u)) * x1.w;
    return s;
}

TESSERA_FN void store_tile_act(v8f a, const float* other, const u8* bias_p, float* y, u32 rm,
                               u32 col, u32 m, u32 n, u32 mode, u32 half_r) {
    if (col >= n) return;
    u32 base = mode & 15u;
    bool bf16 = (mode & 16u) != 0u;
    float bv = 0.0f;
    if (base == 2u || base == 6u || base == 7u) {
        bv = load_bias_1(reinterpret_cast<const u8*>(other), col, bf16);
    } else if (base == 8u) {
        bv = load_bias_1(bias_p, col, bf16);
    }
#pragma unroll
    for (u32 e = 0u; e < 8u; e++) {
        u32 row = rm + 2u * e + half_r;
        if (row < m) {
            u64 idx = u64(row) * n + col;
            float v = a[e];
            if (base == 1u) v += other[idx];
            else if (base == 2u) v += bv;
            else if (base == 3u) v = (v / (1.0f + expf(-v))) * other[idx];
            else if (base == 4u) v *= other[idx];
            else if (base == 5u) v = v / (1.0f + expf(-v));
            else if (base == 6u) v = gelu_tanh(v + bv);
            else if (base == 7u) v = gelu_exact(v + bv);
            else if (base == 8u) v += bv + other[idx];
            y[idx] = v;
        }
    }
}

template <u32 WM, u32 WN, bool SPLIT>
TESSERA_FN void store_wmma_grid(const v8f acc[WM][WN], const float* other, const u8* bias_p,
                                float* y, u32 r0, u32 c0, u32 wm_id, u32 wn_id, u32 m, u32 n,
                                u32 mode, u32 split, u32 lane) {
    u32 half_r = lane >> 4u;
    u32 lane16 = lane & 15u;
    if constexpr (SPLIT) {
#pragma unroll
        for (u32 mi = 0u; mi < WM; mi++) {
#pragma unroll
            for (u32 ni = 0u; ni < WN; ni++) {
                u32 col = c0 + (wn_id * WN + ni) * 16u + lane16;
                if (col < n) {
                    u32 rm = r0 + (wm_id * WM + mi) * 16u;
#pragma unroll
                    for (u32 e = 0u; e < 8u; e++) {
                        u32 row = rm + 2u * e + half_r;
                        if (row < m) y[(u64(split) * m + row) * n + col] = acc[mi][ni][e];
                    }
                }
            }
        }
        return;
    }
    if (mode == 0u) {
#pragma unroll
        for (u32 mi = 0u; mi < WM; mi++) {
#pragma unroll
            for (u32 ni = 0u; ni < WN; ni++) {
                u32 col = c0 + (wn_id * WN + ni) * 16u + lane16;
                if (col < n) {
                    u32 rm = r0 + (wm_id * WM + mi) * 16u;
#pragma unroll
                    for (u32 e = 0u; e < 8u; e++) {
                        u32 row = rm + 2u * e + half_r;
                        if (row < m) y[u64(row) * n + col] = acc[mi][ni][e];
                    }
                }
            }
        }
        return;
    }
#pragma unroll
    for (u32 mi = 0u; mi < WM; mi++) {
#pragma unroll
        for (u32 ni = 0u; ni < WN; ni++) {
            u32 col = c0 + (wn_id * WN + ni) * 16u + lane16;
            u32 rm = r0 + (wm_id * WM + mi) * 16u;
            store_tile_act(acc[mi][ni], other, bias_p, y, rm, col, m, n, mode, half_r);
        }
    }
}

// Clamps finite f32 to f16 range and converts.
TESSERA_FN _Float16 to_f16_safe(float x) {
    return static_cast<_Float16>(fminf(fmaxf(x, -65504.0f), 65504.0f));
}

// Loads FP16 or BF16 activation tile into LDS buffers.
template <u32 BM, u32 NT, bool HI_LO, bool AS_F16, typename Dst>
TESSERA_FN void load_x_tile(const float* x, Dst s_xh[BM][40], Dst s_xl[HI_LO ? BM : 1u][40],
                            u32 r0, u32 kb, u32 m, u32 k, u32 tid) {
    typedef _Float16 f16x4 __attribute__((ext_vector_type(4)));
    constexpr u32 RSTEP = NT / 8u;
    u32 tr = tid >> 3u;
    u32 tc = (tid & 7u) << 2u;
#pragma unroll
    for (u32 s = 0u; s < BM / RSTEP; s++) {
        u32 rl = s * RSTEP + tr;
        u32 row = r0 + rl;
        u32 ck = kb + tc;
        float4 v = (row < m && ck < k) ? *reinterpret_cast<const float4*>(x + u64(row) * k + ck)
                                       : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if constexpr (AS_F16) {
            _Float16 h0 = to_f16_safe(v.x);
            _Float16 h1 = to_f16_safe(v.y);
            _Float16 h2 = to_f16_safe(v.z);
            _Float16 h3 = to_f16_safe(v.w);
            *reinterpret_cast<f16x4*>(&s_xh[rl][tc]) = (f16x4){h0, h1, h2, h3};
            if constexpr (HI_LO) {
                *reinterpret_cast<f16x4*>(&s_xl[rl][tc]) = (f16x4){
                    to_f16_safe(v.x - float(h0)), to_f16_safe(v.y - float(h1)),
                    to_f16_safe(v.z - float(h2)), to_f16_safe(v.w - float(h3))};
            }
        } else {
            u16 h0 = static_cast<u16>(to_bf16_fast(v.x));
            u16 h1 = static_cast<u16>(to_bf16_fast(v.y));
            u16 h2 = static_cast<u16>(to_bf16_fast(v.z));
            u16 h3 = static_cast<u16>(to_bf16_fast(v.w));
            *reinterpret_cast<uint2*>(&s_xh[rl][tc]) =
                make_uint2(u32(h0) | (u32(h1) << 16u), u32(h2) | (u32(h3) << 16u));
            if constexpr (HI_LO) {
                u16 l0 = static_cast<u16>(to_bf16_fast(v.x - prim::from_bf16(h0)));
                u16 l1 = static_cast<u16>(to_bf16_fast(v.y - prim::from_bf16(h1)));
                u16 l2 = static_cast<u16>(to_bf16_fast(v.z - prim::from_bf16(h2)));
                u16 l3 = static_cast<u16>(to_bf16_fast(v.w - prim::from_bf16(h3)));
                *reinterpret_cast<uint2*>(&s_xl[rl][tc]) =
                    make_uint2(u32(l0) | (u32(l1) << 16u), u32(l2) | (u32(l3) << 16u));
            }
        }
    }
}

// Loads BF16 weights into BF16 or FP16 LDS.
template <u32 BN, u32 NT, bool TO_F16, typename Dst>
TESSERA_FN void load_w_bf16(const u16* w, Dst s_w[BN][40], u32 c0, u32 kb, u32 n, u32 k, u32 tid) {
    typedef _Float16 f16x4 __attribute__((ext_vector_type(4)));
    constexpr u32 CSTEP = NT / 4u;
    u32 wr = tid >> 2u;
    u32 wc = (tid & 3u) << 3u;
#pragma unroll
    for (u32 s = 0u; s < BN / CSTEP; s++) {
        u32 cl = s * CSTEP + wr;
        u32 col = c0 + cl;
        u32 ck = kb + wc;
        uint4 wv = (col < n && ck < k) ? *reinterpret_cast<const uint4*>(w + u64(col) * k + ck)
                                       : make_uint4(0u, 0u, 0u, 0u);
        if constexpr (TO_F16) {
            *reinterpret_cast<f16x4*>(&s_w[cl][wc]) = (f16x4){
                to_f16_safe(prim::from_bf16(static_cast<u16>(wv.x))),
                to_f16_safe(prim::from_bf16(static_cast<u16>(wv.x >> 16u))),
                to_f16_safe(prim::from_bf16(static_cast<u16>(wv.y))),
                to_f16_safe(prim::from_bf16(static_cast<u16>(wv.y >> 16u)))};
            *reinterpret_cast<f16x4*>(&s_w[cl][wc + 4u]) = (f16x4){
                to_f16_safe(prim::from_bf16(static_cast<u16>(wv.z))),
                to_f16_safe(prim::from_bf16(static_cast<u16>(wv.z >> 16u))),
                to_f16_safe(prim::from_bf16(static_cast<u16>(wv.w))),
                to_f16_safe(prim::from_bf16(static_cast<u16>(wv.w >> 16u)))};
        } else {
            *reinterpret_cast<uint4*>(&s_w[cl][wc]) = wv;
        }
    }
}

// Loads INT4_G32 (pre-scaled) or MXFP8 weight block into FP16 LDS.
template <bool IS_MXFP8, u32 BN, u32 NT>
TESSERA_FN void load_w_blk(const u8* w, _Float16 s_w[BN][40], float s_sc[IS_MXFP8 ? BN : 1u],
                           const _Float16* lut, u32 c0, u32 g, u32 ng, u32 n, u32 tid) {
    constexpr u32 TPC = NT / BN;
    constexpr u32 ELEMS = 32u / TPC;
    constexpr u32 BLK_BYTES = IS_MXFP8 ? 33u : 18u;
    u32 cl = tid / TPC;
    u32 part = tid % TPC;
    u32 col = c0 + cl;
    _Float16* wd = &s_w[cl][part * ELEMS];
    if (col < n) {
        const u8* bp = w + (u64(col) * ng + g) * BLK_BYTES;
        if constexpr (IS_MXFP8) {
            if (part == 0u) s_sc[cl] = __uint_as_float(u32(bp[0]) << 23u);
            const u8* qp = bp + 1u + part * ELEMS;
#pragma unroll
            for (u32 i = 0u; i < ELEMS; i++) wd[i] = lut[qp[i]];
        } else {
            const u16* b16 = reinterpret_cast<const u16*>(bp);
            _Float16 hsc = __builtin_bit_cast(_Float16, b16[0]);
            const u16* qp = b16 + 1u + part * (ELEMS / 4u);
#pragma unroll
            for (u32 q = 0u; q < ELEMS / 4u; q++) {
                u32 w2 = qp[q];
                wd[q * 4u] = hsc * static_cast<_Float16>(int(w2 & 15u) - 8);
                wd[q * 4u + 1u] = hsc * static_cast<_Float16>(int((w2 >> 4u) & 15u) - 8);
                wd[q * 4u + 2u] = hsc * static_cast<_Float16>(int((w2 >> 8u) & 15u) - 8);
                wd[q * 4u + 3u] = hsc * static_cast<_Float16>(int((w2 >> 12u) & 15u) - 8);
            }
        }
    } else {
        if constexpr (IS_MXFP8) {
            if (part == 0u) s_sc[cl] = 0.0f;
        }
#pragma unroll
        for (u32 i = 0u; i < ELEMS; i++) wd[i] = static_cast<_Float16>(0.0f);
    }
}

template <u32 BM, u32 BN, u32 NT, bool SPLIT>
TESSERA_FN void rocm_mm_bf16_run(const float* x, const u16* w, const float* other,
                                 const u8* bias_p, float* y, u32 m, u32 n, u32 k, u32 mode,
                                 u32 ks) {
    u32 tid = threadIdx.x;
    u32 warp = tid >> 5u;
    u32 lane = tid & 31u;
    u32 bx = blockIdx.x;
    u32 bn = SPLIT ? bx / ks : bx;
    u32 split = SPLIT ? bx - bn * ks : 0u;
    u32 r0 = blockIdx.y * BM;
    u32 c0 = bn * BN;
    u32 k_chunks = (k + 31u) >> 5u;
    u32 step0 = SPLIT ? (split * k_chunks) / ks : 0u;
    u32 step1 = SPLIT ? ((split + 1u) * k_chunks) / ks : k_chunks;

    constexpr bool USE_F16 = (BM >= 64u);
    constexpr u32 WM = BM == 128u ? 4u : (BM == 64u ? 2u : 1u);
    constexpr u32 WN = BM == 16u ? 1u : 2u;
    u32 wm_id = BM == 128u ? (warp >> 2u) : (BM == 64u ? (warp >> 1u) : 0u);
    u32 wn_id = BM == 128u ? (warp & 3u) : (BM == 64u ? (warp & 1u) : warp);

    v8f acc[WM][WN];
#pragma unroll
    for (u32 mi = 0u; mi < WM; mi++) {
#pragma unroll
        for (u32 ni = 0u; ni < WN; ni++) acc[mi][ni] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    }

    u32 lane16 = lane & 15u;
    if constexpr (USE_F16) {
        __shared__ alignas(16) _Float16 s_xh[BM][40];
        __shared__ alignas(16) _Float16 s_xl[1][40];
        __shared__ alignas(16) _Float16 s_w[BN][40];
        for (u32 st = step0; st < step1; st++) {
            u32 kb = st * 32u;
            load_x_tile<BM, NT, false, true>(x, s_xh, s_xl, r0, kb, m, k, tid);
            load_w_bf16<BN, NT, true>(w, s_w, c0, kb, n, k, tid);
            prim::block_sync();
#pragma unroll
            for (u32 kk = 0u; kk < 32u; kk += 16u) {
                f16x16 wf[WN];
#pragma unroll
                for (u32 ni = 0u; ni < WN; ni++) {
                    u32 lc = (wn_id * WN + ni) * 16u + lane16;
                    wf[ni] = *reinterpret_cast<const f16x16*>(&s_w[lc][kk]);
                }
#pragma unroll
                for (u32 mi = 0u; mi < WM; mi++) {
                    u32 lr = (wm_id * WM + mi) * 16u + lane16;
                    f16x16 xh = *reinterpret_cast<const f16x16*>(&s_xh[lr][kk]);
#pragma unroll
                    for (u32 ni = 0u; ni < WN; ni++) {
                        acc[mi][ni] =
                            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xh, wf[ni], acc[mi][ni]);
                    }
                }
            }
            prim::block_sync();
        }
    } else {
        __shared__ alignas(16) short s_xh[BM][40];
        __shared__ alignas(16) short s_xl[BM][40];
        __shared__ alignas(16) short s_w[BN][40];
        for (u32 st = step0; st < step1; st++) {
            u32 kb = st * 32u;
            load_x_tile<BM, NT, true, false>(x, s_xh, s_xl, r0, kb, m, k, tid);
            load_w_bf16<BN, NT, false>(w, s_w, c0, kb, n, k, tid);
            prim::block_sync();
#pragma unroll
            for (u32 kk = 0u; kk < 32u; kk += 16u) {
                bf16x16 wf = *reinterpret_cast<const bf16x16*>(&s_w[wn_id * 16u + lane16][kk]);
                bf16x16 xh = *reinterpret_cast<const bf16x16*>(&s_xh[lane16][kk]);
                bf16x16 xl = *reinterpret_cast<const bf16x16*>(&s_xl[lane16][kk]);
                acc[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(xh, wf, acc[0][0]);
                acc[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(xl, wf, acc[0][0]);
            }
            prim::block_sync();
        }
    }

    store_wmma_grid<WM, WN, SPLIT>(acc, other, bias_p, y, r0, c0, wm_id, wn_id, m, n, mode, split,
                                   lane);
}

template <u32 WM, u32 WN, u32 BM, u32 BN, bool HI_LO>
TESSERA_FN void wmma_blk_k32(const _Float16 (&s_xh)[BM][40],
                             const _Float16 (&s_xl)[HI_LO ? BM : 1u][40],
                             const _Float16 (&s_w)[BN][40], u32 wm_id, u32 wn_id, u32 lane16,
                             v8f (&dst)[WM][WN]) {
#pragma unroll
    for (u32 kk = 0u; kk < 32u; kk += 16u) {
        f16x16 xh[WM], wf[WN];
#pragma unroll
        for (u32 mi = 0u; mi < WM; mi++) {
            u32 lr = (wm_id * WM + mi) * 16u + lane16;
            xh[mi] = *reinterpret_cast<const f16x16*>(&s_xh[lr][kk]);
        }
#pragma unroll
        for (u32 ni = 0u; ni < WN; ni++) {
            u32 lc = (wn_id * WN + ni) * 16u + lane16;
            wf[ni] = *reinterpret_cast<const f16x16*>(&s_w[lc][kk]);
        }
#pragma unroll
        for (u32 mi = 0u; mi < WM; mi++) {
#pragma unroll
            for (u32 ni = 0u; ni < WN; ni++) {
                dst[mi][ni] =
                    __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xh[mi], wf[ni], dst[mi][ni]);
            }
        }
        if constexpr (HI_LO) {
            f16x16 xl = *reinterpret_cast<const f16x16*>(&s_xl[lane16][kk]);
            dst[0][0] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(xl, wf[0], dst[0][0]);
        }
    }
}

template <bool IS_MXFP8, u32 BM, u32 BN, u32 NT, bool SPLIT>
TESSERA_FN void rocm_mm_blk_run(const float* x, const u8* w, const float* other, const u8* bias_p,
                                float* y, u32 m, u32 n, u32 k, u32 mode, u32 ks) {
    u32 tid = threadIdx.x;
    u32 warp = tid >> 5u;
    u32 lane = tid & 31u;
    u32 bx = blockIdx.x;
    u32 bn = SPLIT ? bx / ks : bx;
    u32 split = SPLIT ? bx - bn * ks : 0u;
    u32 r0 = blockIdx.y * BM;
    u32 c0 = bn * BN;
    u32 ng = k >> 5u;
    u32 g0 = SPLIT ? (split * ng) / ks : 0u;
    u32 g1 = SPLIT ? ((split + 1u) * ng) / ks : ng;

    constexpr bool HI_LO = (BM == 16u);
    __shared__ alignas(16) _Float16 s_xh[BM][40];
    __shared__ alignas(16) _Float16 s_xl[HI_LO ? BM : 1u][40];
    __shared__ alignas(16) _Float16 s_w[BN][40];
    __shared__ float s_sc[IS_MXFP8 ? BN : 1u];
    __shared__ _Float16 lut[IS_MXFP8 ? 256u : 1u];

    if constexpr (IS_MXFP8) {
        for (u32 i = tid; i < 256u; i += NT) {
            lut[i] = static_cast<_Float16>(prim::from_e4m3(static_cast<u8>(i)));
        }
        prim::block_sync();
    }

    constexpr u32 WARPS_N = (BM == 16u) ? 4u : (BN == 128u ? 4u : 2u);
    constexpr u32 WARPS_M = (NT / 32u) / WARPS_N;
    constexpr u32 WM = BM / (WARPS_M * 16u);
    constexpr u32 WN = BN / (WARPS_N * 16u);
    u32 wm_id = warp / WARPS_N;
    u32 wn_id = warp % WARPS_N;

    v8f acc[WM][WN];
#pragma unroll
    for (u32 mi = 0u; mi < WM; mi++) {
#pragma unroll
        for (u32 ni = 0u; ni < WN; ni++) acc[mi][ni] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    }

    u32 lane16 = lane & 15u;
    for (u32 g = g0; g < g1; g++) {
        load_x_tile<BM, NT, HI_LO, true>(x, s_xh, s_xl, r0, g << 5u, m, k, tid);
        load_w_blk<IS_MXFP8, BN, NT>(w, s_w, s_sc, lut, c0, g, ng, n, tid);
        prim::block_sync();
        if constexpr (!IS_MXFP8) {
            wmma_blk_k32<WM, WN, BM, BN, HI_LO>(s_xh, s_xl, s_w, wm_id, wn_id, lane16, acc);
        } else {
            v8f tmp[WM][WN];
#pragma unroll
            for (u32 mi = 0u; mi < WM; mi++) {
#pragma unroll
                for (u32 ni = 0u; ni < WN; ni++) tmp[mi][ni] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
            }
            wmma_blk_k32<WM, WN, BM, BN, HI_LO>(s_xh, s_xl, s_w, wm_id, wn_id, lane16, tmp);
#pragma unroll
            for (u32 ni = 0u; ni < WN; ni++) {
                float sc = s_sc[(wn_id * WN + ni) * 16u + lane16];
#pragma unroll
                for (u32 mi = 0u; mi < WM; mi++) {
#pragma unroll
                    for (u32 e = 0u; e < 8u; e++) acc[mi][ni][e] += sc * tmp[mi][ni][e];
                }
            }
        }
        prim::block_sync();
    }

    store_wmma_grid<WM, WN, SPLIT>(acc, other, bias_p, y, r0, c0, wm_id, wn_id, m, n, mode, split,
                                   lane);
}

TESSERA_FN float dot_int4_blk(const u8* bp, const float* xp) {
    float sc = f16_to_f32(u16(bp[0]) | (u16(bp[1]) << 8u));
    float part = 0.0f;
#pragma unroll
    for (u32 q = 0u; q < 4u; q++) {
        u32 w4 = u32(bp[2u + q * 4u]) | (u32(bp[3u + q * 4u]) << 8u) |
                 (u32(bp[4u + q * 4u]) << 16u) | (u32(bp[5u + q * 4u]) << 24u);
        float4 x0 = *reinterpret_cast<const float4*>(xp + q * 8u);
        float4 x1 = *reinterpret_cast<const float4*>(xp + q * 8u + 4u);
        part += float(int(w4 & 15u) - 8) * x0.x;
        part += float(int((w4 >> 4u) & 15u) - 8) * x0.y;
        part += float(int((w4 >> 8u) & 15u) - 8) * x0.z;
        part += float(int((w4 >> 12u) & 15u) - 8) * x0.w;
        part += float(int((w4 >> 16u) & 15u) - 8) * x1.x;
        part += float(int((w4 >> 20u) & 15u) - 8) * x1.y;
        part += float(int((w4 >> 24u) & 15u) - 8) * x1.z;
        part += float(int((w4 >> 28u) & 15u) - 8) * x1.w;
    }
    return sc * part;
}

TESSERA_FN float dot_mxfp8_blk(const u8* bp, const float* xp, const float* lut) {
    float sc = __uint_as_float(u32(bp[0]) << 23u);
    float part = 0.0f;
#pragma unroll
    for (u32 q = 0u; q < 8u; q++) {
        float4 xv = *reinterpret_cast<const float4*>(xp + q * 4u);
        part += lut[bp[1u + q * 4u]] * xv.x;
        part += lut[bp[2u + q * 4u]] * xv.y;
        part += lut[bp[3u + q * 4u]] * xv.z;
        part += lut[bp[4u + q * 4u]] * xv.w;
    }
    return sc * part;
}

TESSERA_FN float gemv_int4_row(const float* x, const u8* wr, u32 ng, u32 lane, u32* dst) {
    float sum = 0.0f;
    u32 g = 0u;
    for (; g + 32u <= ng; g += 32u) {
        const u32* src = reinterpret_cast<const u32*>(wr + u64(g) * 18u);
        for (u32 idx = lane; idx < 144u; idx += 32u) dst[idx] = src[idx];
        asm volatile("" ::: "memory");
        const u8* bp = reinterpret_cast<const u8*>(dst) + lane * 18u;
        sum += dot_int4_blk(bp, x + ((g + lane) << 5u));
    }
    if (g + lane < ng) {
        sum += dot_int4_blk(wr + u64(g + lane) * 18u, x + ((g + lane) << 5u));
    }
    return wave_sum(sum);
}

TESSERA_FN float gemv_mxfp8_row(const float* x, const u8* wr, u32 ng, u32 lane, u32* dst,
                                const float* lut) {
    float sum = 0.0f;
    u32 g = 0u;
    for (; g + 32u <= ng; g += 32u) {
        const u32* src = reinterpret_cast<const u32*>(wr + u64(g) * 33u);
        for (u32 idx = lane; idx < 264u; idx += 32u) dst[idx] = src[idx];
        asm volatile("" ::: "memory");
        const u8* bp = reinterpret_cast<const u8*>(dst) + lane * 33u;
        sum += dot_mxfp8_blk(bp, x + ((g + lane) << 5u), lut);
    }
    if (g + lane < ng) {
        sum += dot_mxfp8_blk(wr + u64(g + lane) * 33u, x + ((g + lane) << 5u), lut);
    }
    return wave_sum(sum);
}

}  // namespace

extern "C" __global__ __launch_bounds__(64) void rocm_gemv_bf16(
    const float* x, const u16* w, float* y, u32 n, u32 k, u32 mode, float a, float b) {
    u32 warp = threadIdx.x >> 5u;
    u32 lane = threadIdx.x & 31u;
    u32 base = blockIdx.x * 16u + warp * 8u;
    for (u32 off = 0u; off < 8u; off += 4u) {
        u32 r0 = base + off;
        if (r0 >= n) return;
        u32 cnt = n - r0;
        if (cnt > 4u) cnt = 4u;
        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (u32 kk = lane * 8u; kk < k; kk += 256u) {
            const float* xp = x + kk;
#pragma unroll
            for (u32 r = 0u; r < 4u; r++) {
                if (r < cnt) {
                    s[r] += dot8_bf16(*reinterpret_cast<const uint4*>(w + u64(r0 + r) * k + kk), xp);
                }
            }
        }
#pragma unroll
        for (u32 r = 0u; r < 4u; r++) {
            float v = wave_sum(s[r]);
            if (lane == 0u && r < cnt) y[r0 + r] = apply_gemv_act(v, mode, a, b);
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void rocm_gemv_bf16_narrow(
    const float* x, const u16* w, float* y, u32 n, u32 k, u32 mode, float a, float b) {
    u32 row = blockIdx.x;
    if (row >= n) return;
    u32 tid = threadIdx.x;
    u32 warp = tid >> 5u;
    u32 lane = tid & 31u;
    const u16* wr = w + u64(row) * k;
    float sum = 0.0f;
    for (u32 kk = tid * 8u; kk < k; kk += 2048u) {
        uint4 wv = *reinterpret_cast<const uint4*>(wr + kk);
        sum += dot8_bf16(wv, x + kk);
    }
    sum = wave_sum(sum);
    __shared__ float partial[8];
    if (lane == 0u) partial[warp] = sum;
    prim::block_sync();
    if (tid == 0u) {
        float total = 0.0f;
#pragma unroll
        for (u32 i = 0u; i < 8u; i++) total += partial[i];
        y[row] = apply_gemv_act(total, mode, a, b);
    }
}

extern "C" __global__ __launch_bounds__(64) void rocm_gemv_int4(
    const float* x, const u8* w, float* y, u32 n, u32 k) {
    u32 warp = threadIdx.x >> 5u;
    u32 lane = threadIdx.x & 31u;
    u32 base = blockIdx.x * 16u + warp * 8u;
    u32 ng = k >> 5u;
    __shared__ alignas(16) u32 sb[2][144];
    for (u32 off = 0u; off < 8u; off++) {
        u32 row = base + off;
        if (row >= n) return;
        float s = gemv_int4_row(x, w + u64(row) * ng * 18u, ng, lane, sb[warp]);
        if (lane == 0u) y[row] = s;
    }
}

extern "C" __global__ __launch_bounds__(64) void rocm_gemv_mxfp8(
    const float* x, const u8* w, float* y, u32 n, u32 k) {
    u32 tid = threadIdx.x;
    u32 warp = tid >> 5u;
    u32 lane = tid & 31u;
    u32 base = blockIdx.x * 16u + warp * 8u;
    u32 ng = k >> 5u;
    __shared__ alignas(16) u32 sb[2][264];
    __shared__ float lut[256];
#pragma unroll
    for (u32 i = 0u; i < 4u; i++) {
        lut[i * 64u + tid] = prim::from_e4m3(static_cast<u8>(i * 64u + tid));
    }
    prim::block_sync();
    for (u32 off = 0u; off < 8u; off++) {
        u32 row = base + off;
        if (row >= n) return;
        float s = gemv_mxfp8_row(x, w + u64(row) * ng * 33u, ng, lane, sb[warp], lut);
        if (lane == 0u) y[row] = s;
    }
}

#define ROCM_MM_BF16_DEF(NAME, BM, BN, NT, SPLIT, MINW)                                   \
    extern "C" __global__ __launch_bounds__(NT)                                           \
        __attribute__((amdgpu_waves_per_eu(MINW, 16))) void NAME(                         \
            const float* x, const u16* w, const float* other, const u8* bias_p, float* y, \
            u32 m, u32 n, u32 k, u32 mode, u32 ks) {                                      \
        rocm_mm_bf16_run<BM, BN, NT, SPLIT>(x, w, other, bias_p, y, m, n, k, mode, ks);   \
    }

ROCM_MM_BF16_DEF(rocm_mm_bf16_r16, 16u, 64u, 128u, false, 4)
ROCM_MM_BF16_DEF(rocm_mm_bf16_r16_sk, 16u, 64u, 128u, true, 4)
ROCM_MM_BF16_DEF(rocm_mm_bf16_r64, 64u, 64u, 128u, false, 12)
ROCM_MM_BF16_DEF(rocm_mm_bf16_r64_sk, 64u, 64u, 128u, true, 4)
ROCM_MM_BF16_DEF(rocm_mm_bf16_r128, 128u, 128u, 256u, false, 12)
ROCM_MM_BF16_DEF(rocm_mm_bf16_r128_sk, 128u, 128u, 256u, true, 12)
#undef ROCM_MM_BF16_DEF

#define ROCM_MM_BLK_DEF(NAME, MX8, BM, BN, NT, SPLIT)                                     \
    extern "C" __global__ __launch_bounds__(NT) void NAME(                                \
        const float* x, const u8* w, const float* other, const u8* bias_p, float* y,      \
        u32 m, u32 n, u32 k, u32 mode, u32 ks) {                                          \
        rocm_mm_blk_run<MX8, BM, BN, NT, SPLIT>(x, w, other, bias_p, y, m, n, k, mode,    \
                                                ks);                                      \
    }

ROCM_MM_BLK_DEF(rocm_mm_int4_r16, false, 16u, 64u, 128u, false)
ROCM_MM_BLK_DEF(rocm_mm_int4_r16_sk, false, 16u, 64u, 128u, true)
ROCM_MM_BLK_DEF(rocm_mm_int4_r64, false, 64u, 64u, 128u, false)
ROCM_MM_BLK_DEF(rocm_mm_int4_r64_sk, false, 64u, 64u, 128u, true)
ROCM_MM_BLK_DEF(rocm_mm_int4_r128, false, 128u, 128u, 256u, false)
ROCM_MM_BLK_DEF(rocm_mm_int4_r128_sk, false, 128u, 128u, 256u, true)
ROCM_MM_BLK_DEF(rocm_mm_mxfp8_r16, true, 16u, 64u, 128u, false)
ROCM_MM_BLK_DEF(rocm_mm_mxfp8_r16_sk, true, 16u, 64u, 128u, true)
ROCM_MM_BLK_DEF(rocm_mm_mxfp8_r64, true, 64u, 64u, 128u, false)
ROCM_MM_BLK_DEF(rocm_mm_mxfp8_r64_sk, true, 64u, 64u, 128u, true)
ROCM_MM_BLK_DEF(rocm_mm_mxfp8_r128, true, 128u, 64u, 256u, false)
ROCM_MM_BLK_DEF(rocm_mm_mxfp8_r128_sk, true, 128u, 64u, 256u, true)
#undef ROCM_MM_BLK_DEF

extern "C" __global__ __launch_bounds__(256) void rocm_mm_reduce(
    const float* part, const float* other, const u8* bias_p, float* y, u32 m, u32 n, u32 ks,
    u32 mode) {
    u64 idx = u64(blockIdx.x) * 256u + threadIdx.x;
    u64 total = u64(m) * n;
    if (idx >= total) return;
    float sum = 0.0f;
    for (u32 s = 0u; s < ks; s++) sum += part[u64(s) * total + idx];
    u32 col = static_cast<u32>(idx % n);
    y[idx] = apply_mm_act(sum, mode, other, bias_p, idx, col);
}

#endif
