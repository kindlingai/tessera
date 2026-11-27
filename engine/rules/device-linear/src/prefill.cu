#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

#if defined(TESSERA_CUDA)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define DTR_MX8 1
constexpr u32 DTR_TILE = trellis::MX8_TILE;
#ifndef DTR_T8_BLK
#define DTR_T8_BLK 3
#endif
#else
#define DTR_MX8 0
constexpr u32 DTR_TILE = 512u;
#ifndef DTR_T8_BLK
#define DTR_T8_BLK 2
#endif
#endif

#ifndef DTR_PREP_BLK
#define DTR_PREP_BLK 3
#endif
#ifndef DTR_T16_BLK
#define DTR_T16_BLK 2
#endif
#ifndef DTR_GEN8_BLK
#define DTR_GEN8_BLK 2
#endif
#ifndef DTR_GEN16_BLK
#define DTR_GEN16_BLK 2
#endif
#ifndef MM_PREP_BLK
#define MM_PREP_BLK 4
#endif
#ifndef MM_MX8_BLK
#define MM_MX8_BLK 1
#endif
#ifndef MM_MX8_SK_BLK
#define MM_MX8_SK_BLK 1
#endif

constexpr u32 DTR_STRIDE = 8u * DTR_TILE;

template <u32 NT>
union alignas(16) dtr_pre_smem {
    struct {
        u32 btab[NT * DTR_TILE];
        u32 wtiles[2048];
    } k;
    float out[32][128];
};

static __device__ __forceinline__ void dtr_store_frag(u32* tile, u32 row, u32 lane, bool valid,
                                                      const float v[4]) {
#if DTR_MX8
    (void)valid;
    trellis::store_bfrag_mx8(tile, row, lane, v);
#else
    if (valid) trellis::store_bfrag(tile, row, lane, v);
    else {
        float z[4] = {};
        trellis::store_bfrag(tile, row, lane, z);
    }
#endif
}

template <u32 K, u32 BOOK, u32 NT>
static __device__ __forceinline__ void dtr_run_mma(const u32* wtiles, const u32* btab, u32 j,
                                                   u32 lane, const trellis::word_map& wm,
                                                   float acc[NT][4]) {
#if DTR_MX8
    trellis::blk_mma_mx8<K, BOOK, NT>(wtiles, btab, j, lane, wm, acc);
#else
    trellis::blk_mma<K, BOOK, NT>(wtiles, btab + (lane << 1u), 512u, j, lane, wm, acc);
#endif
}

// Rotates and quantizes dense inputs once per 64x128 block.
extern "C" __global__ __launch_bounds__(256, DTR_PREP_BLK) void dtr_prep(const float* x, const float* suh,
                                                              float* out, u32 m, u32 k,
                                                              u32 turned) {
    u32 c0 = blockIdx.x, rt = blockIdx.y, r0 = rt << 6u;
    if (r0 >= m) return;
    u32 nr = min(64u, m - r0), tid = threadIdx.x, lane = tid & 31u, j = tid >> 5u;
    __shared__ u32 btab[DTR_STRIDE];
    u32 col = (c0 << 7u) + (lane << 2u);
    float sc[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (turned != 0u) trellis::load4(suh + col, sc);
#pragma unroll
    for (u32 ti = 0u; ti < 8u; ti++) {
        u32 r = (ti << 3u) + j;
        float v[4] = {};
        if (r < nr) {
            trellis::load4(x + u64(r0 + r) * k + col, v);
            if (turned != 0u) {
                v[0] *= sc[0]; v[1] *= sc[1]; v[2] *= sc[2]; v[3] *= sc[3];
                trellis::wht128(v, trellis::INV_SQRT128);
            }
        }
        dtr_store_frag(btab + ti * DTR_TILE, j, lane, r < nr, v);
    }
    prim::block_sync();
    u32 nblk = k >> 7u;
    uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<u32*>(out) +
                                          (u64(rt) * nblk + c0) * DTR_STRIDE);
    const uint4* src = reinterpret_cast<const uint4*>(btab);
#pragma unroll
    for (u32 i = tid; i < (DTR_STRIDE >> 2u); i += 256u) dst[i] = src[i];
}

template <u32 K, u32 BOOK, u32 NT, bool TURNED>
static __device__ __forceinline__ void dtr_pre_body(const float* xpre_f, const u8* w,
                                                    const float* svh, float* out, u32 m, u32 n,
                                                    u32 k, dtr_pre_smem<NT>& sm) {
    constexpr u32 TB = 32u * K, SUBS = NT / 8u;
    const u32* xpre = reinterpret_cast<const u32*>(xpre_f);
    u32 nblk = n >> 7u, ob = blockIdx.x, rblk = blockIdx.y, r0 = rblk * (8u * NT);
    if (ob >= nblk || r0 >= m) return;
    u32 nr = min(8u * NT, m - r0), tid = threadIdx.x, lane = tid & 31u, j = tid >> 5u;
    u32 blk_per = k >> 7u;
    bool aligned = (reinterpret_cast<u64>(w + trellis::HEAD) & 15ull) == 0ull;
    bool input_side = (*reinterpret_cast<const u32*>(w + 8u) & trellis::SIDE_MASK) == trellis::INPUT_SIDE;
    u64 sG = (input_side ? u64(n >> 1u) : 64ull) * u64(TB);
    const u8* gb = trellis::tile_ptr<K>(w, k, n, 0u, 0u, ob, input_side);
    u32 nw[8];
    trellis::blk_load<K>(gb, tid, aligned, nw);
    trellis::word_map wm = trellis::map_word<K>(lane);
    float acc[NT][4];
#pragma unroll
    for (u32 ti = 0u; ti < NT; ti++) acc[ti][0] = acc[ti][1] = acc[ti][2] = acc[ti][3] = 0.0f;
    for (u32 c0 = 0u; c0 < blk_per; c0++) {
        if (c0 > 0u) prim::block_sync();
#pragma unroll
        for (u32 s = 0u; s < SUBS; s++) {
            u32 rt = rblk * SUBS + s;
            if ((rt << 6u) < m) {
                const uint4* src =
                    reinterpret_cast<const uint4*>(xpre + (u64(rt) * blk_per + c0) * DTR_STRIDE);
                uint4* dst = reinterpret_cast<uint4*>(sm.k.btab + s * DTR_STRIDE);
#pragma unroll
                for (u32 i = tid; i < (DTR_STRIDE >> 2u); i += 256u) dst[i] = src[i];
            }
        }
        trellis::blk_store<K>(nw, tid, sm.k.wtiles);
        prim::block_sync();
        if (c0 + 1u < blk_per) {
            gb += sG;
            trellis::blk_load<K>(gb, tid, aligned, nw);
        }
        dtr_run_mma<K, BOOK, NT>(sm.k.wtiles, sm.k.btab, j, lane, wm, acc);
    }
    u32 col = (j << 4u) + (lane >> 2u), rc = (lane & 3u) << 1u;
    if constexpr (!TURNED) {
        u32 ob0 = ob << 7u;
#pragma unroll
        for (u32 ti = 0u; ti < NT; ti++) {
            u32 r = r0 + (ti << 3u) + rc;
            if (r < m) { out[u64(r) * n + ob0 + col] = acc[ti][0]; out[u64(r) * n + ob0 + col + 8u] = acc[ti][2]; }
            if (r + 1u < m) { out[u64(r + 1u) * n + ob0 + col] = acc[ti][1]; out[u64(r + 1u) * n + ob0 + col + 8u] = acc[ti][3]; }
        }
        return;
    }
    u32 base = (ob << 7u) + (lane << 2u);
    float sc[4];
    trellis::load4(svh + base, sc);
#pragma unroll
    for (u32 h = 0u; h < NT; h += 4u) {
        prim::block_sync();
#pragma unroll
        for (u32 ti = 0u; ti < 4u; ti++) {
            u32 r = ((h + ti) << 3u) + rc, lr = (ti << 3u) + rc;
            if (r < nr) { sm.out[lr][col] = acc[h + ti][0]; sm.out[lr][col + 8u] = acc[h + ti][2]; }
            if (r + 1u < nr) { sm.out[lr + 1u][col] = acc[h + ti][1]; sm.out[lr + 1u][col + 8u] = acc[h + ti][3]; }
        }
        prim::block_sync();
#pragma unroll
        for (u32 ti = 0u; ti < 4u; ti++) {
            u32 r = ((h + ti) << 3u) + j;
            if (r < nr) {
                float v[4];
                trellis::load4(&sm.out[(ti << 3u) + j][lane << 2u], v);
                trellis::wht128(v, trellis::INV_SQRT128);
                *reinterpret_cast<float4*>(out + u64(r0 + r) * n + base) =
                    make_float4(v[0] * sc[0], v[1] * sc[1], v[2] * sc[2], v[3] * sc[3]);
            }
        }
    }
}

template <u32 K, u32 BOOK, u32 NT>
static __device__ __noinline__ void dtr_pre_k(const float* xpre, const u8* w, const float* svh,
                                              float* out, u32 m, u32 n, u32 k, u32 turned,
                                              dtr_pre_smem<NT>& sm) {
    if (turned != 0u) dtr_pre_body<K, BOOK, NT, true>(xpre, w, svh, out, m, n, k, sm);
    else dtr_pre_body<K, BOOK, NT, false>(xpre, w, svh, out, m, n, k, sm);
}

template <u32 BOOK, u32 NT>
static __device__ void dtr_pre_book(const float* xpre, const u8* w, const float* svh, float* out,
                                    u32 m, u32 n, u32 k, u32 bits, u32 turned,
                                    dtr_pre_smem<NT>& sm) {
    if (bits == 2u) dtr_pre_k<2u, BOOK, NT>(xpre, w, svh, out, m, n, k, turned, sm);
    else if (bits == 3u) dtr_pre_k<3u, BOOK, NT>(xpre, w, svh, out, m, n, k, turned, sm);
    else dtr_pre_k<4u, BOOK, NT>(xpre, w, svh, out, m, n, k, turned, sm);
}

template <u32 NT>
static __device__ void dtr_pre_gen(const float* xpre, const u8* w, const float* svh, float* out,
                                   u32 m, u32 n, u32 k, u32 bits, u32 book, u32 turned,
                                   dtr_pre_smem<NT>& sm) {
    if (book == trellis::BOOK_PMCGX) dtr_pre_book<trellis::BOOK_PMCGX, NT>(xpre, w, svh, out, m, n, k, bits, turned, sm);
    else if (book == trellis::BOOK_MUL1) dtr_pre_book<trellis::BOOK_MUL1, NT>(xpre, w, svh, out, m, n, k, bits, turned, sm);
    else dtr_pre_book<trellis::BOOK_MCG, NT>(xpre, w, svh, out, m, n, k, bits, turned, sm);
}

// Rotated 4-bit dense trellis prefill, 64 rows per block.
extern "C" __global__ __launch_bounds__(256, DTR_T8_BLK) void dtr_pre_k4_t8(
    const float* xpre, const u8* w, const float* svh, float* out, u32 m, u32 n, u32 k, u32 bits,
    u32 book, u32 turned) {
    (void)bits; (void)book; (void)turned;
    __shared__ dtr_pre_smem<8u> sm;
    dtr_pre_body<4u, trellis::BOOK_MCG, 8u, true>(xpre, w, svh, out, m, n, k, sm);
}

// Rotated 4-bit dense trellis prefill, 128 rows per block.
extern "C" __global__ __launch_bounds__(256, DTR_T16_BLK) void dtr_pre_k4_t16(
    const float* xpre, const u8* w, const float* svh, float* out, u32 m, u32 n, u32 k, u32 bits,
    u32 book, u32 turned) {
    (void)bits; (void)book; (void)turned;
    __shared__ dtr_pre_smem<16u> sm;
    dtr_pre_body<4u, trellis::BOOK_MCG, 16u, true>(xpre, w, svh, out, m, n, k, sm);
}

// Generic dense trellis prefill, 64 rows per block.
extern "C" __global__ __launch_bounds__(256, DTR_GEN8_BLK) void dtr_pre_gen8(
    const float* xpre, const u8* w, const float* svh, float* out, u32 m, u32 n, u32 k, u32 bits,
    u32 book, u32 turned) {
    __shared__ dtr_pre_smem<8u> sm;
    dtr_pre_gen<8u>(xpre, w, svh, out, m, n, k, bits, book, turned, sm);
}

// Generic dense trellis prefill, 128 rows per block.
extern "C" __global__ __launch_bounds__(256, DTR_GEN16_BLK) void dtr_pre_gen16(
    const float* xpre, const u8* w, const float* svh, float* out, u32 m, u32 n, u32 k, u32 bits,
    u32 book, u32 turned) {
    __shared__ dtr_pre_smem<16u> sm;
    dtr_pre_gen<16u>(xpre, w, svh, out, m, n, k, bits, book, turned, sm);
}

static __device__ __forceinline__ float mx8_silu(float v) {
    return v / (1.0f + expf(-v));
}

static __device__ __forceinline__ float mx8_epilogue(float v, const float* other, u64 off, u32 col,
                                                     u32 mode) {
    if (mode == 1u) return v + other[off];
    if (mode == 2u) return v + other[col];
    if (mode == 3u) return mx8_silu(v) * other[off];
    if (mode == 4u) return v * other[off];
    if (mode == 5u) return mx8_silu(v);
    return v;
}

// Quantizes X[m, k] once to packed E4M3 and UE8M0 scales.
extern "C" __global__ __launch_bounds__(256, MM_PREP_BLK) void tessera_mm_mxfp8_prep(
    const float* x, float* out, u32 m, u32 k) {
#if DTR_MX8
    u32 blk = blockIdx.x * 32u + (threadIdx.x >> 3u);
    u32 row = blockIdx.y;
    u32 nblk = k >> 5u;
    if (row >= m || blk >= nblk) return;
    u32 sub = threadIdx.x & 7u;
    float4 v = *reinterpret_cast<const float4*>(x + u64(row) * k + (blk << 5u) + (sub << 2u));
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 4u));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2u));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1u));
    float inv;
    u32 e = trellis::mx_scale(amax, inv);
    u16 p0 = trellis::f2_e4m3x2(v.x * inv, v.y * inv);
    u16 p1 = trellis::f2_e4m3x2(v.z * inv, v.w * inv);
    u8* xq = reinterpret_cast<u8*>(out);
    u8* xs = xq + u64(m) * k;
    *reinterpret_cast<u32*>(xq + u64(row) * k + (blk << 5u) + (sub << 2u)) =
        u32(p0) | (u32(p1) << 16u);
    if (sub == 0u) xs[u64(row) * nblk + blk] = u8(e);
#else
    (void)x; (void)out; (void)m; (void)k;
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
static __device__ __forceinline__ void mx8_cp16(void* smem, const void* gmem, bool pred) {
    u32 dst = static_cast<u32>(__cvta_generic_to_shared(smem));
    if (pred) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem));
    else *reinterpret_cast<uint4*>(smem) = make_uint4(0u, 0u, 0u, 0u);
}

static __device__ __forceinline__ void mx8_ldm4(u32 (&r)[4], const void* smem_ptr) {
    u32 addr = static_cast<u32>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

template <bool SPLIT>
static __device__ __forceinline__ void mx8_store_vec4(
    float4 v, const float* other, float* out, u64 off, u32 col, u32 n, u32 mode, bool vec4) {
    if (vec4 && col + 3u < n) {
        if (!SPLIT && mode != 0u) {
            v.x = mx8_epilogue(v.x, other, off, col, mode);
            v.y = mx8_epilogue(v.y, other, off + 1u, col + 1u, mode);
            v.z = mx8_epilogue(v.z, other, off + 2u, col + 2u, mode);
            v.w = mx8_epilogue(v.w, other, off + 3u, col + 3u, mode);
        }
        *reinterpret_cast<float4*>(out + off) = v;
    } else {
        float vals[4] = {v.x, v.y, v.z, v.w};
        for (u32 j = 0u; j < 4u && col + j < n; j++) {
            out[off + j] = SPLIT ? vals[j] : mx8_epilogue(vals[j], other, off + j, col + j, mode);
        }
    }
}

template <u32 MI, bool SPLIT>
static __device__ __forceinline__ void mx8_epilogue_smem(
    float (*out_sm)[132], const float (&acc)[2][MI][4], const float* other, float* out,
    u32 m, u32 n, u32 r0, u32 n0, u32 p, u32 mode, u32 tid, u32 wm, u32 n_base, u32 g, u32 t) {
    constexpr u32 Q_ROWS = MI << 3u;
    constexpr u32 Q_SHIFT = (MI == 8u) ? 6u : 5u;
    u32 sr = tid >> 5u, sc4 = (tid & 31u) << 2u;
    bool vec4 = (n & 3u) == 0u;
#pragma unroll
    for (u32 q = 0u; q < 2u; q++) {
        if (wm == q) {
#pragma unroll
            for (u32 ni = 0u; ni < 2u; ni++) {
                u32 lc0 = n_base + (ni << 4u) + g, lc1 = lc0 + 8u;
#pragma unroll
                for (u32 mi = 0u; mi < MI; mi++) {
                    u32 lr = (mi << 3u) + (t << 1u);
                    out_sm[lr][lc0] = acc[ni][mi][0];
                    out_sm[lr][lc1] = acc[ni][mi][2];
                    out_sm[lr + 1u][lc0] = acc[ni][mi][1];
                    out_sm[lr + 1u][lc1] = acc[ni][mi][3];
                }
            }
        }
        prim::block_sync();
        u32 col = n0 + sc4;
#pragma unroll
        for (u32 dr = 0u; dr < Q_ROWS; dr += 8u) {
            u32 row = r0 + (q << Q_SHIFT) + sr + dr;
            if (row < m && col < n) {
                float4 v = *reinterpret_cast<const float4*>(&out_sm[sr + dr][sc4]);
                u64 off = u64(SPLIT ? (p * m + row) : row) * n + col;
                mx8_store_vec4<SPLIT>(v, other, out, off, col, n, mode, vec4);
            }
        }
        if (q == 0u) prim::block_sync();
    }
}
#endif

#if DTR_MX8
union alignas(16) mx8_128_smem {
    struct {
        u32 sw[2][128][36];
        u32 sx[2][128][32];
        u32 sxs[2][128];
    } k;
    float out[64][132];
};

static __device__ __forceinline__ void mx8_cp4(void* smem, const void* gmem, bool pred) {
    u32 dst = static_cast<u32>(__cvta_generic_to_shared(smem));
    if (pred) asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dst), "l"(gmem));
    else *reinterpret_cast<u32*>(smem) = 0u;
}

template <u32 SH>
static __device__ __forceinline__ void mx8_load_fa(
    const u32* sw0, const u32* sw1, u32 w_base, u32 t, u32 (&fa)[4]) {
    u32 w0 = w_base + t;
    if constexpr (SH == 0u) {
        fa[0] = sw0[w0];
        fa[1] = sw1[w0];
        fa[2] = sw0[w0 + 4u];
        fa[3] = sw1[w0 + 4u];
    } else {
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(fa[0]) : "r"(sw0[w0]), "r"(sw0[w0 + 1u]), "n"(SH));
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(fa[1]) : "r"(sw1[w0]), "r"(sw1[w0 + 1u]), "n"(SH));
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(fa[2]) : "r"(sw0[w0 + 4u]), "r"(sw0[w0 + 5u]), "n"(SH));
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(fa[3]) : "r"(sw1[w0 + 4u]), "r"(sw1[w0 + 5u]), "n"(SH));
    }
}

static __device__ __forceinline__ void mx8_load_stage(
    mx8_128_smem& sm, u32 st, u32 ch128, const u8* xq, const u8* xs, const u8* w,
    u32 m, u32 n, u32 k, u32 nblk, u64 row_bytes, u32 n0, u32 r0, u32 tid, u32 r0_t, u32 col16) {
    u64 aligned_off = (u64(ch128) * 132ull) & ~15ull;
#pragma unroll
    for (u32 dr = 0u; dr < 128u; dr += 32u) {
        u32 nr = r0_t + dr;
        mx8_cp16(&sm.k.sw[st][nr][col16 << 2u],
                 w + u64(n0 + nr) * row_bytes + aligned_off + (col16 << 4u), n0 + nr < n);
    }
    if (tid < 128u) {
        mx8_cp16(&sm.k.sw[st][tid][32u], w + u64(n0 + tid) * row_bytes + aligned_off + 128u, n0 + tid < n);
    }
    u32 k_off = ch128 << 7u;
#pragma unroll
    for (u32 dr = 0u; dr < 128u; dr += 32u) {
        u32 mr = r0_t + dr;
        mx8_cp16(&sm.k.sx[st][mr][(col16 ^ (mr & 7u)) << 2u],
                 xq + u64(r0 + mr) * k + k_off + (col16 << 4u), r0 + mr < m);
    }
    if (tid < 128u) {
        mx8_cp4(&sm.k.sxs[st][tid], xs + u64(r0 + tid) * nblk + (ch128 << 2u), r0 + tid < m);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

template <u32 PH, u32 BLK>
static __device__ __forceinline__ void mx8_run_blk(
    const u32 (*sw_st)[36], const u32 (*sx_st)[32], const u32 sb4[8],
    u32 n_base, u32 m_base, u32 g, u32 t, u32 lb_row, u32 lb_half, float acc[2][8][4]) {
    constexpr u32 B_OFF = PH * 4u + BLK * 33u;
    constexpr u32 Q_BASE = B_OFF + 1u;
    constexpr u32 SH = (Q_BASE & 3u) << 3u;
    constexpr u32 W_BASE = Q_BASE >> 2u;
    constexpr u32 S_SH = BLK << 3u;
    u32 fb[8][2], sb[8];
#pragma unroll
    for (u32 q = 0u; q < 4u; q++) {
        u32 mr = m_base + (q << 4u) + lb_row, tmp[4];
        mx8_ldm4(tmp, &sx_st[mr][(((BLK << 1u) | lb_half) ^ (mr & 7u)) << 2u]);
        fb[q << 1u][0] = tmp[0]; fb[q << 1u][1] = tmp[1];
        fb[(q << 1u) + 1u][0] = tmp[2]; fb[(q << 1u) + 1u][1] = tmp[3];
    }
#pragma unroll
    for (u32 mi = 0u; mi < 8u; mi++) sb[mi] = (sb4[mi] >> S_SH) & 0xFFu;

    u32 fa0[4], fa1[4], sa0, sa1;
    {
        u32 nr0 = n_base + g, nr1 = nr0 + 8u;
        const u32 *sw0 = sw_st[nr0], *sw1 = sw_st[nr1];
        sa0 = u32(reinterpret_cast<const u8*>((t & 1u) ? sw1 : sw0)[B_OFF]);
        mx8_load_fa<SH>(sw0, sw1, W_BASE, t, fa0);
    }
    {
        u32 nr0 = n_base + 16u + g, nr1 = nr0 + 8u;
        const u32 *sw0 = sw_st[nr0], *sw1 = sw_st[nr1];
        sa1 = u32(reinterpret_cast<const u8*>((t & 1u) ? sw1 : sw0)[B_OFF]);
        mx8_load_fa<SH>(sw0, sw1, W_BASE, t, fa1);
    }
#pragma unroll
    for (u32 mi = 0u; mi < 8u; mi++) trellis::mma_mx8(acc[0][mi], fa0, fb[mi], sa0, sb[mi]);
#pragma unroll
    for (u32 mi = 0u; mi < 8u; mi++) trellis::mma_mx8(acc[1][mi], fa1, fb[mi], sa1, sb[mi]);
}

template <u32 PH>
static __device__ __forceinline__ void mx8_phase_step(
    mx8_128_smem& sm, u32 st, u32 nxt_ch, bool has_nxt,
    const u8* xq, const u8* xs, const u8* w, u32 m, u32 n, u32 k, u32 nblk, u64 row_bytes,
    u32 n0, u32 r0, u32 tid, u32 r0_t, u32 col16,
    u32 n_base, u32 m_base, u32 g, u32 t, u32 lb_row, u32 lb_half, float acc[2][8][4]) {
    if (has_nxt) {
        mx8_load_stage(sm, st ^ 1u, nxt_ch, xq, xs, w, m, n, k, nblk, row_bytes, n0, r0, tid, r0_t, col16);
        asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
        asm volatile("cp.async.wait_group 0;\n" ::);
    }
    prim::block_sync();
    u32 sb4[8];
#pragma unroll
    for (u32 mi = 0u; mi < 8u; mi++) sb4[mi] = sm.k.sxs[st][m_base + (mi << 3u) + g];
    mx8_run_blk<PH, 0u>(sm.k.sw[st], sm.k.sx[st], sb4, n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 1u>(sm.k.sw[st], sm.k.sx[st], sb4, n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 2u>(sm.k.sw[st], sm.k.sx[st], sb4, n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 3u>(sm.k.sw[st], sm.k.sx[st], sb4, n_base, m_base, g, t, lb_row, lb_half, acc);
    if (has_nxt) prim::block_sync();
}

template <bool SPLIT>
static __device__ __forceinline__ void fg_mxfp8_mx8(
    const float* xpre, const u8* w, const float* other, float* out,
    u32 m, u32 n, u32 k, u32 mode, u32 ks) {
    extern __shared__ u8 raw_smem[];
    mx8_128_smem& sm = *reinterpret_cast<mx8_128_smem*>(raw_smem);
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
    u32 g = lane >> 2u, t = lane & 3u, wn = warp & 3u, wm = warp >> 2u;
    u32 nb = (n + 127u) >> 7u, mb = (m + 127u) >> 7u;
    u32 ob = SPLIT ? (blockIdx.x % nb) : blockIdx.x;
    u32 p = SPLIT ? (blockIdx.x / nb) : 0u;
    u32 lin = blockIdx.y * nb + ob, grp_sz = 8u * mb;
    u32 gid = lin / grp_sz, rem = lin - gid * grp_sz;
    u32 first_n = gid * 8u, width = min(8u, nb - first_n);
    u32 by = rem / width, bx = first_n + (rem - by * width);
    u32 n0 = bx << 7u, r0 = by << 7u;

    u32 chunks = k >> 9u, ch_per = SPLIT ? (chunks / ks) : chunks;
    u32 c5_0 = SPLIT ? (p * ch_per) : 0u, c5_1 = c5_0 + ch_per;
    u32 nblk = k >> 5u;
    u64 row_bytes = u64(nblk) * 33ull;
    const u8* xq = reinterpret_cast<const u8*>(xpre);
    const u8* xs = xq + u64(m) * k;
    u32 r0_t = tid >> 3u, col16 = tid & 7u;
    u32 n_base = wn << 5u, m_base = wm << 6u;
    u32 lb_row = ((lane >> 4u) << 3u) | (lane & 7u), lb_half = (lane >> 3u) & 1u;

    float acc[2][8][4];
#pragma unroll
    for (u32 ni = 0u; ni < 2u; ni++)
#pragma unroll
        for (u32 mi = 0u; mi < 8u; mi++)
            acc[ni][mi][0] = acc[ni][mi][1] = acc[ni][mi][2] = acc[ni][mi][3] = 0.0f;

    mx8_load_stage(sm, 0u, c5_0 << 2u, xq, xs, w, m, n, k, nblk, row_bytes, n0, r0, tid, r0_t, col16);
    for (u32 c5 = c5_0; c5 < c5_1; c5++) {
        u32 base = c5 << 2u;
        mx8_phase_step<0u>(sm, 0u, base + 1u, true, xq, xs, w, m, n, k, nblk, row_bytes,
                           n0, r0, tid, r0_t, col16, n_base, m_base, g, t, lb_row, lb_half, acc);
        mx8_phase_step<1u>(sm, 1u, base + 2u, true, xq, xs, w, m, n, k, nblk, row_bytes,
                           n0, r0, tid, r0_t, col16, n_base, m_base, g, t, lb_row, lb_half, acc);
        mx8_phase_step<2u>(sm, 0u, base + 3u, true, xq, xs, w, m, n, k, nblk, row_bytes,
                           n0, r0, tid, r0_t, col16, n_base, m_base, g, t, lb_row, lb_half, acc);
        mx8_phase_step<3u>(sm, 1u, base + 4u, c5 + 1u < c5_1, xq, xs, w, m, n, k, nblk, row_bytes,
                           n0, r0, tid, r0_t, col16, n_base, m_base, g, t, lb_row, lb_half, acc);
    }
    prim::block_sync();
    mx8_epilogue_smem<8u, SPLIT>(sm.out, acc, other, out, m, n, r0, n0, p, mode, tid, wm, n_base, g, t);
}
#endif

// Blackwell native MXFP8 MMA matmul, 128x128 per block.
extern "C" __global__ __launch_bounds__(256, MM_MX8_BLK) void tessera_mm_mxfp8_mx8(
    const float* xpre, const u8* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode) {
#if DTR_MX8
    fg_mxfp8_mx8<false>(xpre, w, other, out, m, n, k, mode, 1u);
#else
    (void)xpre; (void)w; (void)other; (void)out; (void)m; (void)n; (void)k; (void)mode;
#endif
}

// Split-K Blackwell native MXFP8 MMA matmul, 128x128 per block.
extern "C" __global__ __launch_bounds__(256, MM_MX8_SK_BLK) void tessera_mm_mxfp8_mx8_sk(
    const float* xpre, const u8* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode,
    u32 ks) {
#if DTR_MX8
    fg_mxfp8_mx8<true>(xpre, w, other, out, m, n, k, mode, ks);
#else
    (void)xpre; (void)w; (void)other; (void)out; (void)m; (void)n; (void)k; (void)mode; (void)ks;
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
static __device__ __forceinline__ u32 i4_dec(u32 b) {
    u32 p = b * 0x1001u, v, out;
    asm("lop3.b32 %0, %1, 0x000F000F, 0x64006400, 0xEA;" : "=r"(v) : "r"(p));
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(out) : "r"(v), "r"(0x64086408u));
    return out;
}

static __device__ __forceinline__ u32 i4_dec_s(u32 b, u32 sc_h) {
    u32 v = i4_dec(b), out;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(out) : "r"(v), "r"(sc_h));
    return out;
}

union alignas(16) i4_r16_smem {
    u32 sw[2][16][36];
    float red[4][16][17];
};

static __device__ __forceinline__ void i4_hilo_pair(const float* p, u32& hi, u32& lo) {
    float2 v = *reinterpret_cast<const float2*>(p);
    hi = trellis::f32x2_to_f16x2(v.x, v.y);
    lo = trellis::f32x2_to_f16x2(
        v.x - trellis::f16_to_f32(hi & 0xFFFFu),
        v.y - trellis::f16_to_f32(hi >> 16u));
}

template <bool SPLIT>
static __device__ __forceinline__ void fg_int4_r16(
    const float* x, const u8* w, const float* other, float* out,
    u32 m, u32 n, u32 k, u32 mode, u32 ks, i4_r16_smem& sm) {
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u, g = lane >> 2u, t = lane & 3u;
    u32 r0 = blockIdx.y << 4u, nb = n >> 4u;
    u32 ob = SPLIT ? (blockIdx.x % nb) : blockIdx.x, p = SPLIT ? (blockIdx.x / nb) : 0u;
    u32 n0 = ob << 4u, chunks = k >> 8u, ch_per = SPLIT ? (chunks / ks) : chunks;
    u32 ch0 = SPLIT ? (p * ch_per) : 0u, ch1 = ch0 + ch_per;
    u64 row_bytes = u64(k >> 5u) * 18ull;
    u32 r_l = tid >> 3u, c_l = tid & 7u;
    auto load_w = [&](u32 st, u32 ch) {
        u64 off = u64(ch) * 144ull;
        mx8_cp16(&sm.sw[st][r_l][c_l << 2u], w + u64(n0 + r_l) * row_bytes + off + (c_l << 4u), true);
        if (tid < 16u) mx8_cp16(&sm.sw[st][tid][32u], w + u64(n0 + tid) * row_bytes + off + 128u, true);
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    load_w(0u, ch0);
    for (u32 ch = ch0; ch < ch1; ch++) {
        u32 st = (ch - ch0) & 1u;
        bool has_nxt = (ch + 1u < ch1);
        if (has_nxt) {
            load_w(st ^ 1u, ch + 1u);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        prim::block_sync();
        const u8* sb0 = reinterpret_cast<const u8*>(sm.sw[st][g]);
        const u8* sb1 = reinterpret_cast<const u8*>(sm.sw[st][g + 8u]);
#pragma unroll
        for (u32 bl = 0u; bl < 2u; bl++) {
            u32 blk = (warp << 1u) + bl, b_off = blk * 18u;
            float sc0 = trellis::f16_to_f32(*reinterpret_cast<const u16*>(sb0 + b_off));
            float sc1 = trellis::f16_to_f32(*reinterpret_cast<const u16*>(sb1 + b_off));
            float tmp[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
            for (u32 ks_i = 0u; ks_i < 2u; ks_i++) {
                u32 q_off = b_off + 2u + (ks_i << 3u) + t;
                u32 a[4] = {i4_dec(sb0[q_off]), i4_dec(sb1[q_off]),
                            i4_dec(sb0[q_off + 4u]), i4_dec(sb1[q_off + 4u])};
                u32 k_col = (ch << 8u) + (blk << 5u) + (ks_i << 4u) + (t << 1u);
#pragma unroll
                for (u32 ti = 0u; ti < 2u; ti++) {
                    u32 tok = r0 + (ti << 3u) + g, bh[2] = {0u, 0u}, blv[2] = {0u, 0u};
                    if (tok < m) {
                        const float* xp = x + u64(tok) * k + k_col;
                        i4_hilo_pair(xp, bh[0], blv[0]);
                        i4_hilo_pair(xp + 8u, bh[1], blv[1]);
                    }
                    trellis::mma_m16n8k16(tmp[ti], a, bh);
                    trellis::mma_m16n8k16(tmp[ti], a, blv);
                }
            }
#pragma unroll
            for (u32 ti = 0u; ti < 2u; ti++) {
                acc[ti][0] = fmaf(sc0, tmp[ti][0], acc[ti][0]);
                acc[ti][1] = fmaf(sc0, tmp[ti][1], acc[ti][1]);
                acc[ti][2] = fmaf(sc1, tmp[ti][2], acc[ti][2]);
                acc[ti][3] = fmaf(sc1, tmp[ti][3], acc[ti][3]);
            }
        }
        if (has_nxt) prim::block_sync();
    }
    prim::block_sync();
#pragma unroll
    for (u32 ti = 0u; ti < 2u; ti++) {
        u32 tr = (ti << 3u) + (t << 1u);
        sm.red[warp][tr][g] = acc[ti][0]; sm.red[warp][tr + 1u][g] = acc[ti][1];
        sm.red[warp][tr][g + 8u] = acc[ti][2]; sm.red[warp][tr + 1u][g + 8u] = acc[ti][3];
    }
    prim::block_sync();
    u32 col = tid & 15u, tr0 = tid >> 4u;
    for (u32 tr = tr0; tr < 16u; tr += 8u) {
        if (r0 + tr >= m) continue;
        float v = sm.red[0][tr][col] + sm.red[1][tr][col] +
                  sm.red[2][tr][col] + sm.red[3][tr][col];
        u64 off = u64(SPLIT ? (p * m + r0 + tr) : (r0 + tr)) * n + n0 + col;
        out[off] = SPLIT ? v : mx8_epilogue(v, other, off, n0 + col, mode);
    }
}

union alignas(16) i4_r64_smem {
    struct {
        u32 sw[128][36];
        u32 sx[2][64][32];
    } k;
    float out[32][132];
};

static __device__ __forceinline__ void i4_load_x(
    u32 (*sx_st)[32], const float* x, u32 r0, u32 k_sub, u32 m, u32 k, u32 r0_t, u32 col16) {
#pragma unroll
    for (u32 dr = 0u; dr < 64u; dr += 32u) {
        u32 r = r0_t + dr;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (r0 + r < m) {
            const float* xp = x + u64(r0 + r) * k + k_sub + (col16 << 3u);
            float4 x0 = *reinterpret_cast<const float4*>(xp);
            float4 x1 = *reinterpret_cast<const float4*>(xp + 4u);
            v = make_uint4(trellis::f32x2_to_f16x2(x0.x, x0.y),
                           trellis::f32x2_to_f16x2(x0.z, x0.w),
                           trellis::f32x2_to_f16x2(x1.x, x1.y),
                           trellis::f32x2_to_f16x2(x1.z, x1.w));
        }
        *reinterpret_cast<uint4*>(&sx_st[r][(col16 ^ (r & 7u)) << 2u]) = v;
    }
}

template <bool SPLIT>
static __device__ __forceinline__ void fg_int4_r64(
    const float* x, const u8* w, const float* other, float* out,
    u32 m, u32 n, u32 k, u32 mode, u32 ks, i4_r64_smem& sm) {
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
    u32 g = lane >> 2u, t = lane & 3u, wn = warp & 3u, wm = warp >> 2u;
    u32 nb = (n + 127u) >> 7u, mb = (m + 63u) >> 6u;
    u32 ob = SPLIT ? (blockIdx.x % nb) : blockIdx.x, p = SPLIT ? (blockIdx.x / nb) : 0u;
    u32 lin = blockIdx.y * nb + ob, grp_sz = 8u * mb;
    u32 gid = lin / grp_sz, rem = lin - gid * grp_sz;
    u32 first_n = gid * 8u, width = min(8u, nb - first_n);
    u32 by = rem / width, bx = first_n + (rem - by * width);
    u32 n0 = bx << 7u, r0 = by << 6u;

    u32 chunks = k >> 8u, ch_per = SPLIT ? (chunks / ks) : chunks;
    u32 ch0 = SPLIT ? (p * ch_per) : 0u, ch1 = ch0 + ch_per;
    u64 row_bytes = u64(k >> 5u) * 18ull;
    u32 r0_t = tid >> 3u, col16 = tid & 7u;
    u32 n_base = wn << 5u, m_base = wm << 5u;
    u32 lb_row = ((lane >> 4u) << 3u) | (lane & 7u), lb_half = (lane >> 3u) & 1u;

    float acc[2][4][4];
#pragma unroll
    for (u32 ni = 0u; ni < 2u; ni++)
#pragma unroll
        for (u32 mi = 0u; mi < 4u; mi++)
            acc[ni][mi][0] = acc[ni][mi][1] = acc[ni][mi][2] = acc[ni][mi][3] = 0.0f;

    const u8* sb0[2] = {reinterpret_cast<const u8*>(sm.k.sw[n_base + g]),
                        reinterpret_cast<const u8*>(sm.k.sw[n_base + 16u + g])};
    const u8* sb1[2] = {reinterpret_cast<const u8*>(sm.k.sw[n_base + g + 8u]),
                        reinterpret_cast<const u8*>(sm.k.sw[n_base + 24u + g])};

    for (u32 ch = ch0; ch < ch1; ch++) {
        u64 ch_off = u64(ch) * 144ull;
#pragma unroll
        for (u32 dr = 0u; dr < 128u; dr += 32u) {
            u32 nr = r0_t + dr;
            mx8_cp16(&sm.k.sw[nr][col16 << 2u],
                     w + u64(n0 + nr) * row_bytes + ch_off + (col16 << 4u), n0 + nr < n);
        }
        if (tid < 128u) {
            mx8_cp16(&sm.k.sw[tid][32u],
                     w + u64(n0 + tid) * row_bytes + ch_off + 128u, n0 + tid < n);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        i4_load_x(sm.k.sx[0], x, r0, ch << 8u, m, k, r0_t, col16);
        asm volatile("cp.async.wait_group 0;\n" ::);
        prim::block_sync();
#pragma unroll
        for (u32 sub = 0u; sub < 4u; sub++) {
            u32 st = sub & 1u;
#pragma unroll
            for (u32 bl = 0u; bl < 2u; bl++) {
                u32 b_off = ((sub << 1u) + bl) * 18u;
                u32 sc0_h[2], sc1_h[2];
#pragma unroll
                for (u32 ni = 0u; ni < 2u; ni++) {
                    u32 s0 = *reinterpret_cast<const u16*>(sb0[ni] + b_off);
                    u32 s1 = *reinterpret_cast<const u16*>(sb1[ni] + b_off);
                    sc0_h[ni] = s0 | (s0 << 16u);
                    sc1_h[ni] = s1 | (s1 << 16u);
                }
#pragma unroll
                for (u32 ks_i = 0u; ks_i < 2u; ks_i++) {
                    u32 k16 = (bl << 1u) + ks_i, fb[4][2];
                    {
                        u32 mr0 = m_base + lb_row, tmp[4];
                        mx8_ldm4(tmp, &sm.k.sx[st][mr0][(((k16 << 1u) | lb_half) ^ (mr0 & 7u)) << 2u]);
                        fb[0][0] = tmp[0]; fb[0][1] = tmp[1]; fb[1][0] = tmp[2]; fb[1][1] = tmp[3];
                    }
                    {
                        u32 mr2 = m_base + 16u + lb_row, tmp[4];
                        mx8_ldm4(tmp, &sm.k.sx[st][mr2][(((k16 << 1u) | lb_half) ^ (mr2 & 7u)) << 2u]);
                        fb[2][0] = tmp[0]; fb[2][1] = tmp[1]; fb[3][0] = tmp[2]; fb[3][1] = tmp[3];
                    }
                    u32 q_off = b_off + 2u + (ks_i << 3u) + t;
#pragma unroll
                    for (u32 ni = 0u; ni < 2u; ni++) {
                        u32 a[4] = {i4_dec_s(sb0[ni][q_off], sc0_h[ni]),
                                    i4_dec_s(sb1[ni][q_off], sc1_h[ni]),
                                    i4_dec_s(sb0[ni][q_off + 4u], sc0_h[ni]),
                                    i4_dec_s(sb1[ni][q_off + 4u], sc1_h[ni])};
#pragma unroll
                        for (u32 mi = 0u; mi < 4u; mi++) trellis::mma_m16n8k16(acc[ni][mi], a, fb[mi]);
                    }
                }
            }
            if (sub + 1u < 4u) {
                i4_load_x(sm.k.sx[st ^ 1u], x, r0, (ch << 8u) + ((sub + 1u) << 6u), m, k, r0_t, col16);
            }
            prim::block_sync();
        }
    }
    mx8_epilogue_smem<4u, SPLIT>(sm.out, acc, other, out, m, n, r0, n0, p, mode, tid, wm, n_base, g, t);
}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define I4_RUN(SM, FN, SPLIT, KS) \
    __shared__ SM sm; \
    FN<SPLIT>(x, w, other, out, m, n, k, mode, KS, sm)
#else
#define I4_RUN(SM, FN, SPLIT, KS) \
    (void)x; (void)w; (void)other; (void)out; (void)m; (void)n; (void)k; (void)mode
#endif

#define I4_PAIR(TAG, TH, BLK, SM, FN) \
    extern "C" __global__ __launch_bounds__(TH, BLK) void tessera_mm_int4_##TAG( \
        const float* x, const u8* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode) { \
        I4_RUN(SM, FN, false, 1u); \
    } \
    extern "C" __global__ __launch_bounds__(TH, BLK) void tessera_mm_int4_##TAG##_sk( \
        const float* x, const u8* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode, u32 ks) { \
        (void)ks; \
        I4_RUN(SM, FN, true, ks); \
    }

// Streaming and staged-X INT4_G32 Tensor-Core matmul entries.
I4_PAIR(r16, 128, 4, i4_r16_smem, fg_int4_r16)
I4_PAIR(r64, 256, 2, i4_r64_smem, fg_int4_r64)

#undef I4_PAIR
#undef I4_RUN

#endif
