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
extern "C" __global__ __launch_bounds__(256, DTR_PREP_BLK) void dtr_prep(const trellis::x_t* x,
                                                              const float* suh,
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
            trellis::load_x4(x + u64(r0 + r) * k + col, v);
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
                                                    const float* svh, trellis::y_t* out, u32 m,
                                                    u32 n, u32 k, dtr_pre_smem<NT>& sm) {
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
        trellis::store_tile<NT>(out, r0, m, n, ob << 7u, col, rc, acc);
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
                trellis::store_y4(out + u64(r0 + r) * n + base,
                                  make_float4(v[0] * sc[0], v[1] * sc[1], v[2] * sc[2], v[3] * sc[3]));
            }
        }
    }
}

template <u32 K, u32 BOOK, u32 NT>
static __device__ __noinline__ void dtr_pre_k(const float* xpre, const u8* w, const float* svh,
                                              trellis::y_t* out, u32 m, u32 n, u32 k, u32 turned,
                                              dtr_pre_smem<NT>& sm) {
    if (turned != 0u) dtr_pre_body<K, BOOK, NT, true>(xpre, w, svh, out, m, n, k, sm);
    else dtr_pre_body<K, BOOK, NT, false>(xpre, w, svh, out, m, n, k, sm);
}

template <u32 BOOK, u32 NT>
static __device__ void dtr_pre_book(const float* xpre, const u8* w, const float* svh,
                                    trellis::y_t* out, u32 m, u32 n, u32 k, u32 bits, u32 turned,
                                    dtr_pre_smem<NT>& sm) {
    if (bits == 2u) dtr_pre_k<2u, BOOK, NT>(xpre, w, svh, out, m, n, k, turned, sm);
    else if (bits == 3u) dtr_pre_k<3u, BOOK, NT>(xpre, w, svh, out, m, n, k, turned, sm);
    else dtr_pre_k<4u, BOOK, NT>(xpre, w, svh, out, m, n, k, turned, sm);
}

template <u32 NT>
static __device__ void dtr_pre_gen(const float* xpre, const u8* w, const float* svh,
                                   trellis::y_t* out, u32 m, u32 n, u32 k, u32 bits, u32 book,
                                   u32 turned, dtr_pre_smem<NT>& sm) {
    if (book == trellis::BOOK_PMCGX) dtr_pre_book<trellis::BOOK_PMCGX, NT>(xpre, w, svh, out, m, n, k, bits, turned, sm);
    else if (book == trellis::BOOK_MUL1) dtr_pre_book<trellis::BOOK_MUL1, NT>(xpre, w, svh, out, m, n, k, bits, turned, sm);
    else dtr_pre_book<trellis::BOOK_MCG, NT>(xpre, w, svh, out, m, n, k, bits, turned, sm);
}

// Rotated 4-bit dense trellis prefill, 64 rows per block.
extern "C" __global__ __launch_bounds__(256, DTR_T8_BLK) void dtr_pre_k4_t8(
    const float* xpre, const u8* w, const float* svh, trellis::y_t* out, u32 m, u32 n, u32 k,
    u32 bits, u32 book, u32 turned) {
    (void)bits; (void)book; (void)turned;
    __shared__ dtr_pre_smem<8u> sm;
    dtr_pre_body<4u, trellis::BOOK_MCG, 8u, true>(xpre, w, svh, out, m, n, k, sm);
}

// Rotated 4-bit dense trellis prefill, 128 rows per block.
extern "C" __global__ __launch_bounds__(256, DTR_T16_BLK) void dtr_pre_k4_t16(
    const float* xpre, const u8* w, const float* svh, trellis::y_t* out, u32 m, u32 n, u32 k,
    u32 bits, u32 book, u32 turned) {
    (void)bits; (void)book; (void)turned;
    __shared__ dtr_pre_smem<16u> sm;
    dtr_pre_body<4u, trellis::BOOK_MCG, 16u, true>(xpre, w, svh, out, m, n, k, sm);
}

// Generic dense trellis prefill, 64 rows per block.
extern "C" __global__ __launch_bounds__(256, DTR_GEN8_BLK) void dtr_pre_gen8(
    const float* xpre, const u8* w, const float* svh, trellis::y_t* out, u32 m, u32 n, u32 k,
    u32 bits, u32 book, u32 turned) {
    __shared__ dtr_pre_smem<8u> sm;
    dtr_pre_gen<8u>(xpre, w, svh, out, m, n, k, bits, book, turned, sm);
}

// Generic dense trellis prefill, 128 rows per block.
extern "C" __global__ __launch_bounds__(256, DTR_GEN16_BLK) void dtr_pre_gen16(
    const float* xpre, const u8* w, const float* svh, trellis::y_t* out, u32 m, u32 n, u32 k,
    u32 bits, u32 book, u32 turned) {
    __shared__ dtr_pre_smem<16u> sm;
    dtr_pre_gen<16u>(xpre, w, svh, out, m, n, k, bits, book, turned, sm);
}

#if defined(O_MXFP8)
using mx8_o_t = u8;
static __device__ __forceinline__ float mx8_load_o(const u8* p, u64 idx) {
    u64 base = (idx >> 5u) * 33ull;
    u32 e = p[base];
    float sc = __uint_as_float(e == 0u ? 0x00400000u : (e << 23u));
    return sc * trellis::e4m3_to_f32(p[base + 1u + (idx & 31u)]);
}
#else
using mx8_o_t = trellis::o_t;
static __device__ __forceinline__ float mx8_load_o(const trellis::o_t* p, u64 idx) {
    return trellis::load_o(p, idx);
}
#endif

#if defined(Y_MXFP8)
using mx8_y_t = u8;
#else
using mx8_y_t = trellis::y_t;
#endif

static __device__ __forceinline__ float mx8_silu(float v) {
    return v / (1.0f + expf(-v));
}

static __device__ __forceinline__ float mx8_epilogue(float v, const mx8_o_t* other, u64 off,
                                                     u32 col, u32 mode) {
    if (mode == 1u) return v + mx8_load_o(other, off);
    if (mode == 2u) return v + mx8_load_o(other, col);
    if (mode == 3u) return mx8_silu(v) * mx8_load_o(other, off);
    if (mode == 4u) return v * mx8_load_o(other, off);
    if (mode == 5u) return mx8_silu(v);
    return v;
}

#if DTR_MX8
// Quantizes 32 values across 8 lanes into one mxfp8 block.
static __device__ __forceinline__ void mx8_store_blk4(
    float4 v, u8* out, u64 row, u32 nblk, u32 col, u32 sub, bool valid) {
    float amax =
        valid ? fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))) : 0.0f;
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 4u));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2u));
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1u));
    if (!valid) return;
    float inv;
    u32 e = trellis::mx_scale(amax, inv);
    u16 p0 = trellis::f2_e4m3x2(v.x * inv, v.y * inv);
    u16 p1 = trellis::f2_e4m3x2(v.z * inv, v.w * inv);
    u8* dst = out + (row * u64(nblk) + u64(col >> 5u)) * 33ull;
    if (sub == 0u) dst[0] = u8(e);
    u32 q = 1u + (sub << 2u);
    dst[q] = u8(p0 & 0xFFu);
    dst[q + 1u] = u8(p0 >> 8u);
    dst[q + 2u] = u8(p1 & 0xFFu);
    dst[q + 3u] = u8(p1 >> 8u);
}
#endif

// Quantizes X[m, k] once to packed E4M3 and UE8M0 scales.
extern "C" __global__ __launch_bounds__(256, MM_PREP_BLK) void tessera_mm_mxfp8_prep(
    const trellis::x_t* x, float* out, u32 m, u32 k) {
#if DTR_MX8
    u32 blk = blockIdx.x * 32u + (threadIdx.x >> 3u);
    u32 row = blockIdx.y;
    u32 nblk = k >> 5u;
    if (row >= m || blk >= nblk) return;
    u32 sub = threadIdx.x & 7u;
    float4 v = trellis::load_x4(x + u64(row) * k + (blk << 5u) + (sub << 2u));
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

#if !defined(Y_MXFP8)
template <bool SPLIT>
static __device__ __forceinline__ void mx8_store_vec4(
    float4 v, const mx8_o_t* other, trellis::y_t* out, u64 off, u32 col, u32 n, u32 mode,
    bool vec4) {
    if (vec4 && col + 3u < n) {
        if (!SPLIT && mode != 0u) {
            v.x = mx8_epilogue(v.x, other, off, col, mode);
            v.y = mx8_epilogue(v.y, other, off + 1u, col + 1u, mode);
            v.z = mx8_epilogue(v.z, other, off + 2u, col + 2u, mode);
            v.w = mx8_epilogue(v.w, other, off + 3u, col + 3u, mode);
        }
        trellis::store_y4(out + off, v);
    } else {
        float vals[4] = {v.x, v.y, v.z, v.w};
        for (u32 j = 0u; j < 4u && col + j < n; j++) {
            trellis::store_y(out, off + j, SPLIT ? vals[j] : mx8_epilogue(vals[j], other, off + j, col + j, mode));
        }
    }
}
#endif

template <u32 MI, bool SPLIT>
static __device__ __forceinline__ void mx8_epilogue_smem(
    float (*out_sm)[132], const float (&acc)[2][MI][4], const mx8_o_t* other,
    mx8_y_t* out, u32 m, u32 n, u32 r0, u32 n0, u32 p, u32 mode, u32 tid, u32 wm, u32 n_base,
    u32 g, u32 t) {
    constexpr u32 Q_ROWS = MI << 3u;
    constexpr u32 Q_SHIFT = (MI == 8u) ? 6u : 5u;
    u32 sr = tid >> 5u, sc4 = (tid & 31u) << 2u;
#if !defined(Y_MXFP8)
    bool vec4 = (n & 3u) == 0u;
#endif
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
#if defined(Y_MXFP8) && DTR_MX8
            bool valid = row < m && col < n;
            float4 v = valid ? *reinterpret_cast<const float4*>(&out_sm[sr + dr][sc4])
                             : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (valid && !SPLIT && mode != 0u) {
                u64 off = u64(row) * n + col;
                v.x = mx8_epilogue(v.x, other, off, col, mode);
                v.y = mx8_epilogue(v.y, other, off + 1u, col + 1u, mode);
                v.z = mx8_epilogue(v.z, other, off + 2u, col + 2u, mode);
                v.w = mx8_epilogue(v.w, other, off + 3u, col + 3u, mode);
            }
            mx8_store_blk4(v, out, row, n >> 5u, col, tid & 7u, valid);
#else
            if (row < m && col < n) {
                float4 v = *reinterpret_cast<const float4*>(&out_sm[sr + dr][sc4]);
                u64 off = u64(SPLIT ? (p * m + row) : row) * n + col;
                mx8_store_vec4<SPLIT>(v, other, out, off, col, n, mode, vec4);
            }
#endif
        }
        if (q == 0u) prim::block_sync();
    }
}
#endif

#if DTR_MX8
union alignas(16) mx8_128_smem {
    struct {
        u32 sw[2][128][36];
#if defined(X_MXFP8)
        u32 sx[2][128][36];
#else
        u32 sx[2][128][32];
        u32 sxs[2][128];
#endif
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

#if defined(X_MXFP8)
template <u32 SH>
static __device__ __forceinline__ void mx8_load_fb(
    const u32* sx, u32 w_base, u32 t, u32 (&fb)[2]) {
    u32 w0 = w_base + t;
    if constexpr (SH == 0u) {
        fb[0] = sx[w0];
        fb[1] = sx[w0 + 4u];
    } else {
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(fb[0]) : "r"(sx[w0]), "r"(sx[w0 + 1u]), "n"(SH));
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(fb[1]) : "r"(sx[w0 + 4u]), "r"(sx[w0 + 5u]), "n"(SH));
    }
}
#endif

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
#if defined(X_MXFP8)
    (void)xs; (void)k; (void)nblk;
#pragma unroll
    for (u32 dr = 0u; dr < 128u; dr += 32u) {
        u32 mr = r0_t + dr;
        mx8_cp16(&sm.k.sx[st][mr][col16 << 2u],
                 xq + u64(r0 + mr) * row_bytes + aligned_off + (col16 << 4u), r0 + mr < m);
    }
    if (tid < 128u) {
        mx8_cp16(&sm.k.sx[st][tid][32u], xq + u64(r0 + tid) * row_bytes + aligned_off + 128u, r0 + tid < m);
    }
#else
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
#endif
    asm volatile("cp.async.commit_group;\n" ::);
}

template <u32 PH, u32 BLK>
static __device__ __forceinline__ void mx8_run_blk(
#if defined(X_MXFP8)
    const u32 (*sw_st)[36], const u32 (*sx_st)[36],
#else
    const u32 (*sw_st)[36], const u32 (*sx_st)[32], const u32 sb4[8],
#endif
    u32 n_base, u32 m_base, u32 g, u32 t, u32 lb_row, u32 lb_half, float acc[2][8][4]) {
    constexpr u32 B_OFF = PH * 4u + BLK * 33u;
    constexpr u32 Q_BASE = B_OFF + 1u;
    constexpr u32 SH = (Q_BASE & 3u) << 3u;
    constexpr u32 W_BASE = Q_BASE >> 2u;
    u32 fb[8][2], sb[8];
#if defined(X_MXFP8)
    (void)lb_row; (void)lb_half;
#pragma unroll
    for (u32 mi = 0u; mi < 8u; mi++) {
        const u32* sx_mr = sx_st[m_base + (mi << 3u) + g];
        sb[mi] = u32(reinterpret_cast<const u8*>(sx_mr)[B_OFF]);
        mx8_load_fb<SH>(sx_mr, W_BASE, t, fb[mi]);
    }
#else
    constexpr u32 S_SH = BLK << 3u;
#pragma unroll
    for (u32 q = 0u; q < 4u; q++) {
        u32 mr = m_base + (q << 4u) + lb_row, tmp[4];
        mx8_ldm4(tmp, &sx_st[mr][(((BLK << 1u) | lb_half) ^ (mr & 7u)) << 2u]);
        fb[q << 1u][0] = tmp[0]; fb[q << 1u][1] = tmp[1];
        fb[(q << 1u) + 1u][0] = tmp[2]; fb[(q << 1u) + 1u][1] = tmp[3];
    }
#pragma unroll
    for (u32 mi = 0u; mi < 8u; mi++) sb[mi] = (sb4[mi] >> S_SH) & 0xFFu;
#endif

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
#if defined(X_MXFP8)
    mx8_run_blk<PH, 0u>(sm.k.sw[st], sm.k.sx[st], n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 1u>(sm.k.sw[st], sm.k.sx[st], n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 2u>(sm.k.sw[st], sm.k.sx[st], n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 3u>(sm.k.sw[st], sm.k.sx[st], n_base, m_base, g, t, lb_row, lb_half, acc);
#else
    u32 sb4[8];
#pragma unroll
    for (u32 mi = 0u; mi < 8u; mi++) sb4[mi] = sm.k.sxs[st][m_base + (mi << 3u) + g];
    mx8_run_blk<PH, 0u>(sm.k.sw[st], sm.k.sx[st], sb4, n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 1u>(sm.k.sw[st], sm.k.sx[st], sb4, n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 2u>(sm.k.sw[st], sm.k.sx[st], sb4, n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 3u>(sm.k.sw[st], sm.k.sx[st], sb4, n_base, m_base, g, t, lb_row, lb_half, acc);
#endif
    if (has_nxt) prim::block_sync();
}

template <bool SPLIT>
static __device__ __forceinline__ void fg_mxfp8_mx8(
    const float* xpre, const u8* w, const mx8_o_t* other, mx8_y_t* out,
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
    const float* xpre, const u8* w, const mx8_o_t* other, mx8_y_t* out, u32 m, u32 n,
    u32 k, u32 mode) {
#if DTR_MX8
    fg_mxfp8_mx8<false>(xpre, w, other, out, m, n, k, mode, 1u);
#else
    (void)xpre; (void)w; (void)other; (void)out; (void)m; (void)n; (void)k; (void)mode;
#endif
}

// Split-K Blackwell native MXFP8 MMA matmul, 128x128 per block.
extern "C" __global__ __launch_bounds__(256, MM_MX8_SK_BLK) void tessera_mm_mxfp8_mx8_sk(
    const float* xpre, const u8* w, const mx8_o_t* other, mx8_y_t* out, u32 m, u32 n,
    u32 k, u32 mode, u32 ks) {
#if DTR_MX8
    fg_mxfp8_mx8<true>(xpre, w, other, out, m, n, k, mode, ks);
#else
    (void)xpre; (void)w; (void)other; (void)out; (void)m; (void)n; (void)k; (void)mode; (void)ks;
#endif
}

// Split-K reduction with optional MXFP8 epilogue and output quantization.
extern "C" __global__ __launch_bounds__(256, MM_PREP_BLK) void tessera_mm_reduce_mx8(
    const float* part, const mx8_o_t* other, mx8_y_t* out, u32 m, u32 n, u32 ks, u32 mode) {
#if DTR_MX8
    u32 tid = threadIdx.x;
    u32 row = blockIdx.y;
    u32 col = (blockIdx.x * 256u + tid) << 2u;
    bool valid = row < m && col < n;
    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (valid) {
        u64 off = u64(row) * n + col;
        u64 mn = u64(m) * n;
        for (u32 p = 0u; p < ks; p++) {
            float4 s = *reinterpret_cast<const float4*>(part + u64(p) * mn + off);
            v.x += s.x; v.y += s.y; v.z += s.z; v.w += s.w;
        }
        if (mode != 0u) {
            v.x = mx8_epilogue(v.x, other, off, col, mode);
            v.y = mx8_epilogue(v.y, other, off + 1u, col + 1u, mode);
            v.z = mx8_epilogue(v.z, other, off + 2u, col + 2u, mode);
            v.w = mx8_epilogue(v.w, other, off + 3u, col + 3u, mode);
        }
    }
#if defined(Y_MXFP8)
    mx8_store_blk4(v, out, row, n >> 5u, col, tid & 7u, valid);
#else
    if (valid) trellis::store_y4(out + u64(row) * n + col, v);
#endif
#else
    (void)part; (void)other; (void)out; (void)m; (void)n; (void)ks; (void)mode;
#endif
}

#if defined(A_BF16)
using nmx_a_t = u16;
#else
using nmx_a_t = float;
#endif
#if defined(B_BF16)
using nmx_b_t = u16;
#else
using nmx_b_t = float;
#endif
#if defined(C_BF16)
using nmx_c_t = u16;
#else
using nmx_c_t = float;
#endif
#if defined(W_BF16)
using nmx_w_t = u16;
#else
using nmx_w_t = float;
#endif
#if defined(S_BF16)
using nmx_s_t = u16;
#else
using nmx_s_t = float;
#endif

static __device__ __forceinline__ float4 nmx_load_sum(
    const nmx_a_t* a, const nmx_b_t* b, const nmx_c_t* c, u64 off, u32 flags) {
    float4 s = trellis::load_e4(a + off);
    if (flags & 1u) {
        float4 vb = trellis::load_e4(b + off);
        s.x += vb.x; s.y += vb.y; s.z += vb.z; s.w += vb.w;
    }
    if (flags & 2u) {
        float4 vc = trellis::load_e4(c + off);
        s.x += vc.x; s.y += vc.y; s.z += vc.z; s.w += vc.w;
    }
    return s;
}

// Fused RMS norm and optional add to OCP mxfp8.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_norm_mxfp8(
    const nmx_a_t* a, const nmx_b_t* b, const nmx_c_t* c, const nmx_w_t* w, u8* out,
    nmx_s_t* sum_out, u32 rows, u32 width, float eps, u32 flags) {
#if DTR_MX8
    u32 row = blockIdx.x;
    if (row >= rows) return;
    u32 tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    u64 row_off = u64(row) * width;
    float4 cache[8];
    float acc = 0.0f;
    u32 trip = 0u;
    for (u32 base = 0u; base < width; base += 1024u, trip++) {
        u32 col = base + (tid << 2u);
        if (col < width) {
            float4 s = nmx_load_sum(a, b, c, row_off + col, flags);
            if (flags & 8u) trellis::store_e4(sum_out + row_off + col, s);
            if (trip < 8u) cache[trip] = s;
            acc += s.x * s.x + s.y * s.y + s.z * s.z + s.w * s.w;
        }
    }
#pragma unroll
    for (u32 off = 16u; off > 0u; off >>= 1u) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
    __shared__ float wsum[8];
    if (lane == 0u) wsum[warp] = acc;
    prim::block_sync();
    float tot = 0.0f;
#pragma unroll
    for (u32 i = 0u; i < 8u; i++) tot += wsum[i];
    float inv = rsqrtf(tot / float(width) + eps);
    u32 nblk = width >> 5u;
    bool use_cache = width <= 8192u;
    trip = 0u;
    for (u32 base = 0u; base < width; base += 1024u, trip++) {
        u32 col = base + (tid << 2u);
        bool valid = col < width;
        float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (valid) {
            if (use_cache) s = cache[trip];
            else if (flags & 16u) s = trellis::load_e4(sum_out + row_off + col);
            else s = nmx_load_sum(a, b, c, row_off + col, flags);
            if (flags & 4u) {
                float4 gw = trellis::load_e4(w + col);
                s.x *= inv * gw.x; s.y *= inv * gw.y; s.z *= inv * gw.z; s.w *= inv * gw.w;
            } else {
                s.x *= inv; s.y *= inv; s.z *= inv; s.w *= inv;
            }
        }
        mx8_store_blk4(s, out, row, nblk, col, tid & 7u, valid);
    }
#else
    (void)a; (void)b; (void)c; (void)w; (void)out; (void)sum_out;
    (void)rows; (void)width; (void)eps; (void)flags;
#endif
}

#endif
