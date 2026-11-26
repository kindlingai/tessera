#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

#if defined(TESSERA_CUDA)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define DTR_MX8 1
constexpr u32 DTR_TILE = trellis::MX8_TILE;
#define DTR_T8_BLK 3
#else
#define DTR_MX8 0
constexpr u32 DTR_TILE = 512u;
#define DTR_T8_BLK 2
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
extern "C" __global__ __launch_bounds__(256, 3) void dtr_prep(const float* x, const float* suh,
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
extern "C" __global__ __launch_bounds__(256, 2) void dtr_pre_k4_t16(
    const float* xpre, const u8* w, const float* svh, float* out, u32 m, u32 n, u32 k, u32 bits,
    u32 book, u32 turned) {
    (void)bits; (void)book; (void)turned;
    __shared__ dtr_pre_smem<16u> sm;
    dtr_pre_body<4u, trellis::BOOK_MCG, 16u, true>(xpre, w, svh, out, m, n, k, sm);
}

// Generic dense trellis prefill, 64 rows per block.
extern "C" __global__ __launch_bounds__(256, 2) void dtr_pre_gen8(
    const float* xpre, const u8* w, const float* svh, float* out, u32 m, u32 n, u32 k, u32 bits,
    u32 book, u32 turned) {
    __shared__ dtr_pre_smem<8u> sm;
    dtr_pre_gen<8u>(xpre, w, svh, out, m, n, k, bits, book, turned, sm);
}

// Generic dense trellis prefill, 128 rows per block.
extern "C" __global__ __launch_bounds__(256, 2) void dtr_pre_gen16(
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
extern "C" __global__ __launch_bounds__(256, 4) void tessera_mm_mxfp8_prep(
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

#if DTR_MX8
struct alignas(16) mx8_128_smem {
    u32 sw[2][128][36];
    u32 sx[2][128][32];
    u32 sxs[2][128];
};

static __device__ __forceinline__ void mx8_cp16(void* smem, const void* gmem, bool pred) {
    u32 dst = static_cast<u32>(__cvta_generic_to_shared(smem));
    if (pred) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(gmem));
    else *reinterpret_cast<uint4*>(smem) = make_uint4(0u, 0u, 0u, 0u);
}

static __device__ __forceinline__ void mx8_cp4(void* smem, const void* gmem, bool pred) {
    u32 dst = static_cast<u32>(__cvta_generic_to_shared(smem));
    if (pred) asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dst), "l"(gmem));
    else *reinterpret_cast<u32*>(smem) = 0u;
}

static __device__ __forceinline__ void mx8_ldm4(u32 (&r)[4], const void* smem_ptr) {
    u32 addr = static_cast<u32>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
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
        mx8_cp16(&sm.sw[st][nr][col16 << 2u],
                 w + u64(n0 + nr) * row_bytes + aligned_off + (col16 << 4u), n0 + nr < n);
    }
    if (tid < 128u) {
        mx8_cp16(&sm.sw[st][tid][32u], w + u64(n0 + tid) * row_bytes + aligned_off + 128u, n0 + tid < n);
    }
    u32 k_off = ch128 << 7u;
#pragma unroll
    for (u32 dr = 0u; dr < 128u; dr += 32u) {
        u32 mr = r0_t + dr;
        mx8_cp16(&sm.sx[st][mr][(col16 ^ (mr & 7u)) << 2u],
                 xq + u64(r0 + mr) * k + k_off + (col16 << 4u), r0 + mr < m);
    }
    if (tid < 128u) {
        mx8_cp4(&sm.sxs[st][tid], xs + u64(r0 + tid) * nblk + (ch128 << 2u), r0 + tid < m);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

template <u32 PH, u32 BLK>
static __device__ __forceinline__ void mx8_run_blk(
    const u32 (*sw_st)[36], const u32 (*sx_st)[32], const u32* sxs_st,
    u32 n_base, u32 m_base, u32 g, u32 t, u32 lb_row, u32 lb_half, float acc[4][4][4]) {
    constexpr u32 B_OFF = PH * 4u + BLK * 33u;
    constexpr u32 Q_BASE = B_OFF + 1u;
    constexpr u32 SH = (Q_BASE & 3u) << 3u;
    constexpr u32 W_BASE = Q_BASE >> 2u;
    constexpr u32 S_SH = BLK << 3u;
    u32 fb[4][2], sb[4];
    {
        u32 mr0 = m_base + lb_row, tmp[4];
        mx8_ldm4(tmp, &sx_st[mr0][(((BLK << 1u) | lb_half) ^ (mr0 & 7u)) << 2u]);
        fb[0][0] = tmp[0]; fb[0][1] = tmp[1]; fb[1][0] = tmp[2]; fb[1][1] = tmp[3];
    }
    {
        u32 mr2 = m_base + 16u + lb_row, tmp[4];
        mx8_ldm4(tmp, &sx_st[mr2][(((BLK << 1u) | lb_half) ^ (mr2 & 7u)) << 2u]);
        fb[2][0] = tmp[0]; fb[2][1] = tmp[1]; fb[3][0] = tmp[2]; fb[3][1] = tmp[3];
    }
#pragma unroll
    for (u32 mi = 0u; mi < 4u; mi++) sb[mi] = (sxs_st[m_base + (mi << 3u) + g] >> S_SH) & 0xFFu;

    u32 fa_cur[4], fa_nxt[4], sa_cur, sa_nxt;
    {
        u32 nr0 = n_base + g, nr1 = nr0 + 8u;
        const u32 *sw0 = sw_st[nr0], *sw1 = sw_st[nr1];
        sa_cur = u32(reinterpret_cast<const u8*>((t & 1u) ? sw1 : sw0)[B_OFF]);
        mx8_load_fa<SH>(sw0, sw1, W_BASE, t, fa_cur);
    }
#pragma unroll
    for (u32 ni = 0u; ni < 4u; ni++) {
        if (ni + 1u < 4u) {
            u32 nr0 = n_base + ((ni + 1u) << 4u) + g, nr1 = nr0 + 8u;
            const u32 *sw0 = sw_st[nr0], *sw1 = sw_st[nr1];
            sa_nxt = u32(reinterpret_cast<const u8*>((t & 1u) ? sw1 : sw0)[B_OFF]);
            mx8_load_fa<SH>(sw0, sw1, W_BASE, t, fa_nxt);
        }
#pragma unroll
        for (u32 mi = 0u; mi < 4u; mi++) {
            trellis::mma_mx8(acc[ni][mi], fa_cur, fb[mi], sa_cur, sb[mi]);
        }
        if (ni + 1u < 4u) {
            fa_cur[0] = fa_nxt[0]; fa_cur[1] = fa_nxt[1];
            fa_cur[2] = fa_nxt[2]; fa_cur[3] = fa_nxt[3];
            sa_cur = sa_nxt;
        }
    }
}

template <u32 PH>
static __device__ __forceinline__ void mx8_phase_step(
    mx8_128_smem& sm, u32 st, u32 nxt_ch, bool has_nxt,
    const u8* xq, const u8* xs, const u8* w, u32 m, u32 n, u32 k, u32 nblk, u64 row_bytes,
    u32 n0, u32 r0, u32 tid, u32 r0_t, u32 col16,
    u32 n_base, u32 m_base, u32 g, u32 t, u32 lb_row, u32 lb_half, float acc[4][4][4]) {
    if (has_nxt) {
        mx8_load_stage(sm, st ^ 1u, nxt_ch, xq, xs, w, m, n, k, nblk, row_bytes, n0, r0, tid, r0_t, col16);
        asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
        asm volatile("cp.async.wait_group 0;\n" ::);
    }
    prim::block_sync();
    mx8_run_blk<PH, 0u>(sm.sw[st], sm.sx[st], sm.sxs[st], n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 1u>(sm.sw[st], sm.sx[st], sm.sxs[st], n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 2u>(sm.sw[st], sm.sx[st], sm.sxs[st], n_base, m_base, g, t, lb_row, lb_half, acc);
    mx8_run_blk<PH, 3u>(sm.sw[st], sm.sx[st], sm.sxs[st], n_base, m_base, g, t, lb_row, lb_half, acc);
    if (has_nxt) prim::block_sync();
}

template <bool SPLIT>
static __device__ __forceinline__ void fg_mxfp8_mx8(
    const float* xpre, const u8* w, const float* other, float* out,
    u32 m, u32 n, u32 k, u32 mode, u32 ks) {
    extern __shared__ u8 raw_smem[];
    mx8_128_smem& sm = *reinterpret_cast<mx8_128_smem*>(raw_smem);
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
    u32 g = lane >> 2u, t = lane & 3u, wn = warp & 1u, wm = warp >> 1u;
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
    u32 n_base = wn << 6u, m_base = wm << 5u;
    u32 lb_row = ((lane >> 4u) << 3u) | (lane & 7u), lb_half = (lane >> 3u) & 1u;

    float acc[4][4][4];
#pragma unroll
    for (u32 ni = 0u; ni < 4u; ni++)
#pragma unroll
        for (u32 mi = 0u; mi < 4u; mi++)
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
#pragma unroll
    for (u32 ni = 0u; ni < 4u; ni++) {
        u32 col0 = n0 + n_base + (ni << 4u) + g, col1 = col0 + 8u;
#pragma unroll
        for (u32 mi = 0u; mi < 4u; mi++) {
            u32 row0 = r0 + m_base + (mi << 3u) + (t << 1u);
            if (row0 < m) {
                u64 off0 = u64(SPLIT ? (p * m + row0) : row0) * n;
                if (col0 < n) out[off0 + col0] = SPLIT ? acc[ni][mi][0] : mx8_epilogue(acc[ni][mi][0], other, off0 + col0, col0, mode);
                if (col1 < n) out[off0 + col1] = SPLIT ? acc[ni][mi][2] : mx8_epilogue(acc[ni][mi][2], other, off0 + col1, col1, mode);
            }
            if (row0 + 1u < m) {
                u64 off1 = u64(SPLIT ? (p * m + row0 + 1u) : (row0 + 1u)) * n;
                if (col0 < n) out[off1 + col0] = SPLIT ? acc[ni][mi][1] : mx8_epilogue(acc[ni][mi][1], other, off1 + col0, col0, mode);
                if (col1 < n) out[off1 + col1] = SPLIT ? acc[ni][mi][3] : mx8_epilogue(acc[ni][mi][3], other, off1 + col1, col1, mode);
            }
        }
    }
}
#endif

// Blackwell native MXFP8 MMA matmul, 128x128 per block.
extern "C" __global__ __launch_bounds__(256, 1) void tessera_mm_mxfp8_mx8(
    const float* xpre, const u8* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode) {
#if DTR_MX8
    fg_mxfp8_mx8<false>(xpre, w, other, out, m, n, k, mode, 1u);
#else
    (void)xpre; (void)w; (void)other; (void)out; (void)m; (void)n; (void)k; (void)mode;
#endif
}

// Split-K Blackwell native MXFP8 MMA matmul, 128x128 per block.
extern "C" __global__ __launch_bounds__(256, 1) void tessera_mm_mxfp8_mx8_sk(
    const float* xpre, const u8* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode,
    u32 ks) {
#if DTR_MX8
    fg_mxfp8_mx8<true>(xpre, w, other, out, m, n, k, mode, ks);
#else
    (void)xpre; (void)w; (void)other; (void)out; (void)m; (void)n; (void)k; (void)mode; (void)ks;
#endif
}

#endif
