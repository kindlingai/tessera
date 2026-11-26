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

#endif
