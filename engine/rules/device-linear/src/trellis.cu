#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

#ifndef DTR_T1_BLK
#define DTR_T1_BLK 4
#endif
#ifndef DTR_T2_BLK
#define DTR_T2_BLK 2
#endif
#ifndef DTR_T4_BLK
#define DTR_T4_BLK 2
#endif
#ifndef DTR_GEN_BLK
#define DTR_GEN_BLK 2
#endif
#ifndef DTR_RED_BLK
#define DTR_RED_BLK 4
#endif

#if defined(TESSERA_CUDA)

using dtr_smem = trellis::stage_smem<float[32][128]>;

template <u32 K, u32 BOOK, u32 NT, bool TURNED, bool SPLIT>
static __device__ __forceinline__ void dtr_body(const trellis::x_t* x, const u8* w, const float* suh,
                                                const float* svh, trellis::y_t* out, u32 m, u32 n,
                                                u32 k, u32 ks, dtr_smem& sm) {
    constexpr u32 TB = 32u * K;
    constexpr u32 CHUNK = 4u / NT;
    u32 nblk = n >> 7u;
    u32 ob = blockIdx.x % nblk;
    u32 p = blockIdx.x / nblk;
    u32 r0 = blockIdx.y * (8u * NT);
    if (ob >= nblk || p >= ks || r0 >= m) return;
    u32 nr = min(8u * NT, m - r0);
    u32 tid = threadIdx.x;
    u32 lane = tid & 31u;
    u32 j = tid >> 5u;
    u32 blk_per = (k >> 7u) / ks;
    u32 ib0 = p * blk_per;
    bool aligned = (reinterpret_cast<u64>(w + trellis::HEAD) & 15ull) == 0ull;
    bool input_side =
        (*reinterpret_cast<const u32*>(w + 8u) & trellis::SIDE_MASK) == trellis::INPUT_SIDE;
    u64 sG = (input_side ? u64(n >> 1u) : 64ull) * u64(TB);
    const u8* gb = trellis::tile_ptr<K>(w, k, n, 0u, ib0 << 3u, ob, input_side);
    u32 nw[8];
    trellis::blk_load<K>(gb, tid, aligned, nw);
    trellis::word_map wm = trellis::map_word<K>(lane);
    float acc[NT][4];
#pragma unroll
    for (u32 ti = 0u; ti < NT; ti++) acc[ti][0] = acc[ti][1] = acc[ti][2] = acc[ti][3] = 0.0f;
    for (u32 c0 = 0u; c0 < blk_per; c0 += CHUNK) {
        u32 nb = min(CHUNK, blk_per - c0);
        if (c0 > 0u) prim::block_sync();
        u32 wpb = (nb <= 2u) ? (8u / nb) : 2u, b = j / wpb, r_base = j - b * wpb;
        if (b < nb && r_base < nr) {
            u32 col = ((ib0 + c0 + b) << 7u) + (lane << 2u);
            float sc[4];
            if constexpr (TURNED) trellis::load4(suh + col, sc);
            for (u32 r = r_base; r < nr; r += wpb) {
                float v[4];
                trellis::load_x4(x + u64(r0 + r) * k + col, v);
                if constexpr (TURNED) {
                    v[0] *= sc[0]; v[1] *= sc[1]; v[2] *= sc[2]; v[3] *= sc[3];
                    trellis::wht128(v, trellis::INV_SQRT128);
                }
                trellis::store_bfrag(sm.k.btab + ((r >> 3u) * nb + b) * 512u, r & 7u, lane, v);
            }
        }
        trellis::blk_stream<K, BOOK, NT>(gb, sG, c0, nb, blk_per, tid, j, lane, aligned, wm, nw,
                                         sm.k.wtiles, sm.k.btab + (lane << 1u), nb * 512u, acc);
    }
    u32 col = (j << 4u) + (lane >> 2u), c = lane & 3u;
    if constexpr (SPLIT) {
        trellis::y_t* pp = out + (u64(ob) * ks + p) * m * 128u;
        trellis::store_tile<NT>(pp, r0, m, 128u, 0u, col, c << 1u, acc);
    } else if constexpr (!TURNED) {
        trellis::store_tile<NT>(out, r0, m, n, ob << 7u, col, c << 1u, acc);
    } else {
        prim::block_sync();
#pragma unroll
        for (u32 ti = 0u; ti < NT; ti++) {
            u32 lr = (ti << 3u) + (c << 1u);
            if (lr < nr) { sm.out[lr][col] = acc[ti][0]; sm.out[lr][col + 8u] = acc[ti][2]; }
            if (lr + 1u < nr) { sm.out[lr + 1u][col] = acc[ti][1]; sm.out[lr + 1u][col + 8u] = acc[ti][3]; }
        }
        prim::block_sync();
        u32 base = (ob << 7u) + (lane << 2u);
        float sc[4];
        trellis::load4(svh + base, sc);
        for (u32 r = j; r < nr; r += 8u) {
            float v[4];
            trellis::load4(&sm.out[r][lane << 2u], v);
            trellis::wht128(v, trellis::INV_SQRT128);
            trellis::store_y4(out + u64(r0 + r) * n + base,
                              make_float4(v[0] * sc[0], v[1] * sc[1], v[2] * sc[2], v[3] * sc[3]));
        }
    }
}

template <u32 K, u32 BOOK, u32 NT>
static __device__ __noinline__ void dtr_gen_k(const trellis::x_t* x, const u8* w, const float* suh,
                                              const float* svh, trellis::y_t* out, u32 m, u32 n,
                                              u32 k, u32 ks, u32 turned, dtr_smem& sm) {
    if (ks > 1u) {
        if (turned != 0u) dtr_body<K, BOOK, NT, true, true>(x, w, suh, svh, out, m, n, k, ks, sm);
        else dtr_body<K, BOOK, NT, false, true>(x, w, suh, svh, out, m, n, k, ks, sm);
    } else {
        if (turned != 0u) dtr_body<K, BOOK, NT, true, false>(x, w, suh, svh, out, m, n, k, 1u, sm);
        else dtr_body<K, BOOK, NT, false, false>(x, w, suh, svh, out, m, n, k, 1u, sm);
    }
}

template <u32 BOOK, u32 NT>
static __device__ void dtr_gen_book(const trellis::x_t* x, const u8* w, const float* suh,
                                    const float* svh, trellis::y_t* out, u32 m, u32 n, u32 k,
                                    u32 ks, u32 bits, u32 turned, dtr_smem& sm) {
    if (bits == 2u) dtr_gen_k<2u, BOOK, NT>(x, w, suh, svh, out, m, n, k, ks, turned, sm);
    else if (bits == 3u) dtr_gen_k<3u, BOOK, NT>(x, w, suh, svh, out, m, n, k, ks, turned, sm);
    else dtr_gen_k<4u, BOOK, NT>(x, w, suh, svh, out, m, n, k, ks, turned, sm);
}

template <u32 NT>
static __device__ void dtr_gen(const trellis::x_t* x, const u8* w, const float* suh,
                               const float* svh, trellis::y_t* out, u32 m, u32 n, u32 k, u32 ks,
                               u32 bits, u32 book, u32 turned, dtr_smem& sm) {
    if (book == trellis::BOOK_PMCGX) {
        dtr_gen_book<trellis::BOOK_PMCGX, NT>(x, w, suh, svh, out, m, n, k, ks, bits, turned, sm);
    } else if (book == trellis::BOOK_MUL1) {
        dtr_gen_book<trellis::BOOK_MUL1, NT>(x, w, suh, svh, out, m, n, k, ks, bits, turned, sm);
    } else {
        dtr_gen_book<trellis::BOOK_MCG, NT>(x, w, suh, svh, out, m, n, k, ks, bits, turned, sm);
    }
}

#define DTR_DEF(NAME, K, BOOK, NT, TURNED, SPLIT, MIN_BLK)                                    \
    extern "C" __global__ __launch_bounds__(256, MIN_BLK) void NAME(                          \
        const trellis::x_t* x, const u8* w, const float* suh, const float* svh,               \
        trellis::y_t* out, u32 m, u32 n, u32 k, u32 ks, u32 bits, u32 book, u32 turned) {     \
        (void)bits;                                                                           \
        (void)book;                                                                           \
        (void)turned;                                                                         \
        __shared__ dtr_smem sm;                                                               \
        dtr_body<K, BOOK, NT, TURNED, SPLIT>(x, w, suh, svh, out, m, n, k, ks, sm);           \
    }

DTR_DEF(dtr_k4_t1s, 4u, trellis::BOOK_MCG, 1u, true, true, DTR_T1_BLK)
DTR_DEF(dtr_k4_t1, 4u, trellis::BOOK_MCG, 1u, true, false, DTR_T1_BLK)
DTR_DEF(dtr_k4_p1s, 4u, trellis::BOOK_MCG, 1u, false, true, DTR_T1_BLK)
DTR_DEF(dtr_k4_p1, 4u, trellis::BOOK_MCG, 1u, false, false, DTR_T1_BLK)
DTR_DEF(dtr_k4_t2s, 4u, trellis::BOOK_MCG, 2u, true, true, DTR_T2_BLK)
DTR_DEF(dtr_k4_t2, 4u, trellis::BOOK_MCG, 2u, true, false, DTR_T2_BLK)
DTR_DEF(dtr_k4_t4, 4u, trellis::BOOK_MCG, 4u, true, false, DTR_T4_BLK)

// Generic dense trellis matmul for up to 8 rows.
extern "C" __global__ __launch_bounds__(256, DTR_GEN_BLK) void dtr_gen1(
    const trellis::x_t* x, const u8* w, const float* suh, const float* svh, trellis::y_t* out,
    u32 m, u32 n, u32 k, u32 ks, u32 bits, u32 book, u32 turned) {
    __shared__ dtr_smem sm;
    dtr_gen<1u>(x, w, suh, svh, out, m, n, k, ks, bits, book, turned, sm);
}

// Generic dense trellis matmul for 16-row tiles.
extern "C" __global__ __launch_bounds__(256, DTR_GEN_BLK) void dtr_gen2(
    const trellis::x_t* x, const u8* w, const float* suh, const float* svh, trellis::y_t* out,
    u32 m, u32 n, u32 k, u32 ks, u32 bits, u32 book, u32 turned) {
    __shared__ dtr_smem sm;
    dtr_gen<2u>(x, w, suh, svh, out, m, n, k, ks, bits, book, turned, sm);
}

// Generic dense trellis matmul for 32-row tiles.
extern "C" __global__ __launch_bounds__(256, DTR_GEN_BLK) void dtr_gen4(
    const trellis::x_t* x, const u8* w, const float* suh, const float* svh, trellis::y_t* out,
    u32 m, u32 n, u32 k, u32 ks, u32 bits, u32 book, u32 turned) {
    __shared__ dtr_smem sm;
    dtr_gen<4u>(x, w, suh, svh, out, m, n, k, ks, bits, book, turned, sm);
}

// Sums split-K partials in order and applies output rotation.
extern "C" __global__ __launch_bounds__(256, DTR_RED_BLK) void dtr_reduce(
    const float* part, const float* svh, trellis::y_t* out, u32 m, u32 n, u32 ks, u32 turned) {
    u32 ob = blockIdx.x;
    if (ob >= (n >> 7u)) return;
    u32 warp = threadIdx.x >> 5u;
    u32 lane = threadIdx.x & 31u;
    u32 base = (ob << 7u) + (lane << 2u);
    float sc[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (turned != 0u) trellis::load4(svh + base, sc);
    for (u32 r = warp; r < m; r += 8u) {
        const float* pb = part + u64(ob) * ks * m * 128u + u64(r) * 128u + (lane << 2u);
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        u32 p = 0u;
        for (; p + 4u <= ks; p += 4u) {
            float4 w0 = trellis::ld_cg_f4(pb + u64(p) * m * 128u);
            float4 w1 = trellis::ld_cg_f4(pb + u64(p + 1u) * m * 128u);
            float4 w2 = trellis::ld_cg_f4(pb + u64(p + 2u) * m * 128u);
            float4 w3 = trellis::ld_cg_f4(pb + u64(p + 3u) * m * 128u);
            v[0] += w0.x + w1.x + w2.x + w3.x; v[1] += w0.y + w1.y + w2.y + w3.y;
            v[2] += w0.z + w1.z + w2.z + w3.z; v[3] += w0.w + w1.w + w2.w + w3.w;
        }
        for (; p < ks; p++) {
            float4 w = trellis::ld_cg_f4(pb + u64(p) * m * 128u);
            v[0] += w.x; v[1] += w.y; v[2] += w.z; v[3] += w.w;
        }
        if (turned != 0u) {
            trellis::wht128(v, trellis::INV_SQRT128);
            v[0] *= sc[0]; v[1] *= sc[1]; v[2] *= sc[2]; v[3] *= sc[3];
        }
        trellis::store_y4(out + u64(r) * n + base, make_float4(v[0], v[1], v[2], v[3]));
    }
}

#endif
