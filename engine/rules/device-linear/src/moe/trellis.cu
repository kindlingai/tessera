#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

#if defined(TESSERA_CUDA)

#ifndef PTR_CHUNK
#define PTR_CHUNK 4u
#endif
#ifndef PTR_K3_MIN_BLOCKS
#define PTR_K3_MIN_BLOCKS 4
#endif
#ifndef PTR_GEN_MIN_BLOCKS
#define PTR_GEN_MIN_BLOCKS 2
#endif

using ptr_smem = trellis::stage_smem<float[128]>;

template <u32 K, u32 BOOK, bool TURNED>
static __device__ __forceinline__ void ptr_body(
    const trellis::x_t* x, const u8* w, const u32* ids, const float* su, const float* sv,
    const float* extra, float* out, u32 rows, u32 n, u32 k, u32 count, u32 repeat, u32 first,
    u32 post, ptr_smem& sm) {
    constexpr u32 TB = 32u * K;
    constexpr u32 CHUNK = PTR_CHUNK;
    u32 ob = blockIdx.x;
    u32 r = blockIdx.y;
    if (ob >= (n >> 7u) || r >= rows) return;
    u32 tid = threadIdx.x;
    u32 raw = ids[r];
    if (raw < first || raw - first >= count) {
        if (tid < 128u) out[u64(r) * n + (ob << 7u) + tid] = 0.0f;
        return;
    }
    u32 e = raw - first;
    if (tid == 0u && ob == 0u) trellis::note_hit(w, e);
    u32 lane = tid & 31u;
    u32 j = tid >> 5u;
    u32 blk_per = k >> 7u;
    bool aligned = (reinterpret_cast<u64>(w + trellis::HEAD) & 15ull) == 0ull;
    bool input_side =
        (*reinterpret_cast<const u32*>(w + 8u) & trellis::SIDE_MASK) == trellis::INPUT_SIDE;
    u64 sG = (input_side ? u64(n >> 1u) : 64ull) * u64(TB);
    const u8* gb = trellis::tile_ptr<K>(w, k, n, e, 0u, ob, input_side);
    u32 nw[8];
    trellis::blk_load<K>(gb, tid, aligned, nw);
    trellis::word_map wm = trellis::map_word<K>(lane);
    float acc[1][4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    const trellis::x_t* xr = x + u64(r / repeat) * k;
    const float* sur = TURNED ? (su + u64(e) * k) : nullptr;
    for (u32 c0 = 0u; c0 < blk_per; c0 += CHUNK) {
        u32 nb = min(CHUNK, blk_per - c0);
        if (c0 > 0u) prim::block_sync();
        if (j < nb) {
            u32 col = ((c0 + j) << 7u) + (lane << 2u);
            float v[4];
            trellis::load_x4(xr + col, v);
            if constexpr (TURNED) {
                float sc[4];
                trellis::load4(sur + col, sc);
                v[0] *= sc[0];
                v[1] *= sc[1];
                v[2] *= sc[2];
                v[3] *= sc[3];
                trellis::wht128(v, trellis::INV_SQRT128);
            }
            trellis::store_bfrag(sm.k.btab + j * 512u, 0u, lane, v);
        }
        trellis::blk_stream<K, BOOK, 1u>(gb, sG, c0, nb, blk_per, tid, j, lane, aligned, wm, nw,
                                         sm.k.wtiles, sm.k.btab + (lane << 1u), nb * 512u, acc);
    }
    u32 col = (j << 4u) + (lane >> 2u);
    u32 c = lane & 3u;
    if constexpr (!TURNED) {
        if (c == 0u) {
            u64 base = u64(r) * n + (ob << 7u) + col;
            out[base] = acc[0][0];
            out[base + 8u] = acc[0][2];
        }
    } else {
        prim::block_sync();
        if (c == 0u) {
            sm.out[col] = acc[0][0];
            sm.out[col + 8u] = acc[0][2];
        }
        prim::block_sync();
        if (j == 0u) {
            u32 base = (ob << 7u) + (lane << 2u);
            float v[4], sc[4];
            trellis::load4(sm.out + (lane << 2u), v);
            trellis::wht128(v, trellis::INV_SQRT128);
            trellis::load4(sv + u64(e) * n + base, sc);
            v[0] *= sc[0];
            v[1] *= sc[1];
            v[2] *= sc[2];
            v[3] *= sc[3];
            if (post == 1u) {
                float ex[4];
                trellis::load4(extra + base, ex);
                v[0] *= ex[0];
                v[1] *= ex[1];
                v[2] *= ex[2];
                v[3] *= ex[3];
            } else if (post == 2u) {
                float ex[4];
                trellis::load4(extra + u64(r) * n + base, ex);
                v[0] = (v[0] / (1.0f + expf(-v[0]))) * ex[0];
                v[1] = (v[1] / (1.0f + expf(-v[1]))) * ex[1];
                v[2] = (v[2] / (1.0f + expf(-v[2]))) * ex[2];
                v[3] = (v[3] / (1.0f + expf(-v[3]))) * ex[3];
            }
            *reinterpret_cast<float4*>(out + u64(r) * n + base) =
                make_float4(v[0], v[1], v[2], v[3]);
        }
    }
}

template <u32 K, u32 BOOK>
static __device__ __noinline__ void ptr_run_k(
    const trellis::x_t* x, const u8* w, const u32* ids, const float* su, const float* sv,
    const float* extra, float* out, u32 rows, u32 n, u32 k, u32 count, u32 repeat, u32 first,
    u32 turned, u32 post, ptr_smem& sm) {
    if (turned != 0u) {
        ptr_body<K, BOOK, true>(x, w, ids, su, sv, extra, out, rows, n, k, count, repeat, first,
                                post, sm);
    } else {
        ptr_body<K, BOOK, false>(x, w, ids, su, sv, extra, out, rows, n, k, count, repeat, first,
                                 0u, sm);
    }
}

template <u32 BOOK>
static __device__ void ptr_run_book(
    const trellis::x_t* x, const u8* w, const u32* ids, const float* su, const float* sv,
    const float* extra, float* out, u32 rows, u32 n, u32 k, u32 count, u32 repeat, u32 first,
    u32 bits, u32 turned, u32 post, ptr_smem& sm) {
    if (bits == 2u) {
        ptr_run_k<2u, BOOK>(x, w, ids, su, sv, extra, out, rows, n, k, count, repeat, first,
                            turned, post, sm);
    } else if (bits == 3u) {
        ptr_run_k<3u, BOOK>(x, w, ids, su, sv, extra, out, rows, n, k, count, repeat, first,
                            turned, post, sm);
    } else {
        ptr_run_k<4u, BOOK>(x, w, ids, su, sv, extra, out, rows, n, k, count, repeat, first,
                            turned, post, sm);
    }
}

// Fast specialized PMCGX K3 rotated picked expert matmul.
extern "C" __global__ __launch_bounds__(256, PTR_K3_MIN_BLOCKS) void ptr_k3_pmcgx_t(
    const trellis::x_t* x, const u8* w, const u32* ids, const float* su, const float* sv,
    const float* extra, float* out, u32 rows, u32 n, u32 k, u32 count, u32 repeat, u32 first,
    u32 bits, u32 book, u32 turned, u32 post) {
    (void)bits;
    (void)book;
    (void)turned;
    __shared__ ptr_smem sm;
    ptr_body<3u, trellis::BOOK_PMCGX, true>(x, w, ids, su, sv, extra, out, rows, n, k, count,
                                            repeat, first, post, sm);
}

// Generic picked trellis expert matmul (K2/K3/K4, MCG/MUL1/PMCGX).
extern "C" __global__ __launch_bounds__(256, PTR_GEN_MIN_BLOCKS) void ptr_gen(
    const trellis::x_t* x, const u8* w, const u32* ids, const float* su, const float* sv,
    const float* extra, float* out, u32 rows, u32 n, u32 k, u32 count, u32 repeat, u32 first,
    u32 bits, u32 book, u32 turned, u32 post) {
    __shared__ ptr_smem sm;
    if (book == trellis::BOOK_PMCGX) {
        ptr_run_book<trellis::BOOK_PMCGX>(x, w, ids, su, sv, extra, out, rows, n, k, count, repeat,
                                          first, bits, turned, post, sm);
    } else if (book == trellis::BOOK_MUL1) {
        ptr_run_book<trellis::BOOK_MUL1>(x, w, ids, su, sv, extra, out, rows, n, k, count, repeat,
                                         first, bits, turned, post, sm);
    } else {
        ptr_run_book<trellis::BOOK_MCG>(x, w, ids, su, sv, extra, out, rows, n, k, count, repeat,
                                        first, bits, turned, post, sm);
    }
}

#endif
