#include <tessera/1/core>
#include <tessera/1/trellis>
using namespace tessera;
#if defined(TESSERA_CUDA)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
constexpr u32 DTR_PF_TILE = trellis::MX8_TILE;
union alignas(16) dtr_pf_smem {
    struct { u32 btab[2u][8u * DTR_PF_TILE]; u32 wtiles[2048]; } db;
    struct { u32 btab[16u * DTR_PF_TILE]; u32 wtiles[2048]; } s16;
    float out[64][128];
};
#else
constexpr u32 DTR_PF_TILE = 512u;
union alignas(16) dtr_pf_smem {
    struct { u32 btab[8u * DTR_PF_TILE]; u32 wtiles[2048]; } k;
    float out[64][128];
};
#endif
constexpr u32 DTR_PF_V4 = DTR_PF_TILE >> 2u;
static __device__ __forceinline__ void dtr_cp_async16(void* dst, const void* src) {
    u32 addr = static_cast<u32>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(addr), "l"(src));
}
// Packs rotated activations into [K/128][M/8][DTR_PF_TILE] B-fragment tiles.
extern "C" __global__ __launch_bounds__(256, 4) void dtr_turn_x(
    const float* x, const float* suh, u32* x_tiles, u32 m, u32 k, u32 turned) {
    __shared__ alignas(16) u32 sm_tile[DTR_PF_TILE];
    u32 warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u, ib = blockIdx.x, rt = blockIdx.y;
    u32 r = (rt << 3u) + warp;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (r < m) {
        u32 col = (ib << 7u) + (lane << 2u);
        trellis::load4(x + u64(r) * k + col, v);
        if (turned != 0u) {
            float sc[4];
            trellis::load4(suh + col, sc);
            v[0] *= sc[0]; v[1] *= sc[1]; v[2] *= sc[2]; v[3] *= sc[3];
            trellis::wht128(v, trellis::INV_SQRT128);
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    trellis::store_bfrag_mx8(sm_tile, warp, lane, v);
#else
    trellis::store_bfrag(sm_tile, warp, lane, v);
#endif
    prim::block_sync();
    u32 m_tiles = (m + 7u) >> 3u, tid = threadIdx.x;
    uint4* dst = reinterpret_cast<uint4*>(x_tiles + (u64(ib) * m_tiles + rt) * DTR_PF_TILE);
    if (tid < DTR_PF_V4) dst[tid] = reinterpret_cast<const uint4*>(sm_tile)[tid];
}
template <u32 K, u32 BOOK, u32 NT, bool TURNED>
static __device__ __forceinline__ void dtr_pf_body(
    const u32* x_tiles, const u8* w, const float* svh, float* out, u32 m, u32 n, u32 k,
    dtr_pf_smem& sm) {
    constexpr u32 TB = 32u * K;
    u32 ob = blockIdx.x, rt0 = blockIdx.y * NT, r0 = rt0 << 3u;
    if (ob >= (n >> 7u) || r0 >= m) return;
    u32 tid = threadIdx.x, lane = tid & 31u, j = tid >> 5u, blk_per = k >> 7u, m_tiles = (m + 7u) >> 3u;
    u32 valid_tiles = min(NT, m_tiles - rt0);
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    constexpr bool DB = (NT == 8u);
    u32 valid_v4 = valid_tiles * DTR_PF_V4;
    u32* wt = DB ? sm.db.wtiles : sm.s16.wtiles;
    auto btab_of = [&](u32 st) -> u32* { return DB ? sm.db.btab[st] : sm.s16.btab; };
    auto issue_x = [&](u32 st, u32 c0) {
        const uint4* xs = reinterpret_cast<const uint4*>(x_tiles + (u64(c0) * m_tiles + rt0) * DTR_PF_TILE);
        uint4* xd = reinterpret_cast<uint4*>(btab_of(st));
#pragma unroll
        for (u32 idx = tid; idx < NT * DTR_PF_V4; idx += 256u) if (idx < valid_v4) dtr_cp_async16(xd + idx, xs + idx);
        asm volatile("cp.async.commit_group;");
    };
    if constexpr (DB) issue_x(0u, 0u);
    for (u32 c0 = 0u; c0 < blk_per; c0++) {
        u32 st = DB ? (c0 & 1u) : 0u;
        if (c0 > 0u) prim::block_sync();
        if constexpr (!DB) issue_x(0u, c0);
        trellis::blk_store<K>(nw, tid, wt);
        asm volatile("cp.async.wait_group 0;");
        prim::block_sync();
        if (c0 + 1u < blk_per) { if constexpr (DB) issue_x(st ^ 1u, c0 + 1u); gb += sG; trellis::blk_load<K>(gb, tid, aligned, nw); }
        trellis::blk_mma_mx8<K, BOOK, NT>(wt, btab_of(st), j, lane, wm, acc);
    }
#else
    for (u32 c0 = 0u; c0 < blk_per; c0++) {
        if (c0 > 0u) prim::block_sync();
        trellis::blk_store<K>(nw, tid, sm.k.wtiles);
        for (u32 bt = 0u; bt < valid_tiles; bt += 8u) {
            u32 ct = min(8u, valid_tiles - bt);
            const uint4* xs = reinterpret_cast<const uint4*>(x_tiles + (u64(c0) * m_tiles + rt0 + bt) * DTR_PF_TILE);
            for (u32 idx = tid; idx < ct * DTR_PF_V4; idx += 256u) reinterpret_cast<uint4*>(sm.k.btab)[idx] = xs[idx];
            prim::block_sync();
            if (bt == 0u && c0 + 1u < blk_per) { gb += sG; trellis::blk_load<K>(gb, tid, aligned, nw); }
            trellis::blk_mma<K, BOOK, 8u>(sm.k.wtiles, sm.k.btab + (lane << 1u), DTR_PF_TILE, j, lane, wm, &acc[bt]);
            if (bt + 8u < valid_tiles) prim::block_sync();
        }
    }
#endif
    if constexpr (!TURNED) {
        trellis::store_plain_acc<NT>(acc, out, m, n, r0, ob, j, lane);
    } else {
        u32 col = (j << 4u) + (lane >> 2u), c = lane & 3u, base = (ob << 7u) + (lane << 2u);
        float sc[4];
        trellis::load4(svh + base, sc);
        for (u32 half = 0u; half < NT / 8u; half++) {
            u32 hr0 = r0 + half * 64u;
            if (hr0 >= m) break;
            u32 nr = min(64u, m - hr0);
            prim::block_sync();
#pragma unroll
            for (u32 ti = 0u; ti < 8u; ti++) {
                u32 lr = (ti << 3u) + (c << 1u), idx = half * 8u + ti;
                if (lr < nr) { sm.out[lr][col] = acc[idx][0]; sm.out[lr][col + 8u] = acc[idx][2]; }
                if (lr + 1u < nr) { sm.out[lr + 1u][col] = acc[idx][1]; sm.out[lr + 1u][col + 8u] = acc[idx][3]; }
            }
            prim::block_sync();
            for (u32 r = j; r < nr; r += 8u) {
                float v[4];
                trellis::load4(&sm.out[r][lane << 2u], v);
                trellis::wht128(v, trellis::INV_SQRT128);
                *reinterpret_cast<float4*>(out + u64(hr0 + r) * n + base) = make_float4(v[0] * sc[0], v[1] * sc[1], v[2] * sc[2], v[3] * sc[3]);
            }
        }
    }
}
// Prefill dense trellis K4 MCG matmul, 64 rows per block.
extern "C" __global__ __launch_bounds__(256, 2) void dtr_k4_t8_mx(
    const u32* xt, const u8* w, const float* svh, float* out, u32 m, u32 n, u32 k, u32, u32, u32) {
    __shared__ dtr_pf_smem sm;
    dtr_pf_body<4u, trellis::BOOK_MCG, 8u, true>(xt, w, svh, out, m, n, k, sm);
}
// Prefill dense trellis K4 MCG matmul, 128 rows per block.
extern "C" __global__ __launch_bounds__(256, 2) void dtr_k4_t16_mx(
    const u32* xt, const u8* w, const float* svh, float* out, u32 m, u32 n, u32 k, u32, u32, u32) {
    __shared__ dtr_pf_smem sm;
    dtr_pf_body<4u, trellis::BOOK_MCG, 16u, true>(xt, w, svh, out, m, n, k, sm);
}
template <u32 K, u32 BOOK>
static __device__ __noinline__ void dtr_pf_gk(const u32* xt, const u8* w, const float* svh, float* out, u32 m, u32 n, u32 k, u32 t, dtr_pf_smem& sm) {
    if (t != 0u) dtr_pf_body<K, BOOK, 8u, true>(xt, w, svh, out, m, n, k, sm);
    else dtr_pf_body<K, BOOK, 8u, false>(xt, w, svh, out, m, n, k, sm);
}
template <u32 BOOK>
static __device__ void dtr_pf_gb(const u32* xt, const u8* w, const float* svh, float* out, u32 m, u32 n, u32 k, u32 bits, u32 t, dtr_pf_smem& sm) {
    if (bits == 2u) dtr_pf_gk<2u, BOOK>(xt, w, svh, out, m, n, k, t, sm);
    else if (bits == 3u) dtr_pf_gk<3u, BOOK>(xt, w, svh, out, m, n, k, t, sm);
    else dtr_pf_gk<4u, BOOK>(xt, w, svh, out, m, n, k, t, sm);
}
// Generic prefill dense trellis matmul, 64 rows per block.
extern "C" __global__ __launch_bounds__(256, 2) void dtr_gen8_mx(
    const u32* xt, const u8* w, const float* svh, float* out, u32 m, u32 n, u32 k, u32 bits, u32 book, u32 t) {
    __shared__ dtr_pf_smem sm;
    if (book == trellis::BOOK_PMCGX) dtr_pf_gb<trellis::BOOK_PMCGX>(xt, w, svh, out, m, n, k, bits, t, sm);
    else if (book == trellis::BOOK_MUL1) dtr_pf_gb<trellis::BOOK_MUL1>(xt, w, svh, out, m, n, k, bits, t, sm);
    else dtr_pf_gb<trellis::BOOK_MCG>(xt, w, svh, out, m, n, k, bits, t, sm);
}
#endif
