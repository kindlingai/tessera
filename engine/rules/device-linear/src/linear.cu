#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

#define TM_WARPS 8u

#if defined(TESSERA_CUDA)

using dtr_smem = trellis::stage_smem<float[32][128]>;

template <u32 K, u32 BOOK, u32 NT, bool TURNED, bool SPLIT>
static __device__ __forceinline__ void dtr_body(const float* x, const u8* w, const float* suh,
                                                const float* svh, float* out, u32 m, u32 n, u32 k,
                                                u32 ks, dtr_smem& sm) {
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
    bool input_side = (*reinterpret_cast<const u32*>(w + 8u) & trellis::SIDE_MASK) == trellis::INPUT_SIDE;
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
                trellis::load4(x + u64(r0 + r) * k + col, v);
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
        float* pp = out + (u64(ob) * ks + p) * m * 128u;
#pragma unroll
        for (u32 ti = 0u; ti < NT; ti++) {
            u32 r = r0 + (ti << 3u) + (c << 1u);
            if (r < m) { pp[u64(r) * 128u + col] = acc[ti][0]; pp[u64(r) * 128u + col + 8u] = acc[ti][2]; }
            if (r + 1u < m) { pp[u64(r + 1u) * 128u + col] = acc[ti][1]; pp[u64(r + 1u) * 128u + col + 8u] = acc[ti][3]; }
        }
    } else if constexpr (!TURNED) {
        u32 ob0 = ob << 7u;
#pragma unroll
        for (u32 ti = 0u; ti < NT; ti++) {
            u32 r = r0 + (ti << 3u) + (c << 1u);
            if (r < m) { out[u64(r) * n + ob0 + col] = acc[ti][0]; out[u64(r) * n + ob0 + col + 8u] = acc[ti][2]; }
            if (r + 1u < m) { out[u64(r + 1u) * n + ob0 + col] = acc[ti][1]; out[u64(r + 1u) * n + ob0 + col + 8u] = acc[ti][3]; }
        }
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
            *reinterpret_cast<float4*>(out + u64(r0 + r) * n + base) =
                make_float4(v[0] * sc[0], v[1] * sc[1], v[2] * sc[2], v[3] * sc[3]);
        }
    }
}

template <u32 K, u32 BOOK, u32 NT>
static __device__ __noinline__ void dtr_gen_k(const float* x, const u8* w, const float* suh,
                                              const float* svh, float* out, u32 m, u32 n, u32 k,
                                              u32 ks, u32 turned, dtr_smem& sm) {
    if (ks > 1u) {
        if (turned != 0u) dtr_body<K, BOOK, NT, true, true>(x, w, suh, svh, out, m, n, k, ks, sm);
        else dtr_body<K, BOOK, NT, false, true>(x, w, suh, svh, out, m, n, k, ks, sm);
    } else {
        if (turned != 0u) dtr_body<K, BOOK, NT, true, false>(x, w, suh, svh, out, m, n, k, 1u, sm);
        else dtr_body<K, BOOK, NT, false, false>(x, w, suh, svh, out, m, n, k, 1u, sm);
    }
}

template <u32 BOOK, u32 NT>
static __device__ void dtr_gen_book(const float* x, const u8* w, const float* suh, const float* svh,
                                    float* out, u32 m, u32 n, u32 k, u32 ks, u32 bits, u32 turned,
                                    dtr_smem& sm) {
    if (bits == 2u) dtr_gen_k<2u, BOOK, NT>(x, w, suh, svh, out, m, n, k, ks, turned, sm);
    else if (bits == 3u) dtr_gen_k<3u, BOOK, NT>(x, w, suh, svh, out, m, n, k, ks, turned, sm);
    else dtr_gen_k<4u, BOOK, NT>(x, w, suh, svh, out, m, n, k, ks, turned, sm);
}

template <u32 NT>
static __device__ void dtr_gen(const float* x, const u8* w, const float* suh, const float* svh,
                               float* out, u32 m, u32 n, u32 k, u32 ks, u32 bits, u32 book,
                               u32 turned, dtr_smem& sm) {
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
        const float* x, const u8* w, const float* suh, const float* svh, float* out, u32 m,   \
        u32 n, u32 k, u32 ks, u32 bits, u32 book, u32 turned) {                               \
        (void)bits;                                                                           \
        (void)book;                                                                           \
        (void)turned;                                                                         \
        __shared__ dtr_smem sm;                                                               \
        dtr_body<K, BOOK, NT, TURNED, SPLIT>(x, w, suh, svh, out, m, n, k, ks, sm);           \
    }

DTR_DEF(dtr_k4_t1s, 4u, trellis::BOOK_MCG, 1u, true, true, 4)
DTR_DEF(dtr_k4_t1, 4u, trellis::BOOK_MCG, 1u, true, false, 4)
DTR_DEF(dtr_k4_p1s, 4u, trellis::BOOK_MCG, 1u, false, true, 4)
DTR_DEF(dtr_k4_p1, 4u, trellis::BOOK_MCG, 1u, false, false, 4)
DTR_DEF(dtr_k4_t2s, 4u, trellis::BOOK_MCG, 2u, true, true, 2)
DTR_DEF(dtr_k4_t2, 4u, trellis::BOOK_MCG, 2u, true, false, 2)
DTR_DEF(dtr_k4_t4, 4u, trellis::BOOK_MCG, 4u, true, false, 2)

// Generic dense trellis matmul for up to 8 rows.
extern "C" __global__ __launch_bounds__(256, 2) void dtr_gen1(
    const float* x, const u8* w, const float* suh, const float* svh, float* out, u32 m, u32 n,
    u32 k, u32 ks, u32 bits, u32 book, u32 turned) {
    __shared__ dtr_smem sm;
    dtr_gen<1u>(x, w, suh, svh, out, m, n, k, ks, bits, book, turned, sm);
}

// Generic dense trellis matmul for 16-row tiles.
extern "C" __global__ __launch_bounds__(256, 2) void dtr_gen2(
    const float* x, const u8* w, const float* suh, const float* svh, float* out, u32 m, u32 n,
    u32 k, u32 ks, u32 bits, u32 book, u32 turned) {
    __shared__ dtr_smem sm;
    dtr_gen<2u>(x, w, suh, svh, out, m, n, k, ks, bits, book, turned, sm);
}

// Generic dense trellis matmul for 32-row tiles.
extern "C" __global__ __launch_bounds__(256, 2) void dtr_gen4(
    const float* x, const u8* w, const float* suh, const float* svh, float* out, u32 m, u32 n,
    u32 k, u32 ks, u32 bits, u32 book, u32 turned) {
    __shared__ dtr_smem sm;
    dtr_gen<4u>(x, w, suh, svh, out, m, n, k, ks, bits, book, turned, sm);
}

// Sums split-K partials in order and applies output rotation.
extern "C" __global__ __launch_bounds__(256, 4) void dtr_reduce(
    const float* part, const float* svh, float* out, u32 m, u32 n, u32 ks, u32 turned) {
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
        *reinterpret_cast<float4*>(out + u64(r) * n + base) = make_float4(v[0], v[1], v[2], v[3]);
    }
}

static __device__ __forceinline__ float fg_warp_sum(float v) {
#pragma unroll
    for (u32 off = 16u; off > 0u; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    return v;
}

static __device__ __forceinline__ uint4 fg_ld_nc4(const void* p) {
    u32 d[4];
    trellis::ld_nc_v4(static_cast<const u8*>(p), d);
    return make_uint4(d[0], d[1], d[2], d[3]);
}

static __device__ __forceinline__ u32 fg_unaligned_u32(const u32* s, u32 byte_off) {
    u32 w = byte_off >> 2;
    u32 sh = (byte_off & 3u) << 3;
    if (sh == 0u) return s[w];
    u32 out;
    asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(s[w]), "r"(s[w + 1u]), "r"(sh));
    return out;
}

static __device__ __forceinline__ float fg_dot4(u32 h0, u32 h1, uint2 xh) {
    u32 p;
    asm("mul.rn.f16x2 %0, %1, %2; fma.rn.f16x2 %0, %3, %4, %0;"
        : "=r"(p)
        : "r"(h0), "r"(xh.x), "r"(h1), "r"(xh.y));
    return trellis::f16_to_f32(p & 0xFFFFu) + trellis::f16_to_f32(p >> 16);
}

static __device__ __forceinline__ void fg_stage_chunk(u32 x_h[512], const float* x, u32 base) {
    float4 xv = *reinterpret_cast<const float4*>(x + base + (threadIdx.x << 2));
    u32 p0 = trellis::f32x2_to_f16x2(xv.x, xv.y);
    u32 p1 = trellis::f32x2_to_f16x2(xv.z, xv.w);
    *reinterpret_cast<uint2*>(x_h + (threadIdx.x << 1)) = make_uint2(p0, p1);
    prim::block_sync();
}

static __device__ __forceinline__ float fg_dot8_step(float acc, const float* x, const u16* wr,
                                                     u32 i) {
    uint4 h = fg_ld_nc4(wr + i);
    float4 x0 = *reinterpret_cast<const float4*>(x + i);
    float4 x1 = *reinterpret_cast<const float4*>(x + i + 4u);
    acc = fmaf(__uint_as_float(h.x << 16), x0.x,
               fmaf(__uint_as_float(h.x & 0xFFFF0000u), x0.y,
                    fmaf(__uint_as_float(h.y << 16), x0.z,
                         fmaf(__uint_as_float(h.y & 0xFFFF0000u), x0.w, acc))));
    return fmaf(__uint_as_float(h.z << 16), x1.x,
                fmaf(__uint_as_float(h.z & 0xFFFF0000u), x1.y,
                     fmaf(__uint_as_float(h.w << 16), x1.z,
                          fmaf(__uint_as_float(h.w & 0xFFFF0000u), x1.w, acc))));
}

static __device__ __forceinline__ float fg_dot8_bf16(const float* x, const u16* wr, u32 k,
                                                     u32 lane) {
    float acc = 0.0f;
    for (u32 i = lane << 3; i < k; i += 256u) acc = fg_dot8_step(acc, x, wr, i);
    return fg_warp_sum(acc);
}

static __device__ __forceinline__ u32 fg_bf16x2(float lo, float hi) {
    u32 out;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(lo), "f"(hi));
    return out;
}

static __device__ __forceinline__ void fg_mma_bf16(float d[4], const u32 a[4], const u32 b[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

static __device__ __forceinline__ float fg_silu(float v) {
    return v / (1.0f + expf(-v));
}

static __device__ __forceinline__ float fg_epilogue(float v, const float* other, u64 off, u32 col,
                                                    u32 mode) {
    if (mode == 1u) return v + other[off];
    if (mode == 2u) return v + other[col];
    if (mode == 3u) return fg_silu(v) * other[off];
    if (mode == 4u) return v * other[off];
    if (mode == 5u) return fg_silu(v);
    return v;
}

static __device__ __forceinline__ float2 fg_epilogue2(float v0, float v1, const float* other,
                                                      u64 off, u32 col, u32 mode) {
    if (mode == 0u) return make_float2(v0, v1);
    if (mode == 5u) return make_float2(fg_silu(v0), fg_silu(v1));
    float2 b = *reinterpret_cast<const float2*>(other + (mode == 2u ? u64(col) : off));
    if (mode == 1u || mode == 2u) return make_float2(v0 + b.x, v1 + b.y);
    if (mode == 3u) return make_float2(fg_silu(v0) * b.x, fg_silu(v1) * b.y);
    return make_float2(v0 * b.x, v1 * b.y);
}

static __device__ __forceinline__ u16 fg_u16_at(const u32* s, u32 byte_off) {
    u32 w = byte_off >> 2;
    u32 sh = (byte_off & 3u) << 3;
    if (sh <= 16u) return static_cast<u16>(s[w] >> sh);
    u32 out;
    asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(s[w]), "r"(s[w + 1u]), "r"(sh));
    return static_cast<u16>(out);
}

static __device__ __forceinline__ u32 fg_e4m3_f16(u16 q) {
    u32 h;
    asm("{.reg .b16 t; mov.b16 t, %1; cvt.rn.f16x2.e4m3x2 %0, t;}" : "=r"(h) : "h"(q));
    return h;
}

template <u32 KS>
struct alignas(16) fg_bf16_smem {
    u32 sxh[KS][16][36];
    u32 sxl[KS][16][36];
    union {
        u32 sw[4][8][36];
        float4 red[2][32];
    } u;
};

template <u32 KS, bool SPLIT>
static __device__ __forceinline__ void fg_bf16_mm(const float* x, const u16* w, const float* other,
                                                  float* out, u32 m, u32 n, u32 k, u32 w_stride,
                                                  u32 x_rs, u32 o_rs, u32 r0, u32 mode, u32 ks, fg_bf16_smem<KS>& sm) {
    constexpr u32 NTILES = 4u / KS;
    constexpr u32 BN = 8u * NTILES;
    constexpr u32 BK = 64u * KS;
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, g = lane >> 2, t = lane & 3u;
    u32 wt = warp % NTILES, kp = warp / NTILES, nb = n / BN;
    u32 ob = SPLIT ? (blockIdx.x % nb) : blockIdx.x, p = SPLIT ? (blockIdx.x / nb) : 0u;
    u32 n0 = ob * BN + (wt << 3), chunks = k / BK, ch_per = SPLIT ? (chunks / ks) : chunks;
    u32 k_begin = SPLIT ? (p * ch_per * BK) : 0u, k_end = SPLIT ? (k_begin + ch_per * BK) : k;
    const u16* wr = w + u64(n0 + g) * w_stride + (kp << 6);
    u32 xr = threadIdx.x >> 4, xc = (threadIdx.x & 15u) << 2, xw = (threadIdx.x & 15u) << 1;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint4 cur0 = fg_ld_nc4(wr + k_begin + (t << 3));
    uint4 cur1 = fg_ld_nc4(wr + k_begin + 32u + (t << 3));
    for (u32 k0 = k_begin; k0 < k_end; k0 += BK) {
#pragma unroll
        for (u32 s = 0u; s < KS; s++) {
#pragma unroll
            for (u32 dr = 0u; dr < 16u; dr += 8u) {
                u32 r = xr + dr;
                if (r0 + r < m) {
                    const float* xp = x + u64(r0 + r) * x_rs + k0 + (s << 6) + xc;
                    float4 xv = *reinterpret_cast<const float4*>(xp);
                    u32 h0 = fg_bf16x2(xv.x, xv.y);
                    u32 h1 = fg_bf16x2(xv.z, xv.w);
                    u32 l0 = fg_bf16x2(xv.x - __uint_as_float(h0 << 16),
                                       xv.y - __uint_as_float(h0 & 0xFFFF0000u));
                    u32 l1 = fg_bf16x2(xv.z - __uint_as_float(h1 << 16),
                                       xv.w - __uint_as_float(h1 & 0xFFFF0000u));
                    *reinterpret_cast<uint2*>(&sm.sxh[s][r][xw]) = make_uint2(h0, h1);
                    *reinterpret_cast<uint2*>(&sm.sxl[s][r][xw]) = make_uint2(l0, l1);
                } else {
                    *reinterpret_cast<uint2*>(&sm.sxh[s][r][xw]) = make_uint2(0u, 0u);
                    *reinterpret_cast<uint2*>(&sm.sxl[s][r][xw]) = make_uint2(0u, 0u);
                }
            }
        }
        *reinterpret_cast<uint4*>(&sm.u.sw[warp][g][t << 2]) = cur0;
        *reinterpret_cast<uint4*>(&sm.u.sw[warp][g][16u + (t << 2)]) = cur1;
        prim::block_sync();
        if (k0 + BK < k_end) {
            cur0 = fg_ld_nc4(wr + k0 + BK + (t << 3));
            cur1 = fg_ld_nc4(wr + k0 + BK + 32u + (t << 3));
        }
#pragma unroll
        for (u32 ks_i = 0u; ks_i < 4u; ks_i++) {
            u32 off = (ks_i << 3) + t;
            u32 ah[4] = {sm.sxh[kp][g][off], sm.sxh[kp][g + 8u][off], sm.sxh[kp][g][off + 4u],
                         sm.sxh[kp][g + 8u][off + 4u]};
            u32 al[4] = {sm.sxl[kp][g][off], sm.sxl[kp][g + 8u][off], sm.sxl[kp][g][off + 4u],
                         sm.sxl[kp][g + 8u][off + 4u]};
            u32 b[2] = {sm.u.sw[warp][g][off], sm.u.sw[warp][g][off + 4u]};
            fg_mma_bf16(acc, ah, b);
            fg_mma_bf16(acc, al, b);
        }
        prim::block_sync();
    }
    if constexpr (KS == 2u) {
        if (kp == 1u) sm.u.red[wt][lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        prim::block_sync();
        if (kp != 0u) return;
        float4 pr = sm.u.red[wt][lane];
        acc[0] += pr.x; acc[1] += pr.y; acc[2] += pr.z; acc[3] += pr.w;
    }
    u32 col = n0 + (t << 1);
    if constexpr (SPLIT) {
        if (r0 + g < m) *reinterpret_cast<float2*>(out + (u64(p) * m + r0 + g) * n + col) = make_float2(acc[0], acc[1]);
        if (r0 + g + 8u < m) *reinterpret_cast<float2*>(out + (u64(p) * m + r0 + g + 8u) * n + col) = make_float2(acc[2], acc[3]);
    } else {
        if (r0 + g < m) {
            u64 off = u64(r0 + g) * o_rs + col;
            *reinterpret_cast<float2*>(out + off) = fg_epilogue2(acc[0], acc[1], other, off, col, mode);
        }
        if (r0 + g + 8u < m) {
            u64 off = u64(r0 + g + 8u) * o_rs + col;
            *reinterpret_cast<float2*>(out + off) = fg_epilogue2(acc[2], acc[3], other, off, col, mode);
        }
    }
}

// Streaming BF16 Tensor-Core small-batch matmul, 16x32 per block.
extern "C" __global__ __launch_bounds__(128, 6) void tessera_mm_bf16(
    const float* x, const u16* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode) {
    __shared__ fg_bf16_smem<1u> sm;
    fg_bf16_mm<1u, false>(x, w, other, out, m, n, k, k, k, n, blockIdx.y << 4, mode, 1u, sm);
}

// Split-K BF16 Tensor-Core small-batch matmul, 16x16 per block.
extern "C" __global__ __launch_bounds__(128, 4) void tessera_mm_bf16_ks2(
    const float* x, const u16* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode) {
    __shared__ fg_bf16_smem<2u> sm;
    fg_bf16_mm<2u, false>(x, w, other, out, m, n, k, k, k, n, blockIdx.y << 4, mode, 1u, sm);
}

// Cross-CTA split-K BF16 Tensor-Core matmul, 16x16 per block.
extern "C" __global__ __launch_bounds__(128, 4) void tessera_mm_bf16_sk(
    const float* x, const u16* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode,
    u32 ks) {
    __shared__ fg_bf16_smem<2u> sm;
    fg_bf16_mm<2u, true>(x, w, other, out, m, n, k, k, k, n, blockIdx.y << 4, mode, ks, sm);
}

union alignas(16) fg_mx_smem {
    u32 sw[4][16][36];
    float red[4][32][17];
};

template <u32 NT, bool SPLIT>
static __device__ __forceinline__ void fg_mxfp8_mm(const float* x, const u8* w, const float* other,
                                                   float* out, u32 m, u32 n, u32 k, u32 mode,
                                                   u32 ks, fg_mx_smem& sm) {
    u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, g = lane >> 2, t = lane & 3u;
    u32 r0 = blockIdx.y * (NT << 3), nb = n >> 4;
    u32 ob = SPLIT ? (blockIdx.x % nb) : blockIdx.x, p = SPLIT ? (blockIdx.x / nb) : 0u;
    u32 n0 = ob << 4, chunks = k >> 9, ch_per = SPLIT ? (chunks / ks) : chunks;
    u64 row_bytes = u64(k >> 5) * 33ull;
    u32 ch0 = SPLIT ? (p * ch_per) : 0u, ch1 = ch0 + ch_per;
    u32 w_base = warp << 7, shift = warp << 2, x_off = warp << 7;
    u32 lr = lane >> 3, lc = (lane & 7u) << 4;
    const u8* w0 = w + u64(n0) * row_bytes + w_base;
    const u8* wp0 = w0 + u64(lr) * row_bytes + lc;
    const u8 *wp1 = wp0 + 4ull * row_bytes, *wp2 = wp0 + 8ull * row_bytes, *wp3 = wp0 + 12ull * row_bytes;
    const u8* wp4 = w0 + u64(lane & 15u) * row_bytes + 128u;
    float acc[NT][4];
#pragma unroll
    for (u32 ti = 0u; ti < NT; ti++) acc[ti][0] = acc[ti][1] = acc[ti][2] = acc[ti][3] = 0.0f;
    u64 off0 = u64(ch0) * 528ull;
    uint4 cur0 = fg_ld_nc4(wp0 + off0), cur1 = fg_ld_nc4(wp1 + off0);
    uint4 cur2 = fg_ld_nc4(wp2 + off0), cur3 = fg_ld_nc4(wp3 + off0), cur4;
    if (lane < 16u) cur4 = fg_ld_nc4(wp4 + off0);
    const u32* sw0 = sm.sw[warp][g];
    const u32* sw1 = sm.sw[warp][g + 8u];
    const u8* sb0 = reinterpret_cast<const u8*>(sw0);
    const u8* sb1 = reinterpret_cast<const u8*>(sw1);
    for (u32 ch = ch0; ch < ch1; ch++) {
        __syncwarp();
        *reinterpret_cast<uint4*>(&sm.sw[warp][lr][lc >> 2]) = cur0;
        *reinterpret_cast<uint4*>(&sm.sw[warp][lr + 4u][lc >> 2]) = cur1;
        *reinterpret_cast<uint4*>(&sm.sw[warp][lr + 8u][lc >> 2]) = cur2;
        *reinterpret_cast<uint4*>(&sm.sw[warp][lr + 12u][lc >> 2]) = cur3;
        if (lane < 16u) *reinterpret_cast<uint4*>(&sm.sw[warp][lane][32u]) = cur4;
        __syncwarp();
        if (ch + 1u < ch1) {
            u64 next_off = u64(ch + 1u) * 528ull;
            cur0 = fg_ld_nc4(wp0 + next_off); cur1 = fg_ld_nc4(wp1 + next_off);
            cur2 = fg_ld_nc4(wp2 + next_off); cur3 = fg_ld_nc4(wp3 + next_off);
            if (lane < 16u) cur4 = fg_ld_nc4(wp4 + next_off);
        }
#pragma unroll
        for (u32 blk = 0u; blk < 4u; blk++) {
            u32 b_off = shift + blk * 33u;
            u32 e0 = sb0[b_off], e1 = sb1[b_off];
            float sc0 = __uint_as_float(e0 == 0u ? 0x00400000u : (e0 << 23));
            float sc1 = __uint_as_float(e1 == 0u ? 0x00400000u : (e1 << 23));
            float tmp[NT][4];
#pragma unroll
            for (u32 ti = 0u; ti < NT; ti++) tmp[ti][0] = tmp[ti][1] = tmp[ti][2] = tmp[ti][3] = 0.0f;
#pragma unroll
            for (u32 ks = 0u; ks < 2u; ks++) {
                u32 q_off = b_off + 1u + (ks << 4) + (t << 1);
                u32 a[4] = {fg_e4m3_f16(fg_u16_at(sw0, q_off)),
                            fg_e4m3_f16(fg_u16_at(sw1, q_off)),
                            fg_e4m3_f16(fg_u16_at(sw0, q_off + 8u)),
                            fg_e4m3_f16(fg_u16_at(sw1, q_off + 8u))};
                u32 k_col = (ch << 9) + x_off + (blk << 5) + (ks << 4) + (t << 1);
#pragma unroll
                for (u32 ti = 0u; ti < NT; ti++) {
                    u32 tok = r0 + (ti << 3) + g;
                    u32 bh[2] = {0u, 0u}, bl[2] = {0u, 0u};
                    if (tok < m) {
                        const float* xp = x + u64(tok) * k + k_col;
                        float2 v0 = *reinterpret_cast<const float2*>(xp);
                        float2 v1 = *reinterpret_cast<const float2*>(xp + 8u);
                        bh[0] = trellis::f32x2_to_f16x2(v0.x, v0.y);
                        bh[1] = trellis::f32x2_to_f16x2(v1.x, v1.y);
                        bl[0] = trellis::f32x2_to_f16x2(
                            v0.x - trellis::f16_to_f32(bh[0] & 0xFFFFu),
                            v0.y - trellis::f16_to_f32(bh[0] >> 16));
                        bl[1] = trellis::f32x2_to_f16x2(
                            v1.x - trellis::f16_to_f32(bh[1] & 0xFFFFu),
                            v1.y - trellis::f16_to_f32(bh[1] >> 16));
                    }
                    trellis::mma_m16n8k16(tmp[ti], a, bh);
                    trellis::mma_m16n8k16(tmp[ti], a, bl);
                }
            }
#pragma unroll
            for (u32 ti = 0u; ti < NT; ti++) {
                acc[ti][0] = fmaf(sc0, tmp[ti][0], acc[ti][0]);
                acc[ti][1] = fmaf(sc0, tmp[ti][1], acc[ti][1]);
                acc[ti][2] = fmaf(sc1, tmp[ti][2], acc[ti][2]);
                acc[ti][3] = fmaf(sc1, tmp[ti][3], acc[ti][3]);
            }
        }
    }
    prim::block_sync();
#pragma unroll
    for (u32 ti = 0u; ti < NT; ti++) {
        u32 tr = (ti << 3) + (t << 1);
        sm.red[warp][tr][g] = acc[ti][0]; sm.red[warp][tr + 1u][g] = acc[ti][1];
        sm.red[warp][tr][g + 8u] = acc[ti][2]; sm.red[warp][tr + 1u][g + 8u] = acc[ti][3];
    }
    prim::block_sync();
    u32 col = threadIdx.x & 15u, tr0 = threadIdx.x >> 4;
#pragma unroll
    for (u32 ti = 0u; ti < NT; ti++) {
        u32 tr = tr0 + (ti << 3);
        if (r0 + tr < m) {
            float v = sm.red[0][tr][col] + sm.red[1][tr][col] + sm.red[2][tr][col] +
                      sm.red[3][tr][col];
            if constexpr (SPLIT) out[(u64(p) * m + (r0 + tr)) * n + n0 + col] = v;
            else {
                u64 off = u64(r0 + tr) * n + n0 + col;
                out[off] = fg_epilogue(v, other, off, n0 + col, mode);
            }
        }
    }
}

struct alignas(16) fg_mx64_smem {
    u32 sw[64][136];
    u32 sxh[64][36];
};

static __device__ __forceinline__ u16 fg_u16_odd(const u32* s, u32 byte_off) {
    u32 w = byte_off >> 2, out;
    asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(s[w]), "r"(s[w + 1u]), "r"((byte_off & 3u) << 3));
    return static_cast<u16>(out);
}

template <bool SPLIT>
static __device__ __forceinline__ void fg_mxfp8_r64(const float* x, const u8* w, const float* other,
                                                    float* out, u32 m, u32 n, u32 k, u32 mode,
                                                    u32 ks, fg_mx64_smem& sm) {
    u32 tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u, g = lane >> 2, t = lane & 3u;
    u32 wm = warp >> 2, wn = warp & 3u, nb = (n + 63u) >> 6;
    u32 ob = SPLIT ? (blockIdx.x % nb) : blockIdx.x, p = SPLIT ? (blockIdx.x / nb) : 0u;
    u32 n0 = ob << 6, r0 = blockIdx.y << 6, chunks = k >> 9, ch_per = SPLIT ? (chunks / ks) : chunks;
    u32 ch0 = SPLIT ? (p * ch_per) : 0u, ch1 = ch0 + ch_per, wr_row = tid >> 5, wr_col = tid & 31u;
    u32 xr0 = tid >> 4, xc = (tid & 15u) << 2, xw = (tid & 15u) << 1, n_base = wn << 4, m_base = wm << 5;
    u64 row_bytes = u64(k >> 5) * 33ull;
    const u32 *sw0 = sm.sw[n_base + g], *sw1 = sm.sw[n_base + g + 8u];
    const u8 *sb0 = reinterpret_cast<const u8*>(sw0), *sb1 = reinterpret_cast<const u8*>(sw1);
    float acc[4][4];
#pragma unroll
    for (u32 ti = 0u; ti < 4u; ti++) acc[ti][0] = acc[ti][1] = acc[ti][2] = acc[ti][3] = 0.0f;
    for (u32 ch = ch0; ch < ch1; ch++) {
        u64 ch_off = u64(ch) * 528ull;
#pragma unroll
        for (u32 dr = 0u; dr < 64u; dr += 8u) {
            u32 nr = wr_row + dr;
            *reinterpret_cast<uint4*>(&sm.sw[nr][wr_col << 2]) =
                (n0 + nr < n) ? fg_ld_nc4(w + u64(n0 + nr) * row_bytes + ch_off + (wr_col << 4)) : make_uint4(0u, 0u, 0u, 0u);
        }
        if (tid < 64u) *reinterpret_cast<uint4*>(&sm.sw[tid][128u]) =
            (n0 + tid < n) ? fg_ld_nc4(w + u64(n0 + tid) * row_bytes + ch_off + 512u) : make_uint4(0u, 0u, 0u, 0u);
        for (u32 sub = 0u; sub < 8u; sub++) {
            u32 k_sub = (ch << 9) + (sub << 6);
#pragma unroll
            for (u32 dr = 0u; dr < 64u; dr += 16u) {
                u32 r = xr0 + dr;
                uint2 hi = make_uint2(0u, 0u);
                if (r0 + r < m) {
                    float4 xv = *reinterpret_cast<const float4*>(x + u64(r0 + r) * k + k_sub + xc);
                    hi = make_uint2(trellis::f32x2_to_f16x2(xv.x, xv.y), trellis::f32x2_to_f16x2(xv.z, xv.w));
                }
                *reinterpret_cast<uint2*>(&sm.sxh[r][xw]) = hi;
            }
            prim::block_sync();
#pragma unroll
            for (u32 blk = 0u; blk < 2u; blk++) {
                u32 b_off = ((sub << 1) + blk) * 33u, e0 = sb0[b_off], e1 = sb1[b_off];
                float sc0 = __uint_as_float(e0 == 0u ? 0x00400000u : (e0 << 23));
                float sc1 = __uint_as_float(e1 == 0u ? 0x00400000u : (e1 << 23)), tmp[4][4];
#pragma unroll
                for (u32 ti = 0u; ti < 4u; ti++) tmp[ti][0] = tmp[ti][1] = tmp[ti][2] = tmp[ti][3] = 0.0f;
#pragma unroll
                for (u32 ks_i = 0u; ks_i < 2u; ks_i++) {
                    u32 q_off = b_off + 1u + (ks_i << 4) + (t << 1), a[4];
                    if (blk == 0u) {
                        a[0] = fg_e4m3_f16(fg_u16_odd(sw0, q_off)); a[1] = fg_e4m3_f16(fg_u16_odd(sw1, q_off));
                        a[2] = fg_e4m3_f16(fg_u16_odd(sw0, q_off + 8u)); a[3] = fg_e4m3_f16(fg_u16_odd(sw1, q_off + 8u));
                    } else {
                        a[0] = fg_e4m3_f16(*reinterpret_cast<const u16*>(sb0 + q_off)); a[1] = fg_e4m3_f16(*reinterpret_cast<const u16*>(sb1 + q_off));
                        a[2] = fg_e4m3_f16(*reinterpret_cast<const u16*>(sb0 + q_off + 8u)); a[3] = fg_e4m3_f16(*reinterpret_cast<const u16*>(sb1 + q_off + 8u));
                    }
                    u32 xu = (blk << 4) + (ks_i << 3) + t;
#pragma unroll
                    for (u32 ti = 0u; ti < 4u; ti++) {
                        u32 xr = m_base + (ti << 3) + g, bh[2] = {sm.sxh[xr][xu], sm.sxh[xr][xu + 4u]};
                        trellis::mma_m16n8k16(tmp[ti], a, bh);
                    }
                }
#pragma unroll
                for (u32 ti = 0u; ti < 4u; ti++) {
                    acc[ti][0] = fmaf(sc0, tmp[ti][0], acc[ti][0]); acc[ti][1] = fmaf(sc0, tmp[ti][1], acc[ti][1]);
                    acc[ti][2] = fmaf(sc1, tmp[ti][2], acc[ti][2]); acc[ti][3] = fmaf(sc1, tmp[ti][3], acc[ti][3]);
                }
            }
            prim::block_sync();
        }
    }
    u32 col0 = n0 + n_base + g, col1 = col0 + 8u;
#pragma unroll
    for (u32 ti = 0u; ti < 4u; ti++) {
        u32 row0 = r0 + m_base + (ti << 3) + (t << 1);
        if (row0 < m) {
            u64 off0 = u64(SPLIT ? (p * m + row0) : row0) * n;
            if (col0 < n) out[off0 + col0] = SPLIT ? acc[ti][0] : fg_epilogue(acc[ti][0], other, off0 + col0, col0, mode);
            if (col1 < n) out[off0 + col1] = SPLIT ? acc[ti][2] : fg_epilogue(acc[ti][2], other, off0 + col1, col1, mode);
        }
        if (row0 + 1u < m) {
            u64 off1 = u64(SPLIT ? (p * m + row0 + 1u) : (row0 + 1u)) * n;
            if (col0 < n) out[off1 + col0] = SPLIT ? acc[ti][1] : fg_epilogue(acc[ti][1], other, off1 + col0, col0, mode);
            if (col1 < n) out[off1 + col1] = SPLIT ? acc[ti][3] : fg_epilogue(acc[ti][3], other, off1 + col1, col1, mode);
        }
    }
}

// Streaming MXFP8 Tensor-Core matmul, up to 8 rows per block.
extern "C" __global__ __launch_bounds__(128, 6) void tessera_mm_mxfp8_r8(
    const float* x, const u8* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode) {
    __shared__ fg_mx_smem sm;
    fg_mxfp8_mm<1u, false>(x, w, other, out, m, n, k, mode, 1u, sm);
}

// Split-K MXFP8 Tensor-Core matmul, up to 8 rows per block.
extern "C" __global__ __launch_bounds__(128, 6) void tessera_mm_mxfp8_r8_sk(
    const float* x, const u8* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode,
    u32 ks) {
    __shared__ fg_mx_smem sm;
    fg_mxfp8_mm<1u, true>(x, w, other, out, m, n, k, mode, ks, sm);
}

// Split-K MXFP8 Tensor-Core matmul, up to 16 rows per block.
extern "C" __global__ __launch_bounds__(128, 4) void tessera_mm_mxfp8_r16_sk(
    const float* x, const u8* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode,
    u32 ks) {
    __shared__ fg_mx_smem sm;
    fg_mxfp8_mm<2u, true>(x, w, other, out, m, n, k, mode, ks, sm);
}

// Sums split-K partials in order and applies matmul epilogue.
extern "C" __global__ __launch_bounds__(256, 4) void tessera_mm_reduce(
    const float* part, const float* other, float* out, u32 m, u32 n, u32 ks, u32 mode) {
    u64 idx = u64(blockIdx.x) * 256u + threadIdx.x;
    u64 mn = u64(m) * n;
    if (idx >= mn) return;
    float v = 0.0f;
    for (u32 p = 0u; p < ks; p++) {
        v += part[u64(p) * mn + idx];
    }
    out[idx] = fg_epilogue(v, other, idx, u32(idx % n), mode);
}

// Streaming MXFP8 Tensor-Core matmul, up to 16 rows per block.
extern "C" __global__ __launch_bounds__(128, 4) void tessera_mm_mxfp8_r16(
    const float* x, const u8* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode) {
    __shared__ fg_mx_smem sm;
    fg_mxfp8_mm<2u, false>(x, w, other, out, m, n, k, mode, 1u, sm);
}

static __device__ __forceinline__ float fg_gemv_act(float s, u32 mode, float a, float b) {
    if (mode == 1u) return fg_silu(s * a);
    if (mode == 2u) return (1.0f / (1.0f + expf(-(s * a)))) * b;
    return s;
}

// Staged-X MXFP8 Tensor-Core matmul, 64x64 per block.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_mm_mxfp8_r64(
    const float* x, const u8* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode) {
    __shared__ fg_mx64_smem sm;
    fg_mxfp8_r64<false>(x, w, other, out, m, n, k, mode, 1u, sm);
}

// Split-K staged-X MXFP8 Tensor-Core matmul, 64x64 per block.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_mm_mxfp8_r64_sk(
    const float* x, const u8* w, const float* other, float* out, u32 m, u32 n, u32 k, u32 mode,
    u32 ks) {
    __shared__ fg_mx64_smem sm;
    fg_mxfp8_r64<true>(x, w, other, out, m, n, k, mode, ks, sm);
}

// Streaming 8-BF16 decode GEMV, one warp per output row.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_gemv_bf16(
    const float* x, const u16* w, float* out, u32 n, u32 k, u32 mode, float a, float b) {
    u32 row = blockIdx.x * TM_WARPS + (threadIdx.x >> 5);
    if (row >= n) return;
    u32 lane = threadIdx.x & 31u;
    float s = fg_dot8_bf16(x, w + u64(row) * k, k, lane);
    if (lane == 0u) out[row] = fg_gemv_act(s, mode, a, b);
}

// Streaming 8-BF16 narrow decode GEMV, one block per output row.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_gemv_bf16_narrow(
    const float* x, const u16* w, float* out, u32 n, u32 k, u32 mode, float a, float b) {
    __shared__ float part[TM_WARPS];
    u32 row = blockIdx.x;
    if (row >= n) return;
    u32 tid = threadIdx.x;
    const u16* wr = w + u64(row) * k;
    float acc = 0.0f;
    for (u32 i = tid << 3; i < k; i += 2048u) acc = fg_dot8_step(acc, x, wr, i);
    float ws = fg_warp_sum(acc);
    if ((tid & 31u) == 0u) part[tid >> 5] = ws;
    prim::block_sync();
    if (tid == 0u) {
        float s = part[0] + part[1] + part[2] + part[3] + part[4] + part[5] + part[6] + part[7];
        out[row] = fg_gemv_act(s, mode, a, b);
    }
}

// Tensor-Core BF16 stacked matmul, 16x16 per block.
extern "C" __global__ __launch_bounds__(128, 4) void tessera_stacked_lanes_tc(
    const float* x, const u16* w, float* out, u32 rows, u32 n, u32 k, u32 block, u32 first,
    u32 col_step, u32 x_bs, u32 x_rs, u32 o_bs, u32 o_rs) {
    __shared__ fg_bf16_smem<2u> sm;
    u32 rtiles = (rows + 15u) >> 4u;
    u32 batch = blockIdx.y / rtiles;
    u32 r0 = (blockIdx.y % rtiles) << 4u;
    const float* xb = x + u64(batch) * x_bs + u64(r0) * x_rs;
    const u16* wb = w + u64(batch) * block + first;
    float* ob = out + u64(batch) * o_bs + u64(r0) * o_rs;
    u32 nr = min(rows - r0, 16u);
    fg_bf16_mm<2u, false>(xb, wb, nullptr, ob, nr, n, k, col_step, x_rs, o_rs, 0u, 0u, 1u, sm);
}

// Streaming 8-BF16 stacked matmul, up to 8 rows per warp.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_stacked_lanes(
    const float* x, const u16* w, float* out, u32 len, u32 rows, u32 n, u32 k, u32 block,
    u32 first, u32 col_step, u32 x_bs, u32 x_rs, u32 o_bs, u32 o_rs) {
    u32 idx = blockIdx.x * TM_WARPS + (threadIdx.x >> 5);
    if (idx >= len) return;
    u32 lane = threadIdx.x & 31u;
    u32 batch = idx / n;
    u32 col = idx % n;
    u32 r0 = blockIdx.y * 8u;
    u32 nr = min(rows - r0, 8u);
    const u16* wr = w + u64(batch) * block + first + u64(col) * col_step;
    const float* xb = x + u64(batch) * x_bs + u64(r0) * x_rs;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (u32 i = lane << 3; i < k; i += 256u) {
        uint4 h = fg_ld_nc4(wr + i);
        float a0 = __uint_as_float(h.x << 16), a1 = __uint_as_float(h.x & 0xFFFF0000u);
        float a2 = __uint_as_float(h.y << 16), a3 = __uint_as_float(h.y & 0xFFFF0000u);
        float a4 = __uint_as_float(h.z << 16), a5 = __uint_as_float(h.z & 0xFFFF0000u);
        float a6 = __uint_as_float(h.w << 16), a7 = __uint_as_float(h.w & 0xFFFF0000u);
#pragma unroll
        for (u32 r = 0u; r < 8u; r++) {
            if (r >= nr) break;
            const float* xp = xb + u64(r) * x_rs + i;
            float4 x0 = *reinterpret_cast<const float4*>(xp);
            float4 x1 = *reinterpret_cast<const float4*>(xp + 4u);
            acc[r] = fmaf(a0, x0.x, fmaf(a1, x0.y, fmaf(a2, x0.z, fmaf(a3, x0.w, acc[r]))));
            acc[r] = fmaf(a4, x1.x, fmaf(a5, x1.y, fmaf(a6, x1.z, fmaf(a7, x1.w, acc[r]))));
        }
    }
#pragma unroll
    for (u32 r = 0u; r < 8u; r++) {
        if (r >= nr) break;
        float s = fg_warp_sum(acc[r]);
        if (lane == 0u) out[u64(batch) * o_bs + u64(r0 + r) * o_rs + col] = s;
    }
}

// Streaming chunk-tiled MXFP8 decode GEMV, one warp per output.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_gemv_mxfp8(
    const float* x, const u8* w, float* out, u32 n, u32 k) {
    __shared__ __align__(16) u32 x_h[512];
    __shared__ __align__(16) u32 sm_buf[TM_WARPS][268];
    u32 warp = threadIdx.x >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 row = blockIdx.x * TM_WARPS + warp;
    bool live = row < n;
    u32 chunks = k >> 10;
    u64 row_off = u64(row) * (k >> 5) * 33ull;
    u32* sm_u32 = sm_buf[warp];
    u8* sm_u8 = reinterpret_cast<u8*>(sm_u32);
    float acc = 0.0f;
    uint4 cur0, cur1, cur2;
    if (live && chunks > 0u) {
        const u8* p = w + row_off;
        cur0 = fg_ld_nc4(p + (lane << 4));
        cur1 = fg_ld_nc4(p + 512u + (lane << 4));
        if (lane < 2u) cur2 = fg_ld_nc4(p + 1024u + (lane << 4));
    }
    for (u32 ch = 0u; ch < chunks; ch++) {
        fg_stage_chunk(x_h, x, ch << 10);
        if (live) {
            *reinterpret_cast<uint4*>(sm_u32 + (lane << 2)) = cur0;
            *reinterpret_cast<uint4*>(sm_u32 + 128u + (lane << 2)) = cur1;
            if (lane < 2u) *reinterpret_cast<uint4*>(sm_u32 + 256u + (lane << 2)) = cur2;
            __syncwarp();
            if (ch + 1u < chunks) {
                const u8* np = w + row_off + u64(ch + 1u) * 1056ull;
                cur0 = fg_ld_nc4(np + (lane << 4));
                cur1 = fg_ld_nc4(np + 512u + (lane << 4));
                if (lane < 2u) cur2 = fg_ld_nc4(np + 1024u + (lane << 4));
            }
#pragma unroll
            for (u32 step = 0u; step < 8u; step++) {
                u32 b_off = ((step << 2) + (lane >> 3)) * 33u;
                u32 eb = sm_u8[b_off];
                float sc = __uint_as_float(eb == 0u ? 0x00400000u : (eb << 23));
                u32 qw = fg_unaligned_u32(sm_u32, b_off + 1u + ((lane & 7u) << 2));
                u32 h0, h1;
                u16 q0 = static_cast<u16>(qw);
                u16 q1 = static_cast<u16>(qw >> 16);
                asm("{.reg .b16 t; mov.b16 t, %1; cvt.rn.f16x2.e4m3x2 %0, t;}" : "=r"(h0) : "h"(q0));
                asm("{.reg .b16 t; mov.b16 t, %1; cvt.rn.f16x2.e4m3x2 %0, t;}" : "=r"(h1) : "h"(q1));
                uint2 xq = *reinterpret_cast<const uint2*>(x_h + (((step << 7) + (lane << 2)) >> 1));
                acc = fmaf(sc, fg_dot4(h0, h1, xq), acc);
            }
        }
        prim::block_sync();
    }
    if (live) {
        float s = fg_warp_sum(acc);
        if (lane == 0u) out[row] = s;
    }
}

// Streaming chunk-tiled 2-row INT4_G32 decode GEMV.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_gemv_int4(
    const float* x, const u8* w, float* out, u32 n, u32 k) {
    __shared__ __align__(16) u32 x_h[512];
    __shared__ __align__(16) u32 sm_buf[TM_WARPS][296];
    u32 warp = threadIdx.x >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 row0 = (blockIdx.x * TM_WARPS + warp) << 1;
    bool live = row0 + 1u < n;
    u32 chunks = k >> 10;
    u64 row_bytes = u64(k >> 5) * 18ull;
    const u8* w0 = w + u64(row0) * row_bytes;
    const u8* w1 = w0 + row_bytes;
    u32* sm_u32 = sm_buf[warp];
    u8* sm_u8 = reinterpret_cast<u8*>(sm_u32);
    float acc0 = 0.0f, acc1 = 0.0f;
    uint4 cur0, cur1, cur2;
    if (live && chunks > 0u) {
        cur0 = fg_ld_nc4(w0 + (lane << 4));
        cur1 = fg_ld_nc4(w1 + (lane << 4));
        if (lane < 4u) cur2 = fg_ld_nc4(w0 + 512u + (lane << 4));
        else if (lane < 8u) cur2 = fg_ld_nc4(w1 + 512u + ((lane - 4u) << 4));
    }
    for (u32 ch = 0u; ch < chunks; ch++) {
        fg_stage_chunk(x_h, x, ch << 10);
        if (live) {
            *reinterpret_cast<uint4*>(sm_u32 + (lane << 2)) = cur0;
            *reinterpret_cast<uint4*>(sm_u32 + 144u + (lane << 2)) = cur1;
            if (lane < 4u) *reinterpret_cast<uint4*>(sm_u32 + 128u + (lane << 2)) = cur2;
            else if (lane < 8u) *reinterpret_cast<uint4*>(sm_u32 + 272u + ((lane - 4u) << 2)) = cur2;
            __syncwarp();
            if (ch + 1u < chunks) {
                u64 off = u64(ch + 1u) * 576ull;
                cur0 = fg_ld_nc4(w0 + off + (lane << 4));
                cur1 = fg_ld_nc4(w1 + off + (lane << 4));
                if (lane < 4u) cur2 = fg_ld_nc4(w0 + off + 512u + (lane << 4));
                else if (lane < 8u) cur2 = fg_ld_nc4(w1 + off + 512u + ((lane - 4u) << 4));
            }
#pragma unroll
            for (u32 step = 0u; step < 8u; step++) {
                u32 b0 = ((step << 2) + (lane >> 3)) * 18u;
                u32 b1 = 576u + b0;
                u32 q_off = 2u + ((lane & 7u) << 1);
                float sc0 = trellis::f16_to_f32(*reinterpret_cast<const u16*>(sm_u8 + b0));
                float sc1 = trellis::f16_to_f32(*reinterpret_cast<const u16*>(sm_u8 + b1));
                u32 q0 = *reinterpret_cast<const u16*>(sm_u8 + b0 + q_off);
                u32 q1 = *reinterpret_cast<const u16*>(sm_u8 + b1 + q_off);
                uint2 xq = *reinterpret_cast<const uint2*>(x_h + (((step << 7) + (lane << 2)) >> 1));
                u32 a0, a1, b0h, b1h;
                asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(a0) : "r"(0x64006400u | (q0 & 0x0Fu) | ((q0 & 0xF0u) << 12)), "r"(0x64086408u));
                asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(a1) : "r"(0x64006400u | ((q0 >> 8) & 0x0Fu) | (((q0 >> 8) & 0xF0u) << 12)), "r"(0x64086408u));
                asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(b0h) : "r"(0x64006400u | (q1 & 0x0Fu) | ((q1 & 0xF0u) << 12)), "r"(0x64086408u));
                asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(b1h) : "r"(0x64006400u | ((q1 >> 8) & 0x0Fu) | (((q1 >> 8) & 0xF0u) << 12)), "r"(0x64086408u));
                acc0 = fmaf(sc0, fg_dot4(a0, a1, xq), acc0);
                acc1 = fmaf(sc1, fg_dot4(b0h, b1h, xq), acc1);
            }
        }
        prim::block_sync();
    }
    if (live) {
        float s0 = fg_warp_sum(acc0), s1 = fg_warp_sum(acc1);
        if (lane == 0u) {
            out[row0] = s0;
            out[row0 + 1u] = s1;
        }
    }
}

#endif
