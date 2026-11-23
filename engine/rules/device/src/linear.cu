#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

#define TM_WARPS 8u
#define TM_ROWS 4u

template <u32 MAX_R, u32 WARPS = TM_WARPS>
struct alignas(16) tm_smem {
    u32 tiles[WARPS][256];
    float buf[WARPS][MAX_R][128];
};

static __device__ __forceinline__ u32 tm_word(const u8* p, u32 off) {
    return *reinterpret_cast<const u32*>(p + off);
}

static __device__ __forceinline__ void tm_load4(const float* p, float v[4]) {
    float4 f = *reinterpret_cast<const float4*>(p);
    v[0] = f.x;
    v[1] = f.y;
    v[2] = f.z;
    v[3] = f.w;
}

template <u32 K, u32 BOOK, u32 MAX_R, bool TURNED, u32 WARPS = TM_WARPS>
static __device__ __forceinline__ void tm_body(const float* x, const u8* w, const float* suh,
                                               const float* svh, float* out, u32 m, u32 n, u32 k,
                                               tm_smem<MAX_R, WARPS>& sm) {
    u32 ob = blockIdx.x;
    u32 r0 = MAX_R == 1u ? 0u : blockIdx.y * MAX_R;
    if (ob >= (n >> 7) || r0 >= m) return;
    u32 nr = MAX_R == 1u ? 1u : min(MAX_R, m - r0);
    u32 warp = threadIdx.x >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 in_blocks = k >> 7;
    u32 full_ibs = MAX_R == 1u ? (in_blocks / WARPS) * WARPS : in_blocks;
    u32 rem_ibs = in_blocks - full_ibs;
    u32 my_ibs = full_ibs > warp ? ((full_ibs - 1u - warp) / WARPS + 1u) : 0u;
    u32 steps = my_ibs << 3;
    bool wide = (reinterpret_cast<u64>(w + trellis::HEAD) & 15u) == 0u;
    bool input_side = tm_word(w, 8) == trellis::INPUT_SIDE;
    trellis::lane_map pos = trellis::map_lane<K, BOOK>(lane);
    float* wbuf = &sm.buf[warp][0][0];
    u32* wtiles = sm.tiles[warp];
    float acc[8][4];
#pragma unroll
    for (u32 mt = 0; mt < 8u; mt++) {
        acc[mt][0] = 0.0f;
        acc[mt][1] = 0.0f;
        acc[mt][2] = 0.0f;
        acc[mt][3] = 0.0f;
    }
    if (steps > 0u) {
        u32 reg[8];
        trellis::tile_load<K>(trellis::tile_ptr<K>(w, k, n, 0u, warp << 3, ob, input_side), lane,
                              wide, reg);
        for (u32 st = 0u; st < steps; st++) {
            u32 ib = warp + (st >> 3) * WARPS;
            u32 ks = st & 7u;
            if (ks == 0u) {
                __syncwarp();
                u32 col = (ib << 7) + (lane << 2);
                float sc[4];
                if constexpr (TURNED) tm_load4(suh + col, sc);
                for (u32 r = 0u; r < nr; r++) {
                    float v[4];
                    tm_load4(x + u64(r0 + r) * k + col, v);
                    if constexpr (TURNED) {
                        v[0] *= sc[0];
                        v[1] *= sc[1];
                        v[2] *= sc[2];
                        v[3] *= sc[3];
                        trellis::wht128(v, trellis::INV_SQRT128);
                    }
                    *reinterpret_cast<float4*>(wbuf + (r << 7) + (lane << 2)) =
                        make_float4(v[0], v[1], v[2], v[3]);
                }
                __syncwarp();
            }
            __syncwarp();
            trellis::tile_store<K>(reg, lane, wide, wtiles);
            __syncwarp();
            if (st + 1u < steps) {
                u32 nti = ((warp + ((st + 1u) >> 3) * WARPS) << 3) + ((st + 1u) & 7u);
                trellis::tile_load<K>(trellis::tile_ptr<K>(w, k, n, 0u, nti, ob, input_side), lane,
                                      wide, reg);
            }
            u32 fb[2];
            trellis::act_frag(wbuf, nr, ks, lane, fb);
#pragma unroll
            for (u32 mt = 0; mt < 8u; mt++) {
                u32 fa[4];
                trellis::decode_frag<K, BOOK>(wtiles + mt * 8u * K, pos, fa);
                trellis::mma_m16n8k16(acc[mt], fa, fb);
            }
        }
    }
    if constexpr (MAX_R == 1u) {
        if (rem_ibs > 0u) {
            if (warp < rem_ibs) {
                u32 col = ((full_ibs + warp) << 7) + (lane << 2);
                float v[4];
                tm_load4(x + col, v);
                if constexpr (TURNED) {
                    float sc[4];
                    tm_load4(suh + col, sc);
                    v[0] *= sc[0];
                    v[1] *= sc[1];
                    v[2] *= sc[2];
                    v[3] *= sc[3];
                    trellis::wht128(v, trellis::INV_SQRT128);
                }
                *reinterpret_cast<float4*>(wbuf + (lane << 2)) =
                    make_float4(v[0], v[1], v[2], v[3]);
            }
            prim::block_sync();
            u32 ti0 = full_ibs << 3;
            u32 rem_steps = (rem_ibs << 3) / WARPS;
            u32 reg[8];
            trellis::tile_load<K>(trellis::tile_ptr<K>(w, k, n, 0u, ti0 + warp, ob, input_side),
                                  lane, wide, reg);
            for (u32 st = 0u; st < rem_steps; st++) {
                u32 ti = warp + st * WARPS;
                __syncwarp();
                trellis::tile_store<K>(reg, lane, wide, wtiles);
                __syncwarp();
                if (st + 1u < rem_steps) {
                    trellis::tile_load<K>(
                        trellis::tile_ptr<K>(w, k, n, 0u, ti0 + ti + WARPS, ob, input_side),
                        lane, wide, reg);
                }
                u32 fb[2];
                trellis::act_frag(&sm.buf[ti >> 3][0][0], 1u, ti & 7u, lane, fb);
#pragma unroll
                for (u32 mt = 0; mt < 8u; mt++) {
                    u32 fa[4];
                    trellis::decode_frag<K, BOOK>(wtiles + mt * 8u * K, pos, fa);
                    trellis::mma_m16n8k16(acc[mt], fa, fb);
                }
            }
            prim::block_sync();
        }
    }
    __syncwarp();
    trellis::store_acc(acc, nr, lane, wbuf);
    prim::block_sync();
    if (warp < nr) {
        u32 base = (ob << 7) + (lane << 2);
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (u32 wi = 0u; wi < WARPS; wi++) {
            float4 p = *reinterpret_cast<const float4*>(&sm.buf[wi][warp][lane << 2]);
            v[0] += p.x;
            v[1] += p.y;
            v[2] += p.z;
            v[3] += p.w;
        }
        if constexpr (TURNED) {
            trellis::wht128(v, trellis::INV_SQRT128);
            float sc[4];
            tm_load4(svh + base, sc);
            v[0] *= sc[0];
            v[1] *= sc[1];
            v[2] *= sc[2];
            v[3] *= sc[3];
        }
        *reinterpret_cast<float4*>(out + u64(r0 + warp) * n + base) =
            make_float4(v[0], v[1], v[2], v[3]);
    }
}

template <u32 K, u32 BOOK, u32 MAX_R>
static __device__ __noinline__ void tm_kernel(const float* x, const u8* w, const float* suh,
                                              const float* svh, float* out, u32 m, u32 n, u32 k,
                                              u32 turned, tm_smem<MAX_R>& sm) {
    if (turned != 0u) tm_body<K, BOOK, MAX_R, true>(x, w, suh, svh, out, m, n, k, sm);
    else tm_body<K, BOOK, MAX_R, false>(x, w, suh, svh, out, m, n, k, sm);
}

#define TM_DEF_1(NAME, K, BOOK, TURNED)                                                       \
    extern "C" __global__ __launch_bounds__(256, 2) void NAME(                                \
        const float* x, const u8* w, const float* suh, const float* svh, float* out, u32 m,   \
        u32 n, u32 k, u32 bits, u32 book, u32 turned) {                                       \
        (void)bits;                                                                           \
        (void)book;                                                                           \
        (void)turned;                                                                         \
        __shared__ tm_smem<1u> sm;                                                            \
        tm_body<K, BOOK, 1u, TURNED>(x, w, suh, svh, out, m, n, k, sm);                       \
    }

TM_DEF_1(tessera_trellis_k4_mcg_t1, 4u, trellis::BOOK_MCG, true)
TM_DEF_1(tessera_trellis_k4_mcg_p1, 4u, trellis::BOOK_MCG, false)
TM_DEF_1(tessera_trellis_k3_mul1_t1, 3u, trellis::BOOK_MUL1, true)
TM_DEF_1(tessera_trellis_k4_mul1_t1, 4u, trellis::BOOK_MUL1, true)

template <u32 BOOK, u32 MAX_R>
static __device__ void tm_dispatch_k(const float* x, const u8* w, const float* suh,
                                     const float* svh, float* out, u32 m, u32 n, u32 k, u32 bits,
                                     u32 turned, tm_smem<MAX_R>& sm) {
    if (bits == 2u) tm_kernel<2u, BOOK, MAX_R>(x, w, suh, svh, out, m, n, k, turned, sm);
    else if (bits == 3u) tm_kernel<3u, BOOK, MAX_R>(x, w, suh, svh, out, m, n, k, turned, sm);
    else tm_kernel<4u, BOOK, MAX_R>(x, w, suh, svh, out, m, n, k, turned, sm);
}

template <u32 MAX_R>
static __device__ void tm_dispatch(const float* x, const u8* w, const float* suh, const float* svh,
                                   float* out, u32 m, u32 n, u32 k, u32 bits, u32 book, u32 turned,
                                   tm_smem<MAX_R>& sm) {
    if (book == trellis::BOOK_PMCGX) {
        tm_dispatch_k<trellis::BOOK_PMCGX, MAX_R>(x, w, suh, svh, out, m, n, k, bits, turned, sm);
    } else if (book == trellis::BOOK_MUL1) {
        tm_dispatch_k<trellis::BOOK_MUL1, MAX_R>(x, w, suh, svh, out, m, n, k, bits, turned, sm);
    } else {
        tm_dispatch_k<trellis::BOOK_MCG, MAX_R>(x, w, suh, svh, out, m, n, k, bits, turned, sm);
    }
}

// Single-row decode trellis matmul with 12 KiB shared memory.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_trellis_mma1(
    const float* x, const u8* w, const float* suh, const float* svh, float* out, u32 m, u32 n,
    u32 k, u32 bits, u32 book, u32 turned) {
    __shared__ tm_smem<1u> sm;
    tm_dispatch<1u>(x, w, suh, svh, out, m, n, k, bits, book, turned, sm);
}

// Multi-row trellis matmul over 128-wide blocks.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_trellis_mma(
    const float* x, const u8* w, const float* suh, const float* svh, float* out, u32 m, u32 n,
    u32 k, u32 bits, u32 book, u32 turned) {
    __shared__ tm_smem<TM_ROWS> sm;
    tm_dispatch<TM_ROWS>(x, w, suh, svh, out, m, n, k, bits, book, turned, sm);
}

#if defined(TESSERA_CUDA)

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

static __device__ __forceinline__ float fg_dot8_bf16(const float* x, const u16* wr, u32 k,
                                                     u32 lane) {
    float acc = 0.0f;
    for (u32 i = lane << 3; i < k; i += 256u) {
        uint4 h = fg_ld_nc4(wr + i);
        float4 x0 = *reinterpret_cast<const float4*>(x + i);
        float4 x1 = *reinterpret_cast<const float4*>(x + i + 4u);
        acc = fmaf(__uint_as_float(h.x << 16), x0.x,
                   fmaf(__uint_as_float(h.x & 0xFFFF0000u), x0.y,
                        fmaf(__uint_as_float(h.y << 16), x0.z,
                             fmaf(__uint_as_float(h.y & 0xFFFF0000u), x0.w, acc))));
        acc = fmaf(__uint_as_float(h.z << 16), x1.x,
                   fmaf(__uint_as_float(h.z & 0xFFFF0000u), x1.y,
                        fmaf(__uint_as_float(h.w << 16), x1.z,
                             fmaf(__uint_as_float(h.w & 0xFFFF0000u), x1.w, acc))));
    }
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

// Streaming BF16 Tensor-Core small-batch matmul, 16x32 per block.
extern "C" __global__ __launch_bounds__(128, 6) void tessera_mm_bf16(
    const float* x, const u16* w, const float* add, float* out, u32 m, u32 n, u32 k,
    u32 add_mode) {
    __shared__ __align__(16) u32 sxh[16][36];
    __shared__ __align__(16) u32 sxl[16][36];
    __shared__ __align__(16) u32 sw[4][8][36];
    u32 warp = threadIdx.x >> 5;
    u32 lane = threadIdx.x & 31u;
    u32 g = lane >> 2;
    u32 t = lane & 3u;
    u32 r0 = blockIdx.y << 4;
    u32 n0 = (blockIdx.x << 5) + (warp << 3);
    const u16* wr = w + u64(n0 + g) * k;
    u32 xr = threadIdx.x >> 4;
    u32 xc = (threadIdx.x & 15u) << 2;
    u32 xw = (threadIdx.x & 15u) << 1;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint4 cur0 = fg_ld_nc4(wr + (t << 3));
    uint4 cur1 = fg_ld_nc4(wr + 32u + (t << 3));
    for (u32 k0 = 0u; k0 < k; k0 += 64u) {
#pragma unroll
        for (u32 dr = 0u; dr < 16u; dr += 8u) {
            u32 r = xr + dr;
            if (r0 + r < m) {
                float4 xv = *reinterpret_cast<const float4*>(x + u64(r0 + r) * k + k0 + xc);
                u32 h0 = fg_bf16x2(xv.x, xv.y);
                u32 h1 = fg_bf16x2(xv.z, xv.w);
                u32 l0 = fg_bf16x2(xv.x - __uint_as_float(h0 << 16),
                                   xv.y - __uint_as_float(h0 & 0xFFFF0000u));
                u32 l1 = fg_bf16x2(xv.z - __uint_as_float(h1 << 16),
                                   xv.w - __uint_as_float(h1 & 0xFFFF0000u));
                *reinterpret_cast<uint2*>(&sxh[r][xw]) = make_uint2(h0, h1);
                *reinterpret_cast<uint2*>(&sxl[r][xw]) = make_uint2(l0, l1);
            } else {
                *reinterpret_cast<uint2*>(&sxh[r][xw]) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2*>(&sxl[r][xw]) = make_uint2(0u, 0u);
            }
        }
        *reinterpret_cast<uint4*>(&sw[warp][g][t << 2]) = cur0;
        *reinterpret_cast<uint4*>(&sw[warp][g][16u + (t << 2)]) = cur1;
        prim::block_sync();
        if (k0 + 64u < k) {
            cur0 = fg_ld_nc4(wr + k0 + 64u + (t << 3));
            cur1 = fg_ld_nc4(wr + k0 + 96u + (t << 3));
        }
#pragma unroll
        for (u32 ks = 0u; ks < 4u; ks++) {
            u32 ah[4], al[4], b[2];
            u32 off = (ks << 3) + t;
            ah[0] = sxh[g][off];
            ah[1] = sxh[g + 8u][off];
            ah[2] = sxh[g][off + 4u];
            ah[3] = sxh[g + 8u][off + 4u];
            al[0] = sxl[g][off];
            al[1] = sxl[g + 8u][off];
            al[2] = sxl[g][off + 4u];
            al[3] = sxl[g + 8u][off + 4u];
            b[0] = sw[warp][g][off];
            b[1] = sw[warp][g][off + 4u];
            fg_mma_bf16(acc, ah, b);
            fg_mma_bf16(acc, al, b);
        }
        prim::block_sync();
    }
    u32 col = n0 + (t << 1);
    if (r0 + g < m) {
        u64 off = u64(r0 + g) * n + col;
        float2 v = make_float2(acc[0], acc[1]);
        if (add_mode != 0u) {
            float2 b = *reinterpret_cast<const float2*>(add + (add_mode == 1u ? off : u64(col)));
            v.x += b.x;
            v.y += b.y;
        }
        *reinterpret_cast<float2*>(out + off) = v;
    }
    if (r0 + g + 8u < m) {
        u64 off = u64(r0 + g + 8u) * n + col;
        float2 v = make_float2(acc[2], acc[3]);
        if (add_mode != 0u) {
            float2 b = *reinterpret_cast<const float2*>(add + (add_mode == 1u ? off : u64(col)));
            v.x += b.x;
            v.y += b.y;
        }
        *reinterpret_cast<float2*>(out + off) = v;
    }
}

// Streaming 8-BF16 decode GEMV, one warp per output row.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_gemv_bf16(
    const float* x, const u16* w, float* out, u32 n, u32 k) {
    u32 row = blockIdx.x * TM_WARPS + (threadIdx.x >> 5);
    if (row >= n) return;
    u32 lane = threadIdx.x & 31u;
    float s = fg_dot8_bf16(x, w + u64(row) * k, k, lane);
    if (lane == 0u) out[row] = s;
}

// Streaming 8-BF16 stacked matmul, one warp per output.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_stacked_lanes(
    const float* x, const u16* w, float* out, u32 len, u32 rows, u32 n, u32 k, u32 block,
    u32 first, u32 col_step) {
    u32 idx = blockIdx.x * TM_WARPS + (threadIdx.x >> 5);
    if (idx >= len) return;
    u32 lane = threadIdx.x & 31u;
    u32 col = idx % n;
    u32 rh = idx / n;
    u64 at = u64(rh / rows) * block + first + u64(col) * col_step;
    float s = fg_dot8_bf16(x + u64(rh) * k, w + at, k, lane);
    if (lane == 0u) out[idx] = s;
}

// Coalesced BF16 transposed stacked matmul, 128 cols per block.
extern "C" __global__ __launch_bounds__(256, 4) void tessera_stacked_wide(
    const float* x, const u16* w, float* out, u32 rows, u32 n, u32 k, u32 block, u32 first,
    u32 k_step) {
    __shared__ float sx[512];
    __shared__ __align__(16) float part[TM_WARPS][128];
    u32 tile = blockIdx.x;
    u32 rh = blockIdx.y;
    u32 grp = threadIdx.x >> 5;
    u32 tid = threadIdx.x & 31u;
    const float* xr = x + u64(rh) * k;
    for (u32 i = threadIdx.x; i < k; i += 256u) sx[i] = xr[i];
    prim::block_sync();
    const uint2* w2 = reinterpret_cast<const uint2*>(w + u64(rh / rows) * block + first);
    u32 step4 = k_step >> 2;
    u32 col4 = (tile << 5) + tid;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (u32 i = grp; i < k; i += TM_WARPS) {
        uint2 p = w2[u64(i) * step4 + col4];
        float xv = sx[i];
        a0 = fmaf(xv, __uint_as_float(p.x << 16), a0);
        a1 = fmaf(xv, __uint_as_float(p.x & 0xFFFF0000u), a1);
        a2 = fmaf(xv, __uint_as_float(p.y << 16), a2);
        a3 = fmaf(xv, __uint_as_float(p.y & 0xFFFF0000u), a3);
    }
    *reinterpret_cast<float4*>(&part[grp][tid << 2]) = make_float4(a0, a1, a2, a3);
    prim::block_sync();
    if (grp == 0u) {
        float4 s = *reinterpret_cast<const float4*>(&part[0][tid << 2]);
#pragma unroll
        for (u32 widx = 1u; widx < TM_WARPS; widx++) {
            float4 q = *reinterpret_cast<const float4*>(&part[widx][tid << 2]);
            s.x += q.x;
            s.y += q.y;
            s.z += q.z;
            s.w += q.w;
        }
        *reinterpret_cast<float4*>(out + u64(rh) * n + (tile << 7) + (tid << 2)) = s;
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
