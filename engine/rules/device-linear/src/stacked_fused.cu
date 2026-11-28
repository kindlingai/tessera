// Fused view-peeling BF16 stacked matmul (pipelined decode lanes/wide + 64x64x64 MMA).
typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;

#define SF_WARPS 8u

struct alignas(16) sf_p64_smem {
    u32 x[64][36];
    u32 w[64][36];
};

template <u32 BN>
struct alignas(16) sf_p16_smem {
    u32 x[16][36];
    u32 w[BN][36];
};

static __device__ __forceinline__ float sf_warp_sum(float v) {
#pragma unroll
    for (u32 off = 16u; off > 0u; off >>= 1u) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    return v;
}

static __device__ __forceinline__ uint4 sf_ld_nc4(const void* p) {
    u32 d0, d1, d2, d3;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "l"(p));
    return make_uint4(d0, d1, d2, d3);
}

static __device__ __forceinline__ uint2 sf_ld_nc2(const void* p) {
    u32 d0, d1;
    asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];" : "=r"(d0), "=r"(d1) : "l"(p));
    return make_uint2(d0, d1);
}

static __device__ __forceinline__ u32 sf_bf16x2(float lo, float hi) {
    u32 out;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(lo), "f"(hi));
    return out;
}

static __device__ __forceinline__ u32 sf_prmt(u32 a, u32 b, u32 sel) {
    u32 out;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(sel));
    return out;
}

static __device__ __forceinline__ void sf_ldm_x4(u32 d[4], const void* p) {
    u32 at = static_cast<u32>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(at));
}

static __device__ __forceinline__ void sf_ldm_x2(u32 d[2], const void* p) {
    u32 at = static_cast<u32>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(at));
}

static __device__ __forceinline__ void sf_mma_bf16(float d[4], const u32 a[4], const u32 b[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

static __device__ __forceinline__ float sf_bf16_lo(u32 word) {
    return __uint_as_float(word << 16u);
}

static __device__ __forceinline__ float sf_bf16_hi(u32 word) {
    return __uint_as_float(word & 0xFFFF0000u);
}

static __device__ __forceinline__ void sf_tr_pair(u32 w_sm[][36], u32 nc, u32 pair, uint2 p,
                                                  uint2 q) {
    w_sm[nc][pair] = sf_prmt(p.x, q.x, 0x5410u);
    w_sm[nc + 1u][pair] = sf_prmt(p.x, q.x, 0x7632u);
    w_sm[nc + 2u][pair] = sf_prmt(p.y, q.y, 0x5410u);
    w_sm[nc + 3u][pair] = sf_prmt(p.y, q.y, 0x7632u);
}

static __device__ __forceinline__ float sf_ld_x1(const u8* xb, u64 off, u32 x_bf16) {
    if (x_bf16 != 0u) {
        return __uint_as_float(u32(reinterpret_cast<const u16*>(xb)[off]) << 16u);
    }
    return reinterpret_cast<const float*>(xb)[off];
}

static __device__ __forceinline__ float4 sf_ld_xf4(const u8* xb, u64 off, u32 x_bf16) {
    if (x_bf16 != 0u) {
        uint2 h = *reinterpret_cast<const uint2*>(reinterpret_cast<const u16*>(xb) + off);
        return make_float4(sf_bf16_lo(h.x), sf_bf16_hi(h.x), sf_bf16_lo(h.y), sf_bf16_hi(h.y));
    }
    return *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(xb) + off);
}

static __device__ __forceinline__ uint2 sf_ld_xb4(const u8* xb, u64 off, u32 x_bf16) {
    if (x_bf16 != 0u) {
        return *reinterpret_cast<const uint2*>(reinterpret_cast<const u16*>(xb) + off);
    }
    float4 v = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(xb) + off);
    return make_uint2(sf_bf16x2(v.x, v.y), sf_bf16x2(v.z, v.w));
}

static __device__ __forceinline__ void sf_st_out1(u8* ob, u64 off, u32 out_bf16, float v) {
    if (out_bf16 != 0u) {
        reinterpret_cast<u16*>(ob)[off] = static_cast<u16>(sf_bf16x2(v, 0.0f) & 0xFFFFu);
    } else {
        reinterpret_cast<float*>(ob)[off] = v;
    }
}

static __device__ __forceinline__ void sf_st_out2(u8* ob, u64 off, u32 out_bf16, float a, float b) {
    if (out_bf16 != 0u) {
        *reinterpret_cast<u32*>(reinterpret_cast<u16*>(ob) + off) = sf_bf16x2(a, b);
    } else {
        asm volatile("st.global.cs.v2.f32 [%0], {%1,%2};"
                     :: "l"(reinterpret_cast<float*>(ob) + off), "f"(a), "f"(b));
    }
}

static __device__ __forceinline__ void sf_st_out4(u8* ob, u64 off, u32 out_bf16, float4 s) {
    if (out_bf16 != 0u) {
        *reinterpret_cast<uint2*>(reinterpret_cast<u16*>(ob) + off) =
            make_uint2(sf_bf16x2(s.x, s.y), sf_bf16x2(s.z, s.w));
    } else {
        *reinterpret_cast<float4*>(reinterpret_cast<float*>(ob) + off) = s;
    }
}

static __device__ __forceinline__ void sf_put_mma(u8* ob, u32 row0, u32 col, u32 rows, u32 n,
                                                  u32 out_rstep, const float a[4], u32 out_bf16) {
    if (col >= n) return;
    if (row0 < rows) {
        sf_st_out2(ob, u64(row0) * out_rstep + col, out_bf16, a[0], a[1]);
    }
    if (row0 + 8u < rows) {
        sf_st_out2(ob, u64(row0 + 8u) * out_rstep + col, out_bf16, a[2], a[3]);
    }
}

template <bool TRANSPOSED>
static __device__ __forceinline__ void sf_p64_exec(const u8* xb, const u16* w_base, u8* ob,
                                                   u32 r0, u32 n0, u32 rows, u32 n, u32 k,
                                                   u32 w_stride, u32 x_rstep, u32 out_rstep,
                                                   u32 x_bf16, u32 out_bf16, sf_p64_smem& sm) {
    u32 tid = threadIdx.x;
    u32 warp = tid >> 5u;
    u32 lane = tid & 31u;
    u32 wm = (warp >> 2u) << 5u;
    u32 wn = (warp & 3u) << 4u;
    u32 xr = tid >> 4u;
    u32 xlane = tid & 15u;
    u32 a_r0 = wm + (lane & 15u);
    u32 a_c = (lane >> 4u) << 2u;
    u32 b_r = wn + ((lane >> 4u) << 3u) + (lane & 7u);
    u32 b_c = ((lane >> 3u) & 1u) << 2u;
    const uint2* w2 = reinterpret_cast<const uint2*>(w_base);
    u32 col4 = (n0 >> 2u) + xlane;
    u32 step4 = w_stride >> 2u;
    float acc[2][2][4] = {{{0.f}}};
    uint2 xv[4];
    uint2 wv[4];
    auto load_tile = [&](u32 k0) {
#pragma unroll
        for (u32 i = 0u; i < 4u; i++) {
            u32 r = r0 + xr + (i << 4u);
            xv[i] = (r < rows) ? sf_ld_xb4(xb, u64(r) * x_rstep + k0 + (xlane << 2u), x_bf16)
                               : make_uint2(0u, 0u);
        }
        if constexpr (!TRANSPOSED) {
#pragma unroll
            for (u32 i = 0u; i < 4u; i++) {
                u32 wr = n0 + xr + (i << 4u);
                wv[i] = (wr < n) ? sf_ld_nc2(w_base + u64(wr) * w_stride + k0 + (xlane << 2u))
                                 : make_uint2(0u, 0u);
            }
        } else {
#pragma unroll
            for (u32 d = 0u; d < 2u; d++) {
                u32 ke = k0 + ((xr + (d << 4u)) << 1u);
                wv[d << 1u] = sf_ld_nc2(w2 + u64(ke) * step4 + col4);
                wv[(d << 1u) + 1u] = sf_ld_nc2(w2 + u64(ke + 1u) * step4 + col4);
            }
        }
    };
    auto store_tile = [&]() {
#pragma unroll
        for (u32 i = 0u; i < 4u; i++) {
            *reinterpret_cast<uint2*>(&sm.x[xr + (i << 4u)][xlane << 1u]) = xv[i];
        }
        if constexpr (!TRANSPOSED) {
#pragma unroll
            for (u32 i = 0u; i < 4u; i++) {
                *reinterpret_cast<uint2*>(&sm.w[xr + (i << 4u)][xlane << 1u]) = wv[i];
            }
        } else {
#pragma unroll
            for (u32 d = 0u; d < 2u; d++) {
                sf_tr_pair(sm.w, xlane << 2u, xr + (d << 4u), wv[d << 1u], wv[(d << 1u) + 1u]);
            }
        }
    };
    load_tile(0u);
    store_tile();
    __syncthreads();
    for (u32 k0 = 0u; k0 < k; k0 += 64u) {
        bool more = k0 + 64u < k;
        if (more) load_tile(k0 + 64u);
#pragma unroll
        for (u32 ks = 0u; ks < 4u; ks++) {
            u32 koff = ks << 3u;
            u32 a0[4], a1[4], b[4];
            sf_ldm_x4(a0, &sm.x[a_r0][koff + a_c]);
            sf_ldm_x4(a1, &sm.x[a_r0 + 16u][koff + a_c]);
            sf_ldm_x4(b, &sm.w[b_r][koff + b_c]);
            sf_mma_bf16(acc[0][0], a0, b);
            sf_mma_bf16(acc[0][1], a0, b + 2u);
            sf_mma_bf16(acc[1][0], a1, b);
            sf_mma_bf16(acc[1][1], a1, b + 2u);
        }
        if (more) {
            __syncthreads();
            store_tile();
            __syncthreads();
        }
    }
    u32 g = lane >> 2u;
    u32 t = lane & 3u;
#pragma unroll
    for (u32 mi = 0u; mi < 2u; mi++) {
        u32 row0 = r0 + wm + (mi << 4u) + g;
#pragma unroll
        for (u32 ni = 0u; ni < 2u; ni++) {
            sf_put_mma(ob, row0, n0 + wn + (ni << 3u) + (t << 1u), rows, n, out_rstep, acc[mi][ni],
                       out_bf16);
        }
    }
}

// Pipelined 8-BF16 stacked matmul with fused input/output strides.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_sf_lanes(
    const u8* x, const u16* w, u8* out, u32 len, u32 rows, u32 n, u32 k, u32 block, u32 first,
    u32 col_step, u32 x_bstep, u32 x_rstep, u32 x_off, u32 out_bstep, u32 out_rstep, u32 x_bf16,
    u32 out_bf16) {
    u32 idx = blockIdx.x * SF_WARPS + (threadIdx.x >> 5u);
    if (idx >= len) return;
    u32 lane = threadIdx.x & 31u;
    u32 batch = idx / n;
    u32 col = idx - batch * n;
    u32 r0 = blockIdx.y << 3u;
    u32 nr = min(rows - r0, 8u);
    const u16* wrow = w + u64(batch) * block + first + u64(col) * col_step;
    u64 xbase = u64(batch) * x_bstep + u64(r0) * x_rstep + x_off;
    float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    u32 pos = lane << 3u;
    if (pos < k) {
        uint4 nxt = sf_ld_nc4(wrow + pos);
        for (; pos < k; pos += 256u) {
            uint4 cur = nxt;
            if (pos + 256u < k) nxt = sf_ld_nc4(wrow + pos + 256u);
            float wv[8] = {sf_bf16_lo(cur.x), sf_bf16_hi(cur.x), sf_bf16_lo(cur.y),
                           sf_bf16_hi(cur.y), sf_bf16_lo(cur.z), sf_bf16_hi(cur.z),
                           sf_bf16_lo(cur.w), sf_bf16_hi(cur.w)};
            for (u32 r = 0u; r < nr; r++) {
                u64 xp = xbase + u64(r) * x_rstep + pos;
                float4 lo = sf_ld_xf4(x, xp, x_bf16);
                float4 hi = sf_ld_xf4(x, xp + 4u, x_bf16);
                float d = fmaf(wv[0], lo.x, fmaf(wv[1], lo.y, fmaf(wv[2], lo.z, wv[3] * lo.w)));
                d = fmaf(wv[4], hi.x, fmaf(wv[5], hi.y, fmaf(wv[6], hi.z, fmaf(wv[7], hi.w, d))));
                sum[r] += d;
            }
        }
    }
    u64 ob = u64(batch) * out_bstep + u64(r0) * out_rstep + col;
    for (u32 r = 0u; r < nr; r++) {
        float tot = sf_warp_sum(sum[r]);
        if (lane == 0u) sf_st_out1(out, ob + u64(r) * out_rstep, out_bf16, tot);
    }
}

// Pipelined single-barrier transposed BF16 stacked matmul with fused strides.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_sf_wide(
    const u8* x, const u16* w, u8* out, u32 rows, u32 n, u32 k, u32 block, u32 first, u32 k_step,
    u32 x_bstep, u32 x_rstep, u32 x_off, u32 out_bstep, u32 out_rstep, u32 x_bf16, u32 out_bf16) {
    (void)n;
    __shared__ __align__(16) float4 part[4][SF_WARPS][32];
    u32 rtiles = (rows + 3u) >> 2u;
    u32 batch = blockIdx.y / rtiles;
    u32 r0 = (blockIdx.y - batch * rtiles) << 2u;
    u32 nr = min(rows - r0, 4u);
    u32 grp = threadIdx.x >> 5u;
    u32 tid = threadIdx.x & 31u;
    u64 xbase = u64(batch) * x_bstep + u64(r0) * x_rstep + x_off;
    const uint2* w2 = reinterpret_cast<const uint2*>(w + u64(batch) * block + first);
    u32 step4 = k_step >> 2u;
    u32 col4 = (blockIdx.x << 5u) + tid;
    float4 acc[4] = {make_float4(0.f, 0.f, 0.f, 0.f), make_float4(0.f, 0.f, 0.f, 0.f),
                     make_float4(0.f, 0.f, 0.f, 0.f), make_float4(0.f, 0.f, 0.f, 0.f)};
    if (grp < k) {
        uint2 nxt = sf_ld_nc2(w2 + u64(grp) * step4 + col4);
        for (u32 i = grp; i < k; i += SF_WARPS) {
            uint2 cur = nxt;
            if (i + SF_WARPS < k) nxt = sf_ld_nc2(w2 + u64(i + SF_WARPS) * step4 + col4);
            float w0 = sf_bf16_lo(cur.x);
            float w1 = sf_bf16_hi(cur.x);
            float w2f = sf_bf16_lo(cur.y);
            float w3 = sf_bf16_hi(cur.y);
            for (u32 r = 0u; r < nr; r++) {
                float xv = sf_ld_x1(x, xbase + u64(r) * x_rstep + i, x_bf16);
                acc[r].x = fmaf(xv, w0, acc[r].x);
                acc[r].y = fmaf(xv, w1, acc[r].y);
                acc[r].z = fmaf(xv, w2f, acc[r].z);
                acc[r].w = fmaf(xv, w3, acc[r].w);
            }
        }
    }
    for (u32 r = 0u; r < nr; r++) part[r][grp][tid] = acc[r];
    __syncthreads();
    if (grp < nr) {
        float4 s = part[grp][0][tid];
#pragma unroll
        for (u32 widx = 1u; widx < SF_WARPS; widx++) {
            float4 q = part[grp][widx][tid];
            s.x += q.x;
            s.y += q.y;
            s.z += q.z;
            s.w += q.w;
        }
        u64 dst = u64(batch) * out_bstep + u64(r0 + grp) * out_rstep + (blockIdx.x << 7u)
                  + (tid << 2u);
        sf_st_out4(out, dst, out_bf16, s);
    }
}

template <bool TRANSPOSED>
static __device__ __forceinline__ void sf_mma_entry(
    const u8* x, const u16* w, u8* out, u32 rows, u32 n, u32 k, u32 block, u32 first,
    u32 w_stride, u32 x_bstep, u32 x_rstep, u32 x_off, u32 out_bstep, u32 out_rstep, u32 x_bf16,
    u32 out_bf16, sf_p64_smem& sm) {
    u32 rtiles = (rows + 63u) >> 6u;
    u32 batch = blockIdx.y / rtiles;
    u32 r0 = (blockIdx.y - batch * rtiles) << 6u;
    u32 n0 = blockIdx.x << 6u;
    u64 x_sh = x_bf16 ? 1ull : 2ull;
    u64 o_sh = out_bf16 ? 1ull : 2ull;
    sf_p64_exec<TRANSPOSED>(x + ((u64(batch) * x_bstep + x_off) << x_sh),
                            w + u64(batch) * block + first,
                            out + ((u64(batch) * out_bstep) << o_sh), r0, n0, rows, n, k, w_stride,
                            x_rstep, out_rstep, x_bf16, out_bf16, sm);
}

// Cooperative 64x64x64 BF16 Tensor-Core stacked matmul with fused strides.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_sf_mma(
    const u8* x, const u16* w, u8* out, u32 rows, u32 n, u32 k, u32 block, u32 first, u32 col_step,
    u32 x_bstep, u32 x_rstep, u32 x_off, u32 out_bstep, u32 out_rstep, u32 x_bf16, u32 out_bf16) {
    __shared__ sf_p64_smem sm;
    sf_mma_entry<false>(x, w, out, rows, n, k, block, first, col_step, x_bstep, x_rstep, x_off,
                        out_bstep, out_rstep, x_bf16, out_bf16, sm);
}

// Cooperative 64x64x64 transposed BF16 Tensor-Core stacked matmul with fused strides.
extern "C" __global__ __launch_bounds__(256, 2) void tessera_sf_mma_t(
    const u8* x, const u16* w, u8* out, u32 rows, u32 n, u32 k, u32 block, u32 first, u32 k_step,
    u32 x_bstep, u32 x_rstep, u32 x_off, u32 out_bstep, u32 out_rstep, u32 x_bf16, u32 out_bf16) {
    __shared__ sf_p64_smem sm;
    sf_mma_entry<true>(x, w, out, rows, n, k, block, first, k_step, x_bstep, x_rstep, x_off,
                       out_bstep, out_rstep, x_bf16, out_bf16, sm);
}

static __device__ __forceinline__ void sf_p16_ld_x(const u8* xb, u32 r, u32 rows, u32 x_rstep,
                                                   u32 k0, u32 x8, u32 x_bf16, uint2& v0,
                                                   uint2& v1) {
    if (r < rows) {
        u64 xp = u64(r) * x_rstep + k0 + (x8 << 3u);
        v0 = sf_ld_xb4(xb, xp, x_bf16);
        v1 = sf_ld_xb4(xb, xp + 4u, x_bf16);
    } else {
        v0 = make_uint2(0u, 0u);
        v1 = make_uint2(0u, 0u);
    }
}

template <bool TRANSPOSED>
static __device__ __forceinline__ void sf_p16_exec(
    const u8* x, const u16* w, u8* out, u32 rows, u32 n, u32 k, u32 block, u32 first,
    u32 w_stride, u32 x_bstep, u32 x_rstep, u32 x_off, u32 out_bstep, u32 out_rstep, u32 x_bf16,
    u32 out_bf16, sf_p16_smem<TRANSPOSED ? 64u : 32u>& sm) {
    constexpr u32 BN = TRANSPOSED ? 64u : 32u;
    constexpr u32 NT = TRANSPOSED ? 2u : 1u;
    u32 rtiles = (rows + 15u) >> 4u;
    u32 batch = blockIdx.y / rtiles;
    u32 r0 = (blockIdx.y - batch * rtiles) << 4u;
    u32 n0 = blockIdx.x * BN;
    u64 x_sh = x_bf16 ? 1ull : 2ull;
    u64 o_sh = out_bf16 ? 1ull : 2ull;
    const u8* xb = x + ((u64(batch) * x_bstep + x_off) << x_sh);
    const u16* wb = w + u64(batch) * block + first;
    u8* ob = out + ((u64(batch) * out_bstep) << o_sh);
    u32 tid = threadIdx.x, warp = tid >> 5u, lane = tid & 31u;
    u32 xr = tid >> 3u, x8 = tid & 7u, kp = tid >> 4u, xl = tid & 15u;
    u32 wn = warp * (NT << 3u);
    u32 b_r = wn + (TRANSPOSED ? ((lane >> 4u) << 3u) : 0u) + (lane & 7u);
    const uint2* w2 = reinterpret_cast<const uint2*>(wb);
    float acc[NT][4] = {{0.f}};
    uint2 xv0, xv1;
    uint4 wn_v[2];
    uint2 wt_v[8];
    auto fetch = [&](u32 k0) {
        sf_p16_ld_x(xb, r0 + xr, rows, x_rstep, k0, x8, x_bf16, xv0, xv1);
        if constexpr (!TRANSPOSED) {
            wn_v[0] = (n0 + xr < n) ? sf_ld_nc4(wb + u64(n0 + xr) * w_stride + k0 + (x8 << 3u))
                                    : make_uint4(0u, 0u, 0u, 0u);
            wn_v[1] = (n0 + xr + 16u < n)
                          ? sf_ld_nc4(wb + u64(n0 + xr + 16u) * w_stride + k0 + (x8 << 3u))
                          : make_uint4(0u, 0u, 0u, 0u);
        } else {
            u32 col4 = (n0 >> 2u) + xl, step4 = w_stride >> 2u;
#pragma unroll
            for (u32 d = 0u; d < 4u; d++) {
                u32 ke = k0 + ((kp + (d << 3u)) << 1u);
                wt_v[d << 1u] = sf_ld_nc2(w2 + u64(ke) * step4 + col4);
                wt_v[(d << 1u) + 1u] = sf_ld_nc2(w2 + u64(ke + 1u) * step4 + col4);
            }
        }
    };
    auto commit = [&]() {
        *reinterpret_cast<uint4*>(&sm.x[xr][x8 << 2u]) = make_uint4(xv0.x, xv0.y, xv1.x, xv1.y);
        if constexpr (!TRANSPOSED) {
            *reinterpret_cast<uint4*>(&sm.w[xr][x8 << 2u]) = wn_v[0];
            *reinterpret_cast<uint4*>(&sm.w[xr + 16u][x8 << 2u]) = wn_v[1];
        } else {
#pragma unroll
            for (u32 d = 0u; d < 4u; d++) {
                sf_tr_pair(sm.w, xl << 2u, kp + (d << 3u), wt_v[d << 1u], wt_v[(d << 1u) + 1u]);
            }
        }
    };
    fetch(0u);
    commit();
    __syncthreads();
    for (u32 k0 = 0u; k0 < k; k0 += 64u) {
        if (k0 + 64u < k) fetch(k0 + 64u);
#pragma unroll
        for (u32 ks = 0u; ks < 4u; ks++) {
            u32 koff = ks << 3u, a[4];
            sf_ldm_x4(a, &sm.x[lane & 15u][koff + ((lane >> 4u) << 2u)]);
            if constexpr (!TRANSPOSED) {
                u32 b[2];
                sf_ldm_x2(b, &sm.w[b_r][koff + (((lane >> 3u) & 1u) << 2u)]);
                sf_mma_bf16(acc[0], a, b);
            } else {
                u32 b[4];
                sf_ldm_x4(b, &sm.w[b_r][koff + (((lane >> 3u) & 1u) << 2u)]);
                sf_mma_bf16(acc[0], a, b);
                sf_mma_bf16(acc[1], a, b + 2u);
            }
        }
        if (k0 + 64u < k) {
            __syncthreads();
            commit();
            __syncthreads();
        }
    }
#pragma unroll
    for (u32 ni = 0u; ni < NT; ni++) {
        sf_put_mma(ob, r0 + (lane >> 2u), n0 + wn + (ni << 3u) + ((lane & 3u) << 1u), rows, n,
                   out_rstep, acc[ni], out_bf16);
    }
}

// Cooperative 16x32x64 BF16 Tensor-Core stacked decode matmul.
extern "C" __global__ __launch_bounds__(128, 4) void tessera_sf_mma16(
    const u8* x, const u16* w, u8* out, u32 rows, u32 n, u32 k, u32 block, u32 first, u32 col_step,
    u32 x_bstep, u32 x_rstep, u32 x_off, u32 out_bstep, u32 out_rstep, u32 x_bf16, u32 out_bf16) {
    __shared__ sf_p16_smem<32u> sm;
    sf_p16_exec<false>(x, w, out, rows, n, k, block, first, col_step, x_bstep, x_rstep, x_off,
                       out_bstep, out_rstep, x_bf16, out_bf16, sm);
}

// Cooperative 16x64x64 transposed BF16 Tensor-Core stacked decode matmul.
extern "C" __global__ __launch_bounds__(128, 4) void tessera_sf_mma16_t(
    const u8* x, const u16* w, u8* out, u32 rows, u32 n, u32 k, u32 block, u32 first, u32 k_step,
    u32 x_bstep, u32 x_rstep, u32 x_off, u32 out_bstep, u32 out_rstep, u32 x_bf16, u32 out_bf16) {
    __shared__ sf_p16_smem<64u> sm;
    sf_p16_exec<true>(x, w, out, rows, n, k, block, first, k_step, x_bstep, x_rstep, x_off,
                      out_bstep, out_rstep, x_bf16, out_bf16, sm);
}
