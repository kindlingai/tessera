#include <tessera/1/core>
#include <tessera/1/trellis>

using namespace tessera;

#if defined(TESSERA_CUDA) || defined(TESSERA_HIP)

#if defined(TESSERA_HIP)
constexpr u32 DTR_TILE = 512u;
#elif !defined(DTR_FORCE_FP16) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define DTR_MX8 1
constexpr u32 DTR_TILE = trellis::MX8_TILE;
#else
#define DTR_MX8 0
constexpr u32 DTR_TILE = 512u;
#endif

constexpr u32 DTR_STRIDE = 8u * DTR_TILE;

static __device__ __forceinline__ void pf_store_frag(u32* tile, u32 row, u32 lane, bool valid,
                                                     const float v[4]) {
#if defined(TESSERA_HIP)
    typedef _Float16 f16x4 __attribute__((ext_vector_type(4)));
    _Float16(*sh)[8][16] = reinterpret_cast<_Float16(*)[8][16]>(tile);
    u32 u = lane >> 2u, kk = (lane & 3u) << 2u;
    *reinterpret_cast<f16x4*>(&sh[u][row][kk]) =
        valid ? (f16x4){static_cast<_Float16>(v[0]), static_cast<_Float16>(v[1]),
                        static_cast<_Float16>(v[2]), static_cast<_Float16>(v[3])}
              : (f16x4){static_cast<_Float16>(0.0f), static_cast<_Float16>(0.0f),
                        static_cast<_Float16>(0.0f), static_cast<_Float16>(0.0f)};
#elif DTR_MX8
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

static __device__ __forceinline__ void pf_write_v4(float* out, const u32* src, u64 slot_off,
                                                   u64 blk_off, u32 ti, u32 idx) {
#if defined(TESSERA_HIP)
    u32 su = idx >> 4u, sq = idx & 15u;
    u16* dst = reinterpret_cast<u16*>(out) + (slot_off + blk_off) * 2ull + u64(su) * 1024u +
               u64(ti) * 128u;
    reinterpret_cast<uint4*>(dst)[sq] = reinterpret_cast<const uint4*>(src)[idx];
#else
    uint4* dst =
        reinterpret_cast<uint4*>(reinterpret_cast<u32*>(out) + slot_off + blk_off + ti * DTR_TILE);
    dst[idx] = reinterpret_cast<const uint4*>(src)[idx];
#endif
}

static __device__ __forceinline__ void pf_load4(const u8* p, u32 col, bool bf16, float v[4]) {
    if (bf16) {
        uint2 w = *reinterpret_cast<const uint2*>(reinterpret_cast<const u16*>(p) + col);
        v[0] = __uint_as_float(w.x << 16u);
        v[1] = __uint_as_float(w.x & 0xFFFF0000u);
        v[2] = __uint_as_float(w.y << 16u);
        v[3] = __uint_as_float(w.y & 0xFFFF0000u);
    } else {
        trellis::load4(reinterpret_cast<const float*>(p) + col, v);
    }
}

static __device__ __forceinline__ float pf_warp_sum(float v) {
#pragma unroll
    for (u32 off = 16u; off > 0u; off >>= 1u) {
#if defined(TESSERA_HIP)
        v += __shfl_xor(v, static_cast<int>(off));
#else
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
#endif
    }
    return v;
}

// Fused norm and multi-output trellis input rotation per 8-row tile.
extern "C" __global__ __launch_bounds__(256, 4) void dtr_prep_fused(
    const float* x, const u8* w_norm, const u8* b_norm, const float* suh0, const float* suh1,
    const float* suh2, float* out, u32 m, u32 k, float eps, u32 count, u32 flags) {
    u32 r0 = blockIdx.y << 3u, nblk = k >> 7u;
    bool do_norm = (flags & 32u) != 0u;
    if (r0 >= m || (!do_norm && blockIdx.x >= nblk)) return;
    u32 tid = threadIdx.x, lane = tid & 31u, j = tid >> 5u, r = r0 + j;
    bool valid = r < m;
    bool has_w = (flags & 1u) != 0u, w_bf16 = (flags & 2u) != 0u;
    bool has_b = (flags & 4u) != 0u, b_bf16 = (flags & 8u) != 0u;
    bool center = (flags & 16u) != 0u, turned = (flags & 64u) != 0u;
    float mean = 0.0f, inv = 1.0f;
    if (do_norm && valid) {
        const float* xr = x + u64(r) * k + (lane << 2u);
        if (center) {
            float s = 0.0f;
            for (u32 c0 = 0u; c0 < nblk; c0++) {
                float v[4];
                trellis::load4(xr + (c0 << 7u), v);
                s += (v[0] + v[1]) + (v[2] + v[3]);
            }
            mean = pf_warp_sum(s) / float(k);
        }
        float sq = 0.0f;
        for (u32 c0 = 0u; c0 < nblk; c0++) {
            float v[4];
            trellis::load4(xr + (c0 << 7u), v);
            v[0] -= mean;
            v[1] -= mean;
            v[2] -= mean;
            v[3] -= mean;
            sq = fmaf(v[0], v[0], fmaf(v[1], v[1], fmaf(v[2], v[2], fmaf(v[3], v[3], sq))));
        }
        inv = rsqrtf(pf_warp_sum(sq) / float(k) + eps);
    }
    __shared__ alignas(16) u32 sm[2][DTR_TILE];
    u32 rt = blockIdx.y >> 3u, ti = blockIdx.y & 7u, buf = 0u;
    u64 slot_stride = u64((m + 63u) >> 6u) * nblk * DTR_STRIDE;
    constexpr u32 NV4 = DTR_TILE >> 2u;
    u32 c_start = do_norm ? 0u : blockIdx.x, c_end = do_norm ? nblk : (blockIdx.x + 1u);
    for (u32 c0 = c_start; c0 < c_end; c0++) {
        u32 col = (c0 << 7u) + (lane << 2u);
        float vn[4] = {};
        if (valid) {
            trellis::load4(x + u64(r) * k + col, vn);
            if (do_norm) {
                vn[0] = (vn[0] - mean) * inv;
                vn[1] = (vn[1] - mean) * inv;
                vn[2] = (vn[2] - mean) * inv;
                vn[3] = (vn[3] - mean) * inv;
                if (has_w) {
                    float w[4];
                    pf_load4(w_norm, col, w_bf16, w);
                    vn[0] *= w[0];
                    vn[1] *= w[1];
                    vn[2] *= w[2];
                    vn[3] *= w[3];
                }
                if (has_b) {
                    float b[4];
                    pf_load4(b_norm, col, b_bf16, b);
                    vn[0] += b[0];
                    vn[1] += b[1];
                    vn[2] += b[2];
                    vn[3] += b[3];
                }
            }
        }
        u64 blk_off = (u64(rt) * nblk + c0) * DTR_STRIDE;
        if (count == 2u) {
            float t0[4] = {}, t1[4] = {};
            if (valid) {
                float s0[4] = {1.0f, 1.0f, 1.0f, 1.0f}, s1[4] = {1.0f, 1.0f, 1.0f, 1.0f};
                if (turned) {
                    trellis::load4(suh0 + col, s0);
                    trellis::load4(suh1 + col, s1);
                }
                for (u32 i = 0u; i < 4u; i++) {
                    t0[i] = vn[i] * s0[i];
                    t1[i] = vn[i] * s1[i];
                }
                if (turned) {
                    trellis::wht128(t0, trellis::INV_SQRT128);
                    trellis::wht128(t1, trellis::INV_SQRT128);
                }
            }
            pf_store_frag(sm[0], j, lane, valid, t0);
            pf_store_frag(sm[1], j, lane, valid, t1);
            prim::block_sync();
            u32 s = tid >> 7u, idx = tid & 127u;
            if (idx < NV4) pf_write_v4(out, sm[s], u64(s) * slot_stride, blk_off, ti, idx);
            if (c0 + 1u < c_end) prim::block_sync();
            continue;
        }
        for (u32 s = 0u; s < count; s++) {
            const float* suh_p = (s == 0u) ? suh0 : ((s == 1u) ? suh1 : suh2);
            float t[4] = {};
            if (valid) {
                t[0] = vn[0];
                t[1] = vn[1];
                t[2] = vn[2];
                t[3] = vn[3];
                if (turned) {
                    float sc[4];
                    trellis::load4(suh_p + col, sc);
                    t[0] *= sc[0];
                    t[1] *= sc[1];
                    t[2] *= sc[2];
                    t[3] *= sc[3];
                    trellis::wht128(t, trellis::INV_SQRT128);
                }
            }
            pf_store_frag(sm[buf], j, lane, valid, t);
            prim::block_sync();
            if (tid < NV4) pf_write_v4(out, sm[buf], u64(s) * slot_stride, blk_off, ti, tid);
            buf ^= 1u;
        }
    }
}

#endif
